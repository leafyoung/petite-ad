//! Device-oriented buffer types, batch planning, and the DeviceBackend trait.

use crate::multi::compiled::backend::dispatch::BackendKind;
use crate::multi::compiled::CompiledGraph;
use crate::{AutodiffError, Result};

#[cfg(feature = "backend-wgpu")]
use crate::multi::compiled::backend::wgpu::WgpuBackend;

/// Batch memory layout exposed for backend/device planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchLayout {
    RowMajor,
}

/// Logical device buffer role for batch execution planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceBufferKind {
    Inputs,
    Values,
    Outputs,
    PrimaryValues,
    Gradients,
}

/// One logical device buffer description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBufferLayout {
    pub kind: DeviceBufferKind,
    pub len: usize,
    pub element_size: usize,
}

/// Logical memory location for a planned backend buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryLocation {
    Host,
    Device,
}

/// Handle-like description for one planned backend buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBufferHandle {
    pub kind: DeviceBufferKind,
    pub location: DeviceMemoryLocation,
    pub offset: usize,
    pub len: usize,
}

/// Logical transfer direction for backend execution planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTransferKind {
    HostToDevice,
    DeviceToHost,
}

/// One logical host/device transfer needed by a batch plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceTransferPlan {
    pub kind: DeviceTransferKind,
    pub buffer: DeviceBufferKind,
    pub len: usize,
}

/// Device buffer owned by a planned backend execution.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBuffer {
    handle: DeviceBufferHandle,
    data: Vec<f64>,
}

impl DeviceBuffer {
    fn new(handle: DeviceBufferHandle) -> Self {
        Self {
            handle,
            data: vec![0.0; handle.len],
        }
    }

    /// Return the immutable planned buffer handle.
    #[must_use]
    pub fn handle(&self) -> DeviceBufferHandle {
        self.handle
    }

    /// Return the owned buffer data.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

/// Owned buffer set allocated from a [`DeviceBatchPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBufferSet {
    pub(crate) plan: DeviceBatchPlan,
    buffers: Vec<DeviceBuffer>,
}

impl DeviceBufferSet {
    /// Allocate zero-initialized buffers for a batch plan.
    #[must_use]
    pub fn new(plan: DeviceBatchPlan) -> Self {
        let buffers = plan
            .buffer_handles
            .iter()
            .copied()
            .map(DeviceBuffer::new)
            .collect();
        Self { plan, buffers }
    }

    /// Return the plan used to allocate this buffer set.
    #[must_use]
    pub fn plan(&self) -> &DeviceBatchPlan {
        &self.plan
    }

    /// Return all allocated buffers in plan order.
    #[must_use]
    pub fn buffers(&self) -> &[DeviceBuffer] {
        &self.buffers
    }

    /// Return an immutable buffer by logical kind.
    pub fn buffer(&self, kind: DeviceBufferKind) -> Result<&DeviceBuffer> {
        self.buffers
            .iter()
            .find(|buffer| buffer.handle.kind == kind)
            .ok_or(AutodiffError::InvalidGraph {
                reason: "device buffer kind is not in the plan",
            })
    }

    /// Return a mutable buffer by logical kind.
    pub(crate) fn buffer_mut(&mut self, kind: DeviceBufferKind) -> Result<&mut DeviceBuffer> {
        self.buffers
            .iter_mut()
            .find(|buffer| buffer.handle.kind == kind)
            .ok_or(AutodiffError::InvalidGraph {
                reason: "device buffer kind is not in the plan",
            })
    }

    /// Upload host data into a planned buffer.
    pub fn upload(&mut self, kind: DeviceBufferKind, data: &[f64]) -> Result<()> {
        let buffer = self.buffer_mut(kind)?;
        if buffer.data.len() != data.len() {
            return Err(AutodiffError::InvalidGraph {
                reason: "upload length must match planned buffer length",
            });
        }
        buffer.data.copy_from_slice(data);
        Ok(())
    }

    /// Download a planned buffer into an owned vector.
    pub fn download(&self, kind: DeviceBufferKind) -> Result<Vec<f64>> {
        Ok(self.buffer(kind)?.data.clone())
    }
}

/// Device-oriented batch execution plan for a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBatchPlan {
    pub backend: BackendKind,
    pub layout: BatchLayout,
    pub batch_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub gradient_dim: usize,
    pub value_count: usize,
    pub buffers: Vec<DeviceBufferLayout>,
    pub buffer_handles: Vec<DeviceBufferHandle>,
    pub compute_transfer_plan: Vec<DeviceTransferPlan>,
    pub gradient_transfer_plan: Vec<DeviceTransferPlan>,
}

/// Batch execution mode for explicit device transfer planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceExecutionMode {
    ComputeBatch,
    GradientBatch,
}

/// Trace returned by explicit device-style execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceExecutionTrace {
    pub backend: BackendKind,
    pub mode: DeviceExecutionMode,
    pub transfers: Vec<DeviceTransferPlan>,
    /// Whether this execution used a native accelerator kernel instead of host fallback logic.
    pub used_native_kernel: bool,
}

/// Feature-neutral accelerator device family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorDeviceKind {
    MockCpu,
    Cuda,
    Wgpu,
}

/// Feature-neutral accelerator context descriptor for future GPU backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceleratorDeviceContext {
    pub kind: AcceleratorDeviceKind,
    pub device_id: usize,
    pub name: String,
}

impl AcceleratorDeviceContext {
    /// Return a context descriptor for the CPU mock-device backend.
    #[must_use]
    pub fn mock_cpu() -> Self {
        Self {
            kind: AcceleratorDeviceKind::MockCpu,
            device_id: 0,
            name: "mock-device-cpu".to_string(),
        }
    }

    /// Return a context descriptor for a future CUDA backend.
    #[must_use]
    pub fn cuda(device_id: usize) -> Self {
        Self {
            kind: AcceleratorDeviceKind::Cuda,
            device_id,
            name: format!("cuda:{device_id}"),
        }
    }

    /// Return a context descriptor for a future WGPU backend.
    #[must_use]
    pub fn wgpu(device_id: usize) -> Self {
        Self {
            kind: AcceleratorDeviceKind::Wgpu,
            device_id,
            name: format!("wgpu:{device_id}"),
        }
    }
}

/// Host/device transfer policy for future accelerator backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTransferPolicy {
    Explicit,
    Automatic,
}

/// Boundary object for future GPU backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBackendBoundary {
    pub context: AcceleratorDeviceContext,
    pub transfer_policy: DeviceTransferPolicy,
}

impl GpuBackendBoundary {
    /// Create a boundary descriptor for a future accelerator backend.
    #[must_use]
    pub fn new(context: AcceleratorDeviceContext, transfer_policy: DeviceTransferPolicy) -> Self {
        Self {
            context,
            transfer_policy,
        }
    }

    #[cfg(feature = "backend-wgpu")]
    /// Initialize the real WGPU backend skeleton from this boundary.
    pub fn initialize_wgpu(&self) -> Result<WgpuBackend> {
        WgpuBackend::from_boundary(self.clone())
    }

    /// Return an explicit error because generic GPU execution still requires a concrete backend.
    pub fn unsupported_execution_error<T>(&self) -> Result<T> {
        Err(AutodiffError::InvalidGraph {
            reason:
                "real GPU execution requires a concrete backend instance; initialize WgpuBackend",
        })
    }
}

/// Device-oriented batch backend planning interface.
pub trait DeviceBackend {
    fn backend_kind(&self, graph: &CompiledGraph) -> BackendKind;

    fn batch_layout(&self) -> BatchLayout {
        BatchLayout::RowMajor
    }

    fn batch_plan(&self, graph: &CompiledGraph, batch_size: usize) -> DeviceBatchPlan {
        let metadata = graph.metadata();
        let value_count = metadata.value_count.saturating_mul(batch_size);
        let input_count = metadata.num_inputs.saturating_mul(batch_size);
        let output_count = metadata.num_outputs.saturating_mul(batch_size);
        let gradient_count = metadata.num_inputs.saturating_mul(batch_size);
        let backend = self.backend_kind(graph);
        let buffer_location = backend.memory_location();
        let buffers = vec![
            DeviceBufferLayout {
                kind: DeviceBufferKind::Inputs,
                len: input_count,
                element_size: std::mem::size_of::<f64>(),
            },
            DeviceBufferLayout {
                kind: DeviceBufferKind::Values,
                len: value_count,
                element_size: std::mem::size_of::<f64>(),
            },
            DeviceBufferLayout {
                kind: DeviceBufferKind::Outputs,
                len: output_count,
                element_size: std::mem::size_of::<f64>(),
            },
            DeviceBufferLayout {
                kind: DeviceBufferKind::PrimaryValues,
                len: batch_size,
                element_size: std::mem::size_of::<f64>(),
            },
            DeviceBufferLayout {
                kind: DeviceBufferKind::Gradients,
                len: gradient_count,
                element_size: std::mem::size_of::<f64>(),
            },
        ];
        let mut offset = 0;
        let buffer_handles = buffers
            .iter()
            .map(|buffer| {
                let handle = DeviceBufferHandle {
                    kind: buffer.kind,
                    location: buffer_location,
                    offset,
                    len: buffer.len,
                };
                offset = offset.saturating_add(buffer.len);
                handle
            })
            .collect();
        let (compute_transfer_plan, gradient_transfer_plan) =
            if buffer_location == DeviceMemoryLocation::Device {
                (
                    vec![
                        DeviceTransferPlan {
                            kind: DeviceTransferKind::HostToDevice,
                            buffer: DeviceBufferKind::Inputs,
                            len: input_count,
                        },
                        DeviceTransferPlan {
                            kind: DeviceTransferKind::DeviceToHost,
                            buffer: DeviceBufferKind::Outputs,
                            len: output_count,
                        },
                    ],
                    vec![
                        DeviceTransferPlan {
                            kind: DeviceTransferKind::HostToDevice,
                            buffer: DeviceBufferKind::Inputs,
                            len: input_count,
                        },
                        DeviceTransferPlan {
                            kind: DeviceTransferKind::DeviceToHost,
                            buffer: DeviceBufferKind::PrimaryValues,
                            len: batch_size,
                        },
                        DeviceTransferPlan {
                            kind: DeviceTransferKind::DeviceToHost,
                            buffer: DeviceBufferKind::Gradients,
                            len: gradient_count,
                        },
                    ],
                )
            } else {
                (Vec::new(), Vec::new())
            };
        DeviceBatchPlan {
            backend,
            layout: self.batch_layout(),
            batch_size,
            input_dim: metadata.num_inputs,
            output_dim: metadata.num_outputs,
            gradient_dim: metadata.num_inputs,
            value_count: metadata.value_count,
            buffers,
            buffer_handles,
            compute_transfer_plan,
            gradient_transfer_plan,
        }
    }
}
