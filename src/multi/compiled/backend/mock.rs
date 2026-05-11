//! Mock device-style backend that executes on CPU while using device-oriented plans.

use crate::multi::compiled::backend::device::{
    DeviceBackend, DeviceBufferKind, DeviceBufferSet, DeviceExecutionMode, DeviceExecutionTrace,
    DeviceTransferKind,
};
use crate::multi::compiled::backend::dispatch::{BackendKind, ExecutionBackend};
use crate::multi::compiled::backend::scalar::ScalarBackend;
use crate::multi::compiled::backend::types::BackendCapabilities;
use crate::multi::compiled::{BatchGradientsBuffer, BatchInputs, BatchValuesBuffer, CompiledGraph};
use crate::{AutodiffError, Result};

/// Mock device-style backend that executes on CPU while using device-oriented plans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MockDeviceBackend;

impl DeviceBackend for MockDeviceBackend {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        BackendKind::MockDeviceCpu
    }
}

impl MockDeviceBackend {
    /// Allocate a mock-device buffer set for a compiled graph and batch size.
    #[must_use]
    pub fn allocate_batch_buffers(
        &self,
        graph: &CompiledGraph,
        batch_size: usize,
    ) -> DeviceBufferSet {
        DeviceBufferSet::new(self.batch_plan(graph, batch_size))
    }

    /// Execute batch value computation through explicit mock-device transfers.
    pub fn compute_batch_with_buffers(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        graph.check_batch(batch)?;
        if buffers.plan.backend != BackendKind::MockDeviceCpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "mock execution requires a mock-device buffer plan",
            });
        }
        if buffers.plan.batch_size != batch.batch_size || buffers.plan.input_dim != batch.input_dim
        {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch shape must match mock-device buffer plan",
            });
        }

        let transfers = buffers.plan.compute_transfer_plan.clone();
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::HostToDevice {
                if transfer.buffer != DeviceBufferKind::Inputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "mock compute supports host-to-device input transfers only",
                    });
                }
                buffers.upload(DeviceBufferKind::Inputs, batch.data)?;
            }
        }

        let device_inputs = buffers.download(DeviceBufferKind::Inputs)?;
        let device_batch = BatchInputs::new(&device_inputs, batch.batch_size, batch.input_dim)?;
        let mut scratch = BatchValuesBuffer::new();
        ScalarBackend.compute_batch(graph, device_batch, &mut scratch)?;
        buffers.upload(DeviceBufferKind::Outputs, &scratch.data)?;
        output.reset(batch.batch_size, graph.output_nodes.len());
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::DeviceToHost {
                if transfer.buffer != DeviceBufferKind::Outputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "mock compute supports device-to-host output transfers only",
                    });
                }
                output
                    .data
                    .extend_from_slice(&buffers.download(DeviceBufferKind::Outputs)?);
            }
        }

        Ok(DeviceExecutionTrace {
            backend: BackendKind::MockDeviceCpu,
            mode: DeviceExecutionMode::ComputeBatch,
            transfers,
            used_native_kernel: false,
        })
    }

    /// Execute batch gradient computation through explicit mock-device transfers.
    pub fn gradient_batch_with_buffers(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        graph.check_batch(batch)?;
        if buffers.plan.backend != BackendKind::MockDeviceCpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "mock execution requires a mock-device buffer plan",
            });
        }
        if buffers.plan.batch_size != batch.batch_size || buffers.plan.input_dim != batch.input_dim
        {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch shape must match mock-device buffer plan",
            });
        }

        let transfers = buffers.plan.gradient_transfer_plan.clone();
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::HostToDevice {
                if transfer.buffer != DeviceBufferKind::Inputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "mock gradient supports host-to-device input transfers only",
                    });
                }
                buffers.upload(DeviceBufferKind::Inputs, batch.data)?;
            }
        }

        let device_inputs = buffers.download(DeviceBufferKind::Inputs)?;
        let device_batch = BatchInputs::new(&device_inputs, batch.batch_size, batch.input_dim)?;
        let mut scratch = BatchGradientsBuffer::new();
        ScalarBackend.gradient_batch(graph, device_batch, &mut scratch)?;
        buffers.upload(DeviceBufferKind::PrimaryValues, &scratch.values)?;
        buffers.upload(DeviceBufferKind::Gradients, &scratch.gradients)?;
        output.reset(batch.batch_size, graph.num_inputs);
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::DeviceToHost {
                match transfer.buffer {
                    DeviceBufferKind::PrimaryValues => output
                        .values
                        .extend_from_slice(&buffers.download(DeviceBufferKind::PrimaryValues)?),
                    DeviceBufferKind::Gradients => output
                        .gradients
                        .extend_from_slice(&buffers.download(DeviceBufferKind::Gradients)?),
                    _ => {
                        return Err(AutodiffError::InvalidGraph {
                            reason:
                                "mock gradient supports primary-value and gradient downloads only",
                        });
                    }
                }
            }
        }

        Ok(DeviceExecutionTrace {
            backend: BackendKind::MockDeviceCpu,
            mode: DeviceExecutionMode::GradientBatch,
            transfers,
            used_native_kernel: false,
        })
    }
}
impl ExecutionBackend for MockDeviceBackend {
    fn name(&self) -> &'static str {
        "mock-device-cpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::scalar_f64()
    }

    fn compute(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<f64> {
        ScalarBackend.compute(graph, inputs)
    }

    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        ScalarBackend.compute_batch(graph, batch, buffer)
    }

    fn gradient(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        ScalarBackend.gradient(graph, inputs)
    }

    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        ScalarBackend.gradient_batch(graph, batch, buffer)
    }
}
