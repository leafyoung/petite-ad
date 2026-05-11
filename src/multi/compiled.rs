//! Closure-free compiled instruction IR for acceleration-ready graph execution.

use super::multi_ad::MultiAD;
use super::op_rules;
use crate::{AutodiffError, NodeId, Result};

#[cfg(feature = "backend-wgpu")]
use std::sync::mpsc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128d, __m256d};

#[cfg(feature = "backend-wgpu")]
use pollster::block_on;
#[cfg(feature = "backend-wgpu")]
use wgpu::{self, BufferUsages};

/// One closure-free instruction in a compiled scalar graph.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    /// Literal constant node.
    Constant(f64),
    /// Unary operation with one input node id.
    Unary { op: MultiAD, arg: NodeId },
    /// Binary operation with two input node ids.
    Binary {
        op: MultiAD,
        left: NodeId,
        right: NodeId,
    },
}

/// Sentinel node id used by flat backend instructions when an argument is absent.
pub const UNUSED_NODE_ID: NodeId = usize::MAX;

/// Compact operation code used by flat backend instructions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    Constant,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sin,
    Cos,
    Tan,
    Tanh,
    Relu,
    Log1pExp,
    LogAddExp,
    Neg,
    Exp,
    Ln,
    Sqrt,
    Abs,
}

impl OpCode {
    fn from_multi_ad(op: MultiAD) -> Result<Self> {
        Ok(match op {
            MultiAD::Add => OpCode::Add,
            MultiAD::Sub => OpCode::Sub,
            MultiAD::Mul => OpCode::Mul,
            MultiAD::Div => OpCode::Div,
            MultiAD::Pow => OpCode::Pow,
            MultiAD::Sin => OpCode::Sin,
            MultiAD::Cos => OpCode::Cos,
            MultiAD::Tan => OpCode::Tan,
            MultiAD::Tanh => OpCode::Tanh,
            MultiAD::Relu => OpCode::Relu,
            MultiAD::Log1pExp => OpCode::Log1pExp,
            MultiAD::LogAddExp => OpCode::LogAddExp,
            MultiAD::Neg => OpCode::Neg,
            MultiAD::Exp => OpCode::Exp,
            MultiAD::Ln => OpCode::Ln,
            MultiAD::Sqrt => OpCode::Sqrt,
            MultiAD::Abs => OpCode::Abs,
            MultiAD::Inp => {
                return Err(AutodiffError::InvalidGraph {
                    reason: "input markers are not backend instructions",
                });
            }
        })
    }

    fn to_multi_ad(self) -> Option<MultiAD> {
        Some(match self {
            OpCode::Constant => return None,
            OpCode::Add => MultiAD::Add,
            OpCode::Sub => MultiAD::Sub,
            OpCode::Mul => MultiAD::Mul,
            OpCode::Div => MultiAD::Div,
            OpCode::Pow => MultiAD::Pow,
            OpCode::Sin => MultiAD::Sin,
            OpCode::Cos => MultiAD::Cos,
            OpCode::Tan => MultiAD::Tan,
            OpCode::Tanh => MultiAD::Tanh,
            OpCode::Relu => MultiAD::Relu,
            OpCode::Log1pExp => MultiAD::Log1pExp,
            OpCode::LogAddExp => MultiAD::LogAddExp,
            OpCode::Neg => MultiAD::Neg,
            OpCode::Exp => MultiAD::Exp,
            OpCode::Ln => MultiAD::Ln,
            OpCode::Sqrt => MultiAD::Sqrt,
            OpCode::Abs => MultiAD::Abs,
        })
    }

    /// Return the number of input slots consumed by this opcode.
    #[must_use]
    pub fn arity(&self) -> usize {
        match self {
            OpCode::Constant => 0,
            OpCode::Sin
            | OpCode::Cos
            | OpCode::Tan
            | OpCode::Tanh
            | OpCode::Relu
            | OpCode::Log1pExp
            | OpCode::Neg
            | OpCode::Exp
            | OpCode::Ln
            | OpCode::Sqrt
            | OpCode::Abs => 1,
            OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Pow
            | OpCode::LogAddExp => 2,
        }
    }
}

/// Fully explicit instruction shape for SIMD/GPU backend lowering.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatInstruction {
    pub opcode: OpCode,
    pub output: NodeId,
    pub left: NodeId,
    pub right: NodeId,
    pub value: f64,
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn supports_simd_f64x2_runtime() -> bool {
    true
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn supports_simd_f64x2_runtime() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn supports_simd_f64x4_runtime() -> bool {
    std::is_x86_feature_detected!("avx")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn supports_simd_f64x4_runtime() -> bool {
    false
}

#[inline]
fn supports_wgpu_runtime() -> bool {
    cfg!(feature = "backend-wgpu")
}

/// Backend feature declaration used before dispatching compiled graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub supports_f64: bool,
    pub supports_f32: bool,
    pub supports_constants: bool,
    pub supports_unary: bool,
    pub supports_binary: bool,
    pub supports_multi_output: bool,
    pub supports_reverse_gradient: bool,
    pub supports_batch_compute: bool,
    pub supports_batch_gradient: bool,
    pub supported_opcodes: Vec<OpCode>,
}

impl BackendCapabilities {
    /// Capabilities for the reference scalar f64 backend.
    #[must_use]
    pub fn scalar_f64() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: true,
            supports_batch_compute: true,
            supports_batch_gradient: true,
            supported_opcodes: vec![
                OpCode::Constant,
                OpCode::Add,
                OpCode::Sub,
                OpCode::Mul,
                OpCode::Div,
                OpCode::Pow,
                OpCode::Sin,
                OpCode::Cos,
                OpCode::Tan,
                OpCode::Tanh,
                OpCode::Relu,
                OpCode::Log1pExp,
                OpCode::LogAddExp,
                OpCode::Neg,
                OpCode::Exp,
                OpCode::Ln,
                OpCode::Sqrt,
                OpCode::Abs,
            ],
        }
    }

    /// Capabilities for the prototype f64x2 SIMD batch backend.
    #[must_use]
    pub fn simd_f64x2() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: false,
            supports_batch_compute: supports_simd_f64x2_runtime(),
            supports_batch_gradient: supports_simd_f64x2_runtime(),
            supported_opcodes: vec![
                OpCode::Constant,
                OpCode::Add,
                OpCode::Sub,
                OpCode::Mul,
                OpCode::Div,
                OpCode::Pow,
                OpCode::Sin,
                OpCode::Cos,
                OpCode::Tan,
                OpCode::Tanh,
                OpCode::Log1pExp,
                OpCode::LogAddExp,
                OpCode::Neg,
                OpCode::Exp,
                OpCode::Ln,
                OpCode::Sqrt,
                OpCode::Relu,
                OpCode::Abs,
            ],
        }
    }

    /// Capabilities for the prototype f64x4 SIMD batch backend.
    #[must_use]
    pub fn simd_f64x4() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: false,
            supports_batch_compute: supports_simd_f64x4_runtime(),
            supports_batch_gradient: supports_simd_f64x4_runtime(),
            supported_opcodes: vec![
                OpCode::Constant,
                OpCode::Add,
                OpCode::Sub,
                OpCode::Mul,
                OpCode::Div,
                OpCode::Pow,
                OpCode::Sin,
                OpCode::Cos,
                OpCode::Tan,
                OpCode::Tanh,
                OpCode::Log1pExp,
                OpCode::LogAddExp,
                OpCode::Neg,
                OpCode::Exp,
                OpCode::Ln,
                OpCode::Sqrt,
                OpCode::Relu,
                OpCode::Abs,
            ],
        }
    }

    /// Capabilities for the feature-gated WGPU batch backend skeleton.
    #[must_use]
    pub fn wgpu_f64() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: true,
            supports_batch_compute: supports_wgpu_runtime(),
            supports_batch_gradient: supports_wgpu_runtime(),
            supported_opcodes: BackendCapabilities::scalar_f64().supported_opcodes,
        }
    }

    /// Return whether the backend declares support for an opcode.
    #[must_use]
    pub fn supports_opcode(&self, opcode: OpCode) -> bool {
        self.supported_opcodes.contains(&opcode)
    }
}

/// Reference scalar backend for the execution-backend abstraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScalarBackend;

/// Mock device-style backend that executes on CPU while using device-oriented plans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MockDeviceBackend;

/// Prototype f64x2 SIMD backend for batch compute and batch gradients.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimdBackend;

/// Backend selected by automatic compiled-graph dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Scalar,
    MockDeviceCpu,
    Wgpu,
    SimdF64x4,
    SimdF64x2,
}

impl BackendKind {
    /// Return the number of f64 lanes processed by this backend.
    #[must_use]
    pub fn lane_width(&self) -> usize {
        match self {
            BackendKind::Scalar | BackendKind::MockDeviceCpu | BackendKind::Wgpu => 1,
            BackendKind::SimdF64x4 => 4,
            BackendKind::SimdF64x2 => 2,
        }
    }

    /// Return runtime CPU features required by this backend.
    #[must_use]
    pub fn required_runtime_features(&self) -> &'static [&'static str] {
        match self {
            BackendKind::Scalar | BackendKind::MockDeviceCpu => &[],
            BackendKind::Wgpu => &["backend-wgpu"],
            BackendKind::SimdF64x4 => &["x86_64-avx"],
            BackendKind::SimdF64x2 => &["x86_64-sse2"],
        }
    }

    /// Return whether the backend is available on this runtime target.
    #[must_use]
    pub fn runtime_available(&self) -> bool {
        match self {
            BackendKind::Scalar | BackendKind::MockDeviceCpu => true,
            BackendKind::Wgpu => supports_wgpu_runtime(),
            BackendKind::SimdF64x4 => supports_simd_f64x4_runtime(),
            BackendKind::SimdF64x2 => supports_simd_f64x2_runtime(),
        }
    }

    /// Return runtime CPU features that are required but unavailable.
    #[must_use]
    pub fn unavailable_runtime_features(&self) -> Vec<&'static str> {
        if self.runtime_available() {
            Vec::new()
        } else {
            self.required_runtime_features().to_vec()
        }
    }

    /// Return a stable backend name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            BackendKind::Scalar => "scalar",
            BackendKind::MockDeviceCpu => "mock-device-cpu",
            BackendKind::Wgpu => "wgpu",
            BackendKind::SimdF64x4 => "simd-f64x4",
            BackendKind::SimdF64x2 => "simd-f64x2",
        }
    }

    /// Return the logical memory location used by this backend's batch plan.
    #[must_use]
    pub fn memory_location(&self) -> DeviceMemoryLocation {
        match self {
            BackendKind::MockDeviceCpu | BackendKind::Wgpu => DeviceMemoryLocation::Device,
            BackendKind::Scalar | BackendKind::SimdF64x4 | BackendKind::SimdF64x2 => {
                DeviceMemoryLocation::Host
            }
        }
    }

    /// Return declared capabilities for this backend.
    #[must_use]
    pub fn capabilities(&self) -> BackendCapabilities {
        match self {
            BackendKind::Scalar | BackendKind::MockDeviceCpu => BackendCapabilities::scalar_f64(),
            BackendKind::Wgpu => BackendCapabilities::wgpu_f64(),
            BackendKind::SimdF64x4 => BackendCapabilities::simd_f64x4(),
            BackendKind::SimdF64x2 => BackendCapabilities::simd_f64x2(),
        }
    }

    /// Execute batch value computation with this backend.
    pub fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        if matches!(self, BackendKind::Scalar | BackendKind::MockDeviceCpu) {
            return ScalarBackend.compute_batch(graph, batch, buffer);
        }
        if matches!(self, BackendKind::Wgpu) {
            #[cfg(feature = "backend-wgpu")]
            {
                return WgpuBackend::new_default()?.compute_batch(graph, batch, buffer);
            }
            #[cfg(not(feature = "backend-wgpu"))]
            {
                return Err(AutodiffError::InvalidGraph {
                    reason: "wgpu backend requires the backend-wgpu cargo feature",
                });
            }
        }
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_compute {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch compute on this target",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        match self {
            BackendKind::Scalar | BackendKind::MockDeviceCpu | BackendKind::Wgpu => unreachable!(),
            BackendKind::SimdF64x4 => compute_batch_simd_f64x4(graph, batch, buffer),
            BackendKind::SimdF64x2 => compute_batch_simd_f64x2(graph, batch, buffer),
        }
    }

    /// Execute batch gradient computation with this backend.
    pub fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        if matches!(self, BackendKind::Scalar | BackendKind::MockDeviceCpu) {
            return ScalarBackend.gradient_batch(graph, batch, buffer);
        }
        if matches!(self, BackendKind::Wgpu) {
            #[cfg(feature = "backend-wgpu")]
            {
                return WgpuBackend::new_default()?.gradient_batch(graph, batch, buffer);
            }
            #[cfg(not(feature = "backend-wgpu"))]
            {
                return Err(AutodiffError::InvalidGraph {
                    reason: "wgpu backend requires the backend-wgpu cargo feature",
                });
            }
        }
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_gradient {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch gradients on this target",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        match self {
            BackendKind::Scalar | BackendKind::MockDeviceCpu | BackendKind::Wgpu => unreachable!(),
            BackendKind::SimdF64x4 => gradient_batch_simd_f64x4(graph, batch, buffer),
            BackendKind::SimdF64x2 => gradient_batch_simd_f64x2(graph, batch, buffer),
        }
    }
}

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
    plan: DeviceBatchPlan,
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
    fn buffer_mut(&mut self, kind: DeviceBufferKind) -> Result<&mut DeviceBuffer> {
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

#[cfg(feature = "backend-wgpu")]
/// Initialized WGPU backend skeleton with real device allocation and transfer plumbing.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    boundary: GpuBackendBoundary,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

#[cfg(feature = "backend-wgpu")]
/// One real WGPU buffer allocated for a logical batch-plan role.
#[derive(Debug, Clone)]
pub struct WgpuBuffer {
    handle: DeviceBufferHandle,
    buffer: wgpu::Buffer,
}

#[cfg(feature = "backend-wgpu")]
impl WgpuBuffer {
    /// Return the immutable planned buffer handle.
    #[must_use]
    pub fn handle(&self) -> DeviceBufferHandle {
        self.handle
    }
}

#[cfg(feature = "backend-wgpu")]
/// Owned WGPU buffers allocated from a [`DeviceBatchPlan`].
#[derive(Debug, Clone)]
pub struct WgpuBufferSet {
    plan: DeviceBatchPlan,
    buffers: Vec<WgpuBuffer>,
}

#[cfg(feature = "backend-wgpu")]
impl WgpuBufferSet {
    /// Return the plan used to allocate this buffer set.
    #[must_use]
    pub fn plan(&self) -> &DeviceBatchPlan {
        &self.plan
    }

    /// Return all allocated GPU buffers in plan order.
    #[must_use]
    pub fn buffers(&self) -> &[WgpuBuffer] {
        &self.buffers
    }

    /// Return an immutable GPU buffer by logical kind.
    pub fn buffer(&self, kind: DeviceBufferKind) -> Result<&WgpuBuffer> {
        self.buffers
            .iter()
            .find(|buffer| buffer.handle.kind == kind)
            .ok_or(AutodiffError::InvalidGraph {
                reason: "wgpu buffer kind is not in the plan",
            })
    }

    /// Upload host data into a planned GPU buffer.
    pub fn upload(
        &self,
        backend: &WgpuBackend,
        kind: DeviceBufferKind,
        data: &[f64],
    ) -> Result<()> {
        backend.upload_buffer(self, kind, data)
    }

    /// Download a planned GPU buffer into an owned vector.
    pub fn download(&self, backend: &WgpuBackend, kind: DeviceBufferKind) -> Result<Vec<f64>> {
        backend.download_buffer(self, kind)
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

/// Reason a backend cannot execute a graph for a requested batch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendRejectionReason {
    MissingF64,
    UnsupportedOutputs,
    UnsupportedOpcodes,
    UnsupportedArities,
    UnavailableRuntime,
    NoBatchCompute,
    NoBatchGradient,
}

/// Static backend compatibility report for a compiled graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSupportReport {
    pub backend: BackendKind,
    pub supports_f64: bool,
    pub supports_required_outputs: bool,
    pub supports_required_opcodes: bool,
    pub supports_required_arities: bool,
    pub supports_batch_compute: bool,
    pub supports_batch_gradient: bool,
    pub lane_width: usize,
    pub runtime_available: bool,
    pub required_runtime_features: Vec<&'static str>,
    pub unavailable_runtime_features: Vec<&'static str>,
    pub missing_opcodes: Vec<OpCode>,
    pub batch_compute_rejection_reasons: Vec<BackendRejectionReason>,
    pub batch_gradient_rejection_reasons: Vec<BackendRejectionReason>,
}

impl BackendSupportReport {
    /// Return whether this backend can compute batch values for the graph.
    #[must_use]
    pub fn can_compute_batch(&self) -> bool {
        self.batch_compute_rejection_reasons.is_empty()
    }

    /// Return whether this backend can compute batch gradients for the graph.
    #[must_use]
    pub fn can_gradient_batch(&self) -> bool {
        self.batch_gradient_rejection_reasons.is_empty()
    }
}

/// Common execution-backend interface for future scalar, SIMD, and GPU backends.
pub trait ExecutionBackend {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> BackendCapabilities;

    fn compute(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<f64>;
    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()>;
    fn gradient(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<(f64, Vec<f64>)>;
    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()>;
}

/// Flat row-major batch input view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchInputs<'a> {
    /// Flat row-major input data.
    pub data: &'a [f64],
    /// Number of rows in the batch.
    pub batch_size: usize,
    /// Number of inputs per row.
    pub input_dim: usize,
}

impl<'a> BatchInputs<'a> {
    /// Create a validated row-major input view.
    pub fn new(data: &'a [f64], batch_size: usize, input_dim: usize) -> Result<Self> {
        if data.len() != batch_size.saturating_mul(input_dim) {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch data length must equal batch_size * input_dim",
            });
        }
        Ok(Self {
            data,
            batch_size,
            input_dim,
        })
    }

    /// Return one input row.
    ///
    /// Prefer [`BatchInputs::try_row`] when invalid row indices should return an error.
    #[must_use]
    pub fn row(&self, index: usize) -> &[f64] {
        self.try_row(index).expect("batch row index out of bounds")
    }

    /// Return one input row, or an error when `index` is out of range.
    pub fn try_row(&self, index: usize) -> Result<&[f64]> {
        if index >= self.batch_size {
            return Err(AutodiffError::IndexOutOfBounds {
                index,
                max_index: self.batch_size.saturating_sub(1),
            });
        }
        let start = index * self.input_dim;
        Ok(&self.data[start..start + self.input_dim])
    }
}

/// Flat row-major batch values.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchValues {
    pub data: Vec<f64>,
    pub batch_size: usize,
    pub output_dim: usize,
}

/// Reusable flat row-major batch value buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchValuesBuffer {
    pub data: Vec<f64>,
    pub batch_size: usize,
    pub output_dim: usize,
}

impl BatchValuesBuffer {
    /// Create an empty reusable output buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self, batch_size: usize, output_dim: usize) {
        self.data.clear();
        self.data.reserve(batch_size.saturating_mul(output_dim));
        self.batch_size = batch_size;
        self.output_dim = output_dim;
    }

    /// Clone the current buffer contents into an owned result value.
    #[must_use]
    pub fn to_values(&self) -> BatchValues {
        BatchValues {
            data: self.data.clone(),
            batch_size: self.batch_size,
            output_dim: self.output_dim,
        }
    }
}

/// Flat row-major batch scalar-output gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchGradients {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
    pub batch_size: usize,
    pub input_dim: usize,
}

/// Reusable flat row-major batch scalar-output gradient buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchGradientsBuffer {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
    pub batch_size: usize,
    pub input_dim: usize,
}

impl BatchGradientsBuffer {
    /// Create an empty reusable gradient buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self, batch_size: usize, input_dim: usize) {
        self.values.clear();
        self.gradients.clear();
        self.values.reserve(batch_size);
        self.gradients.reserve(batch_size.saturating_mul(input_dim));
        self.batch_size = batch_size;
        self.input_dim = input_dim;
    }

    /// Clone the current buffer contents into an owned result value.
    #[must_use]
    pub fn to_gradients(&self) -> BatchGradients {
        BatchGradients {
            values: self.values.clone(),
            gradients: self.gradients.clone(),
            batch_size: self.batch_size,
            input_dim: self.input_dim,
        }
    }
}

/// Static metadata for backend selection and workspace planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledGraphMetadata {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_instructions: usize,
    pub num_constants: usize,
    pub num_unary: usize,
    pub num_binary: usize,
    pub value_count: usize,
    pub is_scalar_output: bool,
}

/// Reusable buffers for [`CompiledGraph`] execution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompiledWorkspace {
    values: Vec<f64>,
    cotangents: Vec<f64>,
    gradients: Vec<f64>,
    outputs: Vec<f64>,
}

/// Closure-free compiled graph representation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGraph {
    pub(crate) num_inputs: usize,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) output_nodes: Vec<NodeId>,
    flat_instructions: Vec<FlatInstruction>,
}

impl CompiledGraph {
    pub(crate) fn new(
        num_inputs: usize,
        instructions: Vec<Instruction>,
        output_nodes: Vec<NodeId>,
    ) -> Result<Self> {
        let flat_instructions = Self::lower_flat_instructions(num_inputs, &instructions)?;
        Ok(Self {
            num_inputs,
            instructions,
            output_nodes,
            flat_instructions,
        })
    }

    /// Return number of graph inputs.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Return compiled instructions.
    #[must_use]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Return selected output nodes.
    #[must_use]
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }

    fn lower_flat_instructions(
        num_inputs: usize,
        instructions: &[Instruction],
    ) -> Result<Vec<FlatInstruction>> {
        let mut flat = Vec::with_capacity(instructions.len());
        for (offset, instruction) in instructions.iter().enumerate() {
            let output = num_inputs + offset;
            let item = match *instruction {
                Instruction::Constant(value) => FlatInstruction {
                    opcode: OpCode::Constant,
                    output,
                    left: UNUSED_NODE_ID,
                    right: UNUSED_NODE_ID,
                    value,
                },
                Instruction::Unary { op, arg } => FlatInstruction {
                    opcode: OpCode::from_multi_ad(op)?,
                    output,
                    left: arg,
                    right: UNUSED_NODE_ID,
                    value: 0.0,
                },
                Instruction::Binary { op, left, right } => FlatInstruction {
                    opcode: OpCode::from_multi_ad(op)?,
                    output,
                    left,
                    right,
                    value: 0.0,
                },
            };
            flat.push(item);
        }
        Ok(flat)
    }

    /// Return a zero-copy view of flat instructions with explicit output and argument slots.
    #[must_use]
    pub fn flat_instructions_slice(&self) -> &[FlatInstruction] {
        &self.flat_instructions
    }

    /// Return owned flat instructions with explicit output and argument slots.
    pub fn flat_instructions(&self) -> Result<Vec<FlatInstruction>> {
        Ok(self.flat_instructions_slice().to_vec())
    }

    /// Validate whether static graph requirements fit backend capabilities.
    pub fn validate_backend_capabilities(&self, capabilities: &BackendCapabilities) -> Result<()> {
        if !capabilities.supports_f64 {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend must support f64 execution",
            });
        }
        if self.output_nodes.len() > 1 && !capabilities.supports_multi_output {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support multi-output graphs",
            });
        }
        for instruction in self.flat_instructions_slice() {
            if !capabilities.supports_opcode(instruction.opcode) {
                return Err(AutodiffError::InvalidGraph {
                    reason: "backend does not support a required opcode",
                });
            }
            match instruction.opcode.arity() {
                0 if !capabilities.supports_constants => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "backend does not support constants",
                    });
                }
                1 if !capabilities.supports_unary => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "backend does not support unary operations",
                    });
                }
                2 if !capabilities.supports_binary => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "backend does not support binary operations",
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Return static execution metadata for backend planning.
    #[must_use]
    pub fn metadata(&self) -> CompiledGraphMetadata {
        let mut num_constants = 0;
        let mut num_unary = 0;
        let mut num_binary = 0;
        for instruction in &self.instructions {
            match instruction {
                Instruction::Constant(_) => num_constants += 1,
                Instruction::Unary { .. } => num_unary += 1,
                Instruction::Binary { .. } => num_binary += 1,
            }
        }
        CompiledGraphMetadata {
            num_inputs: self.num_inputs,
            num_outputs: self.output_nodes.len(),
            num_instructions: self.instructions.len(),
            num_constants,
            num_unary,
            num_binary,
            value_count: self.num_inputs + self.instructions.len(),
            is_scalar_output: self.output_nodes.len() == 1,
        }
    }

    /// Create a reusable workspace sized for this compiled graph.
    #[must_use]
    pub fn workspace(&self) -> CompiledWorkspace {
        let value_count = self.num_inputs + self.instructions.len();
        CompiledWorkspace {
            values: Vec::with_capacity(value_count),
            cotangents: Vec::with_capacity(value_count),
            gradients: Vec::with_capacity(self.num_inputs),
            outputs: Vec::with_capacity(self.output_nodes.len()),
        }
    }

    fn check_input_len(&self, inputs: &[f64]) -> Result<()> {
        if inputs.len() == self.num_inputs {
            Ok(())
        } else {
            Err(AutodiffError::InvalidGraph {
                reason: "input length must match compiled graph input count",
            })
        }
    }

    fn check_batch(&self, batch: BatchInputs<'_>) -> Result<()> {
        if batch.input_dim == self.num_inputs {
            Ok(())
        } else {
            Err(AutodiffError::InvalidGraph {
                reason: "batch input_dim must match compiled graph input count",
            })
        }
    }

    fn gather_unary(values: &[f64], arg: NodeId) -> Result<[f64; 1]> {
        let value = values
            .get(arg)
            .copied()
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: arg,
                max_index: values.len().saturating_sub(1),
            })?;
        Ok([value])
    }

    fn gather_binary(values: &[f64], left: NodeId, right: NodeId) -> Result<[f64; 2]> {
        let left_value = values
            .get(left)
            .copied()
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: left,
                max_index: values.len().saturating_sub(1),
            })?;
        let right_value = values
            .get(right)
            .copied()
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: right,
                max_index: values.len().saturating_sub(1),
            })?;
        Ok([left_value, right_value])
    }

    fn fill_values(&self, inputs: &[f64], workspace: &mut CompiledWorkspace) -> Result<()> {
        self.check_input_len(inputs)?;
        workspace.values.clear();
        workspace
            .values
            .reserve(self.num_inputs + self.instructions.len());
        workspace.values.extend_from_slice(inputs);

        for instruction in &self.instructions {
            let value = match *instruction {
                Instruction::Constant(value) => value,
                Instruction::Unary { op, arg } => {
                    let args = Self::gather_unary(&workspace.values, arg)?;
                    op_rules::forward_value(op, &args)?
                }
                Instruction::Binary { op, left, right } => {
                    let args = Self::gather_binary(&workspace.values, left, right)?;
                    op_rules::forward_value(op, &args)?
                }
            };
            workspace.values.push(value);
        }
        Ok(())
    }

    /// Compute primary output.
    pub fn compute(&self, inputs: &[f64]) -> Result<f64> {
        let mut workspace = self.workspace();
        self.compute_with_workspace(inputs, &mut workspace)
    }

    /// Compute all selected outputs.
    pub fn compute_many(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        let mut workspace = self.workspace();
        Ok(self
            .compute_many_with_workspace(inputs, &mut workspace)?
            .to_vec())
    }

    /// Compute primary output with a reusable workspace.
    pub fn compute_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut CompiledWorkspace,
    ) -> Result<f64> {
        let outputs = self.compute_many_with_workspace(inputs, workspace)?;
        Ok(outputs.first().copied().unwrap_or(0.0))
    }

    /// Compute all selected outputs with a reusable workspace.
    pub fn compute_many_with_workspace<'a>(
        &self,
        inputs: &[f64],
        workspace: &'a mut CompiledWorkspace,
    ) -> Result<&'a [f64]> {
        self.fill_values(inputs, workspace)?;
        workspace.outputs.clear();
        for &output in &self.output_nodes {
            let value =
                workspace
                    .values
                    .get(output)
                    .copied()
                    .ok_or(AutodiffError::IndexOutOfBounds {
                        index: output,
                        max_index: workspace.values.len().saturating_sub(1),
                    })?;
            workspace.outputs.push(value);
        }
        Ok(&workspace.outputs)
    }

    fn reverse_for_output(&self, output: NodeId, workspace: &mut CompiledWorkspace) -> Result<()> {
        workspace.cotangents.clear();
        workspace.cotangents.resize(workspace.values.len(), 0.0);
        if output >= workspace.cotangents.len() {
            return Err(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index: workspace.cotangents.len().saturating_sub(1),
            });
        }
        workspace.cotangents[output] = 1.0;

        for (offset, instruction) in self.instructions.iter().enumerate().rev() {
            let node_id = self.num_inputs + offset;
            let current = workspace.cotangents[node_id];
            if current == 0.0 {
                continue;
            }
            match *instruction {
                Instruction::Constant(_) => {}
                Instruction::Unary { op, arg } => {
                    let args = Self::gather_unary(&workspace.values, arg)?;
                    let value = workspace.values[node_id];
                    if let op_rules::LocalRule::Unary { dy, .. } =
                        op_rules::local_rule(op, &args, value)?
                    {
                        workspace.cotangents[arg] += current * dy;
                    }
                }
                Instruction::Binary { op, left, right } => {
                    let args = Self::gather_binary(&workspace.values, left, right)?;
                    let value = workspace.values[node_id];
                    if let op_rules::LocalRule::Binary {
                        dy_left, dy_right, ..
                    } = op_rules::local_rule(op, &args, value)?
                    {
                        workspace.cotangents[left] += current * dy_left;
                        workspace.cotangents[right] += current * dy_right;
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute primary output and gradient.
    pub fn gradient(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let mut workspace = self.workspace();
        let (value, gradient) = self.gradient_with_workspace(inputs, &mut workspace)?;
        Ok((value, gradient.to_vec()))
    }

    /// Compute primary output and gradient with a reusable workspace.
    pub fn gradient_with_workspace<'a>(
        &self,
        inputs: &[f64],
        workspace: &'a mut CompiledWorkspace,
    ) -> Result<(f64, &'a [f64])> {
        self.fill_values(inputs, workspace)?;
        workspace.gradients.clear();
        workspace.gradients.resize(self.num_inputs, 0.0);
        let Some(&output) = self.output_nodes.first() else {
            return Ok((0.0, &workspace.gradients));
        };
        self.reverse_for_output(output, workspace)?;
        workspace
            .gradients
            .copy_from_slice(&workspace.cotangents[..self.num_inputs]);
        Ok((workspace.values[output], &workspace.gradients))
    }

    /// Compute Jacobian for all selected outputs.
    pub fn jacobian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut workspace = self.workspace();
        self.jacobian_with_workspace(inputs, &mut workspace)
    }

    /// Compute Jacobian for all selected outputs with a reusable workspace.
    pub fn jacobian_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut CompiledWorkspace,
    ) -> Result<Vec<Vec<f64>>> {
        self.fill_values(inputs, workspace)?;
        let mut jacobian = Vec::with_capacity(self.output_nodes.len());
        for &output in &self.output_nodes {
            self.reverse_for_output(output, workspace)?;
            jacobian.push(workspace.cotangents[..self.num_inputs].to_vec());
        }
        Ok(jacobian)
    }

    /// Return static compatibility details for a backend.
    pub fn backend_support_report(&self, backend: BackendKind) -> Result<BackendSupportReport> {
        let capabilities = backend.capabilities();
        let supports_required_outputs =
            self.output_nodes.len() <= 1 || capabilities.supports_multi_output;
        let mut missing_opcodes = Vec::new();
        let mut supports_required_arities = true;
        for instruction in self.flat_instructions_slice() {
            if !capabilities.supports_opcode(instruction.opcode)
                && !missing_opcodes.contains(&instruction.opcode)
            {
                missing_opcodes.push(instruction.opcode);
            }
            match instruction.opcode.arity() {
                0 if !capabilities.supports_constants => supports_required_arities = false,
                1 if !capabilities.supports_unary => supports_required_arities = false,
                2 if !capabilities.supports_binary => supports_required_arities = false,
                _ => {}
            }
        }
        let runtime_available = backend.runtime_available();
        let supports_required_opcodes = missing_opcodes.is_empty();
        let mut common_reasons = Vec::new();
        if !capabilities.supports_f64 {
            common_reasons.push(BackendRejectionReason::MissingF64);
        }
        if !supports_required_outputs {
            common_reasons.push(BackendRejectionReason::UnsupportedOutputs);
        }
        if !supports_required_opcodes {
            common_reasons.push(BackendRejectionReason::UnsupportedOpcodes);
        }
        if !supports_required_arities {
            common_reasons.push(BackendRejectionReason::UnsupportedArities);
        }
        if !runtime_available {
            common_reasons.push(BackendRejectionReason::UnavailableRuntime);
        }

        let mut batch_compute_rejection_reasons = common_reasons.clone();
        if !capabilities.supports_batch_compute {
            batch_compute_rejection_reasons.push(BackendRejectionReason::NoBatchCompute);
        }
        let mut batch_gradient_rejection_reasons = common_reasons;
        if !capabilities.supports_batch_gradient {
            batch_gradient_rejection_reasons.push(BackendRejectionReason::NoBatchGradient);
        }

        Ok(BackendSupportReport {
            backend,
            supports_f64: capabilities.supports_f64,
            supports_required_outputs,
            supports_required_opcodes,
            supports_required_arities,
            supports_batch_compute: capabilities.supports_batch_compute,
            supports_batch_gradient: capabilities.supports_batch_gradient,
            lane_width: backend.lane_width(),
            runtime_available,
            required_runtime_features: backend.required_runtime_features().to_vec(),
            unavailable_runtime_features: backend.unavailable_runtime_features(),
            missing_opcodes,
            batch_compute_rejection_reasons,
            batch_gradient_rejection_reasons,
        })
    }

    /// Return static compatibility details for all built-in backends.
    pub fn backend_support_reports(&self) -> Result<Vec<BackendSupportReport>> {
        Ok(vec![
            self.backend_support_report(BackendKind::Scalar)?,
            self.backend_support_report(BackendKind::MockDeviceCpu)?,
            self.backend_support_report(BackendKind::Wgpu)?,
            self.backend_support_report(BackendKind::SimdF64x4)?,
            self.backend_support_report(BackendKind::SimdF64x2)?,
        ])
    }

    /// Return a device-oriented batch buffer plan for a backend.
    #[must_use]
    pub fn device_batch_plan(&self, backend: BackendKind, batch_size: usize) -> DeviceBatchPlan {
        backend.batch_plan(self, batch_size)
    }

    /// Allocate mock-device buffers for this compiled graph and batch size.
    #[must_use]
    pub fn allocate_mock_device_buffers(&self, batch_size: usize) -> DeviceBufferSet {
        MockDeviceBackend.allocate_batch_buffers(self, batch_size)
    }

    /// Execute batch value computation through mock-device buffers.
    pub fn compute_batch_mock_device_into(
        &self,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        MockDeviceBackend.compute_batch_with_buffers(self, batch, buffers, output)
    }

    /// Execute batch gradient computation through mock-device buffers.
    pub fn gradient_batch_mock_device_into(
        &self,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        MockDeviceBackend.gradient_batch_with_buffers(self, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Allocate real WGPU buffers for this compiled graph and batch size.
    pub fn allocate_wgpu_buffers(
        &self,
        backend: &WgpuBackend,
        batch_size: usize,
    ) -> Result<WgpuBufferSet> {
        backend.allocate_batch_buffers(self, batch_size)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Execute batch value computation through real WGPU buffers.
    pub fn compute_batch_wgpu_into(
        &self,
        backend: &WgpuBackend,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        backend.compute_batch_with_buffers(self, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Execute batch gradient computation through real WGPU buffers.
    pub fn gradient_batch_wgpu_into(
        &self,
        backend: &WgpuBackend,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        backend.gradient_batch_with_buffers(self, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Return whether this graph is statically eligible for the exact-safe native WGPU compute path.
    #[must_use]
    pub fn supports_native_wgpu_batch_compute(&self, backend: &WgpuBackend) -> bool {
        backend.supports_native_batch_compute(self)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Return whether this graph and concrete batch can use the exact-safe native WGPU compute path.
    #[must_use]
    pub fn supports_native_wgpu_batch_compute_for_batch(
        &self,
        backend: &WgpuBackend,
        batch: BatchInputs<'_>,
    ) -> bool {
        backend.supports_native_batch_compute_for_batch(self, batch)
    }

    /// Return static compatibility details for the preferred SIMD backend.
    pub fn simd_support_report(&self) -> Result<BackendSupportReport> {
        let f64x4 = self.backend_support_report(BackendKind::SimdF64x4)?;
        if f64x4.supports_batch_compute || f64x4.supports_batch_gradient {
            return Ok(f64x4);
        }
        self.backend_support_report(BackendKind::SimdF64x2)
    }

    /// Return the preferred backend for batch value computation.
    #[must_use]
    pub fn recommended_batch_compute_backend(&self) -> BackendKind {
        match self.backend_support_report(BackendKind::SimdF64x4) {
            Ok(report) if report.can_compute_batch() => BackendKind::SimdF64x4,
            _ => match self.backend_support_report(BackendKind::SimdF64x2) {
                Ok(report) if report.can_compute_batch() => BackendKind::SimdF64x2,
                _ => BackendKind::Scalar,
            },
        }
    }

    /// Return the preferred backend for batch gradient computation.
    #[must_use]
    pub fn recommended_batch_gradient_backend(&self) -> BackendKind {
        match self.backend_support_report(BackendKind::SimdF64x4) {
            Ok(report) if report.can_gradient_batch() => BackendKind::SimdF64x4,
            _ => match self.backend_support_report(BackendKind::SimdF64x2) {
                Ok(report) if report.can_gradient_batch() => BackendKind::SimdF64x2,
                _ => BackendKind::Scalar,
            },
        }
    }

    /// Compute all selected outputs for a batch of rows.
    pub fn compute_batch(&self, batch: BatchInputs<'_>) -> Result<BatchValues> {
        let mut buffer = BatchValuesBuffer::new();
        self.compute_batch_into(batch, &mut buffer)?;
        Ok(BatchValues {
            data: buffer.data,
            batch_size: buffer.batch_size,
            output_dim: buffer.output_dim,
        })
    }

    /// Compute all selected outputs for a batch with automatic backend dispatch.
    pub fn compute_batch_auto(&self, batch: BatchInputs<'_>) -> Result<(BackendKind, BatchValues)> {
        let mut buffer = BatchValuesBuffer::new();
        let backend = self.compute_batch_auto_into(batch, &mut buffer)?;
        Ok((
            backend,
            BatchValues {
                data: buffer.data,
                batch_size: buffer.batch_size,
                output_dim: buffer.output_dim,
            },
        ))
    }

    /// Compute all selected outputs into a reusable output buffer.
    pub fn compute_batch_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        self.check_batch(batch)?;
        let mut workspace = self.workspace();
        let output_dim = self.output_nodes.len();
        buffer.reset(batch.batch_size, output_dim);
        for row_index in 0..batch.batch_size {
            let outputs =
                self.compute_many_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
            buffer.data.extend_from_slice(outputs);
        }
        Ok(())
    }

    /// Compute all selected outputs into a reusable buffer with automatic backend dispatch.
    pub fn compute_batch_auto_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<BackendKind> {
        let backend = self.recommended_batch_compute_backend();
        backend.compute_batch(self, batch, buffer)?;
        Ok(backend)
    }

    /// Compute primary-output values and gradients for a batch of rows.
    pub fn gradient_batch(&self, batch: BatchInputs<'_>) -> Result<BatchGradients> {
        let mut buffer = BatchGradientsBuffer::new();
        self.gradient_batch_into(batch, &mut buffer)?;
        Ok(BatchGradients {
            values: buffer.values,
            gradients: buffer.gradients,
            batch_size: buffer.batch_size,
            input_dim: buffer.input_dim,
        })
    }

    /// Compute primary-output values and gradients for a batch with automatic backend dispatch.
    pub fn gradient_batch_auto(
        &self,
        batch: BatchInputs<'_>,
    ) -> Result<(BackendKind, BatchGradients)> {
        let mut buffer = BatchGradientsBuffer::new();
        let backend = self.gradient_batch_auto_into(batch, &mut buffer)?;
        Ok((
            backend,
            BatchGradients {
                values: buffer.values,
                gradients: buffer.gradients,
                batch_size: buffer.batch_size,
                input_dim: buffer.input_dim,
            },
        ))
    }

    /// Compute primary-output values and gradients into a reusable gradient buffer.
    pub fn gradient_batch_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        self.check_batch(batch)?;
        let mut workspace = self.workspace();
        buffer.reset(batch.batch_size, self.num_inputs);
        for row_index in 0..batch.batch_size {
            let (value, gradient) =
                self.gradient_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
            buffer.values.push(value);
            buffer.gradients.extend_from_slice(gradient);
        }
        Ok(())
    }

    /// Compute primary-output values and gradients into a reusable buffer with automatic backend dispatch.
    pub fn gradient_batch_auto_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<BackendKind> {
        let backend = self.recommended_batch_gradient_backend();
        backend.gradient_batch(self, batch, buffer)?;
        Ok(backend)
    }
}

fn simd_unsupported_opcode_error() -> AutodiffError {
    AutodiffError::InvalidGraph {
        reason: "simd backend does not support a required opcode",
    }
}

fn simd_scalar_value(opcode: OpCode, args: &[f64]) -> Result<f64> {
    let op = opcode
        .to_multi_ad()
        .ok_or_else(simd_unsupported_opcode_error)?;
    op_rules::forward_value(op, args)
}

fn simd_scalar_first_derivatives(opcode: OpCode, args: &[f64], value: f64) -> Result<Vec<f64>> {
    let op = opcode
        .to_multi_ad()
        .ok_or_else(simd_unsupported_opcode_error)?;
    op_rules::first_derivatives(op, args, value)
}

fn append_scalar_compute_tail(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    start_row: usize,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    let mut workspace = graph.workspace();
    for row_index in start_row..batch.batch_size {
        let outputs =
            graph.compute_many_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
        buffer.data.extend_from_slice(outputs);
    }
    Ok(())
}

fn append_scalar_gradient_tail(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    start_row: usize,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    let mut workspace = graph.workspace();
    for row_index in start_row..batch.batch_size {
        let (value, gradient) =
            graph.gradient_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
        buffer.values.push(value);
        buffer.gradients.extend_from_slice(gradient);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn checked_m128_lane(values: &[__m128d], index: NodeId) -> Result<__m128d> {
    values
        .get(index)
        .copied()
        .ok_or(AutodiffError::IndexOutOfBounds {
            index,
            max_index: values.len().saturating_sub(1),
        })
}

#[cfg(target_arch = "x86_64")]
unsafe fn add_m128_cotangent(
    cotangents: &mut [__m128d],
    index: NodeId,
    contribution: __m128d,
) -> Result<()> {
    use std::arch::x86_64::_mm_add_pd;

    let max_index = cotangents.len().saturating_sub(1);
    let target = cotangents
        .get_mut(index)
        .ok_or(AutodiffError::IndexOutOfBounds { index, max_index })?;
    *target = _mm_add_pd(*target, contribution);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn active_m128_contribution(current: __m128d, contribution: __m128d) -> __m128d {
    use std::arch::x86_64::{_mm_and_pd, _mm_cmpneq_pd, _mm_setzero_pd};

    let active = _mm_cmpneq_pd(current, _mm_setzero_pd());
    _mm_and_pd(contribution, active)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_unary(input: __m128d, opcode: OpCode) -> Result<__m128d> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut stored = [0.0_f64; 2];
    _mm_storeu_pd(stored.as_mut_ptr(), input);
    let first = simd_scalar_value(opcode, &[stored[0]])?;
    let second = simd_scalar_value(opcode, &[stored[1]])?;
    Ok(_mm_set_pd(second, first))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_unary_derivative(
    input: __m128d,
    output: __m128d,
    opcode: OpCode,
) -> Result<__m128d> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut input_values = [0.0_f64; 2];
    let mut output_values = [0.0_f64; 2];
    _mm_storeu_pd(input_values.as_mut_ptr(), input);
    _mm_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(opcode, &[input_values[0]], output_values[0])?[0];
    let second = simd_scalar_first_derivatives(opcode, &[input_values[1]], output_values[1])?[0];
    Ok(_mm_set_pd(second, first))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_binary(
    left: __m128d,
    right: __m128d,
    opcode: OpCode,
) -> Result<__m128d> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut left_values = [0.0_f64; 2];
    let mut right_values = [0.0_f64; 2];
    _mm_storeu_pd(left_values.as_mut_ptr(), left);
    _mm_storeu_pd(right_values.as_mut_ptr(), right);
    let first = simd_scalar_value(opcode, &[left_values[0], right_values[0]])?;
    let second = simd_scalar_value(opcode, &[left_values[1], right_values[1]])?;
    Ok(_mm_set_pd(second, first))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_binary_derivatives(
    left: __m128d,
    right: __m128d,
    output: __m128d,
    opcode: OpCode,
) -> Result<(__m128d, __m128d)> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut left_values = [0.0_f64; 2];
    let mut right_values = [0.0_f64; 2];
    let mut output_values = [0.0_f64; 2];
    _mm_storeu_pd(left_values.as_mut_ptr(), left);
    _mm_storeu_pd(right_values.as_mut_ptr(), right);
    _mm_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(
        opcode,
        &[left_values[0], right_values[0]],
        output_values[0],
    )?;
    let second = simd_scalar_first_derivatives(
        opcode,
        &[left_values[1], right_values[1]],
        output_values[1],
    )?;
    Ok((
        _mm_set_pd(second[0], first[0]),
        _mm_set_pd(second[1], first[1]),
    ))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_forward_values(
    flat: &[FlatInstruction],
    values: &mut Vec<__m128d>,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm_add_pd, _mm_and_pd, _mm_andnot_pd, _mm_cmpgt_pd, _mm_div_pd, _mm_mul_pd, _mm_set1_pd,
        _mm_setzero_pd, _mm_sqrt_pd, _mm_sub_pd,
    };

    for instruction in flat {
        let value = match instruction.opcode {
            OpCode::Constant => _mm_set1_pd(instruction.value),
            OpCode::Add => _mm_add_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Sub => _mm_sub_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Mul => _mm_mul_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Div => _mm_div_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Pow | OpCode::LogAddExp => {
                let left = checked_m128_lane(values, instruction.left)?;
                let right = checked_m128_lane(values, instruction.right)?;
                simd_f64x2_scalar_binary(left, right, instruction.opcode)?
            }
            OpCode::Neg => _mm_sub_pd(
                _mm_setzero_pd(),
                checked_m128_lane(values, instruction.left)?,
            ),
            OpCode::Sqrt => _mm_sqrt_pd(checked_m128_lane(values, instruction.left)?),
            OpCode::Relu => {
                let input = checked_m128_lane(values, instruction.left)?;
                let mask = _mm_cmpgt_pd(input, _mm_setzero_pd());
                _mm_and_pd(input, mask)
            }
            OpCode::Abs => {
                let input = checked_m128_lane(values, instruction.left)?;
                _mm_andnot_pd(_mm_set1_pd(-0.0), input)
            }
            OpCode::Sin
            | OpCode::Cos
            | OpCode::Tan
            | OpCode::Tanh
            | OpCode::Log1pExp
            | OpCode::Exp
            | OpCode::Ln => {
                let input = checked_m128_lane(values, instruction.left)?;
                simd_f64x2_scalar_unary(input, instruction.opcode)?
            }
        };
        values.push(value);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn checked_m256_lane(values: &[__m256d], index: NodeId) -> Result<__m256d> {
    values
        .get(index)
        .copied()
        .ok_or(AutodiffError::IndexOutOfBounds {
            index,
            max_index: values.len().saturating_sub(1),
        })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_m256_cotangent(
    cotangents: &mut [__m256d],
    index: NodeId,
    contribution: __m256d,
) -> Result<()> {
    use std::arch::x86_64::_mm256_add_pd;

    let max_index = cotangents.len().saturating_sub(1);
    let target = cotangents
        .get_mut(index)
        .ok_or(AutodiffError::IndexOutOfBounds { index, max_index })?;
    *target = _mm256_add_pd(*target, contribution);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn active_m256_contribution(current: __m256d, contribution: __m256d) -> __m256d {
    use std::arch::x86_64::{_mm256_and_pd, _mm256_cmp_pd, _mm256_setzero_pd, _CMP_NEQ_UQ};

    let active = _mm256_cmp_pd(current, _mm256_setzero_pd(), _CMP_NEQ_UQ);
    _mm256_and_pd(contribution, active)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_unary(input: __m256d, opcode: OpCode) -> Result<__m256d> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut stored = [0.0_f64; 4];
    _mm256_storeu_pd(stored.as_mut_ptr(), input);
    let first = simd_scalar_value(opcode, &[stored[0]])?;
    let second = simd_scalar_value(opcode, &[stored[1]])?;
    let third = simd_scalar_value(opcode, &[stored[2]])?;
    let fourth = simd_scalar_value(opcode, &[stored[3]])?;
    Ok(_mm256_set_pd(fourth, third, second, first))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_unary_derivative(
    input: __m256d,
    output: __m256d,
    opcode: OpCode,
) -> Result<__m256d> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut input_values = [0.0_f64; 4];
    let mut output_values = [0.0_f64; 4];
    _mm256_storeu_pd(input_values.as_mut_ptr(), input);
    _mm256_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(opcode, &[input_values[0]], output_values[0])?[0];
    let second = simd_scalar_first_derivatives(opcode, &[input_values[1]], output_values[1])?[0];
    let third = simd_scalar_first_derivatives(opcode, &[input_values[2]], output_values[2])?[0];
    let fourth = simd_scalar_first_derivatives(opcode, &[input_values[3]], output_values[3])?[0];
    Ok(_mm256_set_pd(fourth, third, second, first))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_binary(
    left: __m256d,
    right: __m256d,
    opcode: OpCode,
) -> Result<__m256d> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut left_values = [0.0_f64; 4];
    let mut right_values = [0.0_f64; 4];
    _mm256_storeu_pd(left_values.as_mut_ptr(), left);
    _mm256_storeu_pd(right_values.as_mut_ptr(), right);
    let first = simd_scalar_value(opcode, &[left_values[0], right_values[0]])?;
    let second = simd_scalar_value(opcode, &[left_values[1], right_values[1]])?;
    let third = simd_scalar_value(opcode, &[left_values[2], right_values[2]])?;
    let fourth = simd_scalar_value(opcode, &[left_values[3], right_values[3]])?;
    Ok(_mm256_set_pd(fourth, third, second, first))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_binary_derivatives(
    left: __m256d,
    right: __m256d,
    output: __m256d,
    opcode: OpCode,
) -> Result<(__m256d, __m256d)> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut left_values = [0.0_f64; 4];
    let mut right_values = [0.0_f64; 4];
    let mut output_values = [0.0_f64; 4];
    _mm256_storeu_pd(left_values.as_mut_ptr(), left);
    _mm256_storeu_pd(right_values.as_mut_ptr(), right);
    _mm256_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(
        opcode,
        &[left_values[0], right_values[0]],
        output_values[0],
    )?;
    let second = simd_scalar_first_derivatives(
        opcode,
        &[left_values[1], right_values[1]],
        output_values[1],
    )?;
    let third = simd_scalar_first_derivatives(
        opcode,
        &[left_values[2], right_values[2]],
        output_values[2],
    )?;
    let fourth = simd_scalar_first_derivatives(
        opcode,
        &[left_values[3], right_values[3]],
        output_values[3],
    )?;
    Ok((
        _mm256_set_pd(fourth[0], third[0], second[0], first[0]),
        _mm256_set_pd(fourth[1], third[1], second[1], first[1]),
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_forward_values(
    flat: &[FlatInstruction],
    values: &mut Vec<__m256d>,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_andnot_pd, _mm256_cmp_pd, _mm256_div_pd,
        _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_sqrt_pd, _mm256_sub_pd,
        _CMP_GT_OQ,
    };

    for instruction in flat {
        let value = match instruction.opcode {
            OpCode::Constant => _mm256_set1_pd(instruction.value),
            OpCode::Add => _mm256_add_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Sub => _mm256_sub_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Mul => _mm256_mul_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Div => _mm256_div_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Pow | OpCode::LogAddExp => {
                let left = checked_m256_lane(values, instruction.left)?;
                let right = checked_m256_lane(values, instruction.right)?;
                simd_f64x4_scalar_binary(left, right, instruction.opcode)?
            }
            OpCode::Neg => _mm256_sub_pd(
                _mm256_setzero_pd(),
                checked_m256_lane(values, instruction.left)?,
            ),
            OpCode::Sqrt => _mm256_sqrt_pd(checked_m256_lane(values, instruction.left)?),
            OpCode::Relu => {
                let input = checked_m256_lane(values, instruction.left)?;
                let mask = _mm256_cmp_pd(input, _mm256_setzero_pd(), _CMP_GT_OQ);
                _mm256_and_pd(input, mask)
            }
            OpCode::Abs => {
                let input = checked_m256_lane(values, instruction.left)?;
                _mm256_andnot_pd(_mm256_set1_pd(-0.0), input)
            }
            OpCode::Sin
            | OpCode::Cos
            | OpCode::Tan
            | OpCode::Tanh
            | OpCode::Log1pExp
            | OpCode::Exp
            | OpCode::Ln => {
                let input = checked_m256_lane(values, instruction.left)?;
                simd_f64x4_scalar_unary(input, instruction.opcode)?
            }
        };
        values.push(value);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn compute_batch_simd_f64x2(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    let output_dim = graph.output_nodes.len();
    let value_count = graph.num_inputs + graph.instructions.len();
    buffer.reset(batch.batch_size, output_dim);

    let mut values: Vec<__m128d> = Vec::with_capacity(value_count);
    let mut pair_outputs: Vec<[f64; 2]> = Vec::with_capacity(output_dim);
    let mut row_index = 0;
    while row_index + 1 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            unsafe {
                values.push(_mm_set_pd(second[input_index], first[input_index]));
            }
        }

        unsafe {
            simd_f64x2_forward_values(flat, &mut values)?;
        }

        pair_outputs.clear();
        for &output in &graph.output_nodes {
            let lane = checked_m128_lane(&values, output)?;
            let mut stored = [0.0_f64; 2];
            unsafe {
                _mm_storeu_pd(stored.as_mut_ptr(), lane);
            }
            pair_outputs.push(stored);
        }
        for output in &pair_outputs {
            buffer.data.push(output[0]);
        }
        for output in &pair_outputs {
            buffer.data.push(output[1]);
        }

        row_index += 2;
    }

    append_scalar_compute_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn compute_batch_simd_f64x4_impl(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    let output_dim = graph.output_nodes.len();
    let value_count = graph.num_inputs + graph.instructions.len();
    buffer.reset(batch.batch_size, output_dim);

    let mut values: Vec<__m256d> = Vec::with_capacity(value_count);
    let mut quad_outputs: Vec<[f64; 4]> = Vec::with_capacity(output_dim);
    let mut row_index = 0;
    while row_index + 3 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        let third = batch.try_row(row_index + 2)?;
        let fourth = batch.try_row(row_index + 3)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            values.push(_mm256_set_pd(
                fourth[input_index],
                third[input_index],
                second[input_index],
                first[input_index],
            ));
        }

        simd_f64x4_forward_values(flat, &mut values)?;

        quad_outputs.clear();
        for &output in &graph.output_nodes {
            let lane = checked_m256_lane(&values, output)?;
            let mut stored = [0.0_f64; 4];
            _mm256_storeu_pd(stored.as_mut_ptr(), lane);
            quad_outputs.push(stored);
        }
        for lane_index in 0..4 {
            for output in &quad_outputs {
                buffer.data.push(output[lane_index]);
            }
        }

        row_index += 4;
    }

    append_scalar_compute_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn compute_batch_simd_f64x4(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    if !supports_simd_f64x4_runtime() {
        return Err(AutodiffError::InvalidGraph {
            reason: "simd f64x4 backend requires x86_64 AVX support",
        });
    }
    unsafe { compute_batch_simd_f64x4_impl(graph, batch, buffer) }
}

#[cfg(not(target_arch = "x86_64"))]
fn compute_batch_simd_f64x4(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd f64x4 backend requires x86_64 AVX support",
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn compute_batch_simd_f64x2(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd backend requires x86_64 SSE2 support",
    })
}

#[cfg(target_arch = "x86_64")]
fn gradient_batch_simd_f64x2(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm_add_pd, _mm_and_pd, _mm_cmpgt_pd, _mm_cmplt_pd, _mm_div_pd, _mm_mul_pd, _mm_set1_pd,
        _mm_set_pd, _mm_setzero_pd, _mm_storeu_pd, _mm_sub_pd,
    };

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    buffer.reset(batch.batch_size, graph.num_inputs);

    let value_count = graph.num_inputs + graph.instructions.len();
    let mut values: Vec<__m128d> = Vec::with_capacity(value_count);
    let mut cotangents: Vec<__m128d> = Vec::with_capacity(value_count);
    let mut gradient_pairs = Vec::with_capacity(graph.num_inputs);
    let mut row_index = 0;
    while row_index + 1 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            unsafe {
                values.push(_mm_set_pd(second[input_index], first[input_index]));
            }
        }

        unsafe {
            simd_f64x2_forward_values(flat, &mut values)?;
        }

        let Some(&output) = graph.output_nodes.first() else {
            buffer.values.extend_from_slice(&[0.0, 0.0]);
            buffer
                .gradients
                .resize(buffer.gradients.len() + graph.num_inputs * 2, 0.0);
            row_index += 2;
            continue;
        };

        cotangents.clear();
        cotangents.resize_with(value_count, || unsafe { _mm_setzero_pd() });
        let max_index = cotangents.len().saturating_sub(1);
        *cotangents
            .get_mut(output)
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index,
            })? = unsafe { _mm_set1_pd(1.0) };

        for instruction in flat.iter().rev() {
            let current = checked_m128_lane(&cotangents, instruction.output)?;
            unsafe {
                match instruction.opcode {
                    OpCode::Constant => {}
                    OpCode::Add => {
                        let contribution = active_m128_contribution(current, current);
                        add_m128_cotangent(&mut cotangents, instruction.left, contribution)?;
                        add_m128_cotangent(&mut cotangents, instruction.right, contribution)?;
                    }
                    OpCode::Sub => {
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, current),
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(
                                current,
                                _mm_sub_pd(_mm_setzero_pd(), current),
                            ),
                        )?;
                    }
                    OpCode::Mul => {
                        let left = checked_m128_lane(&values, instruction.left)?;
                        let right = checked_m128_lane(&values, instruction.right)?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, right)),
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(current, _mm_mul_pd(current, left)),
                        )?;
                    }
                    OpCode::Div => {
                        let left = checked_m128_lane(&values, instruction.left)?;
                        let right = checked_m128_lane(&values, instruction.right)?;
                        let right_squared = _mm_mul_pd(right, right);
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_div_pd(current, right)),
                        )?;
                        let right_contribution = _mm_sub_pd(
                            _mm_setzero_pd(),
                            _mm_div_pd(_mm_mul_pd(current, left), right_squared),
                        );
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(current, right_contribution),
                        )?;
                    }
                    OpCode::Pow | OpCode::LogAddExp => {
                        let left = checked_m128_lane(&values, instruction.left)?;
                        let right = checked_m128_lane(&values, instruction.right)?;
                        let output_value = checked_m128_lane(&values, instruction.output)?;
                        let (left_derivative, right_derivative) =
                            simd_f64x2_scalar_binary_derivatives(
                                left,
                                right,
                                output_value,
                                instruction.opcode,
                            )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, left_derivative)),
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(
                                current,
                                _mm_mul_pd(current, right_derivative),
                            ),
                        )?;
                    }
                    OpCode::Neg => {
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(
                                current,
                                _mm_sub_pd(_mm_setzero_pd(), current),
                            ),
                        )?;
                    }
                    OpCode::Sqrt => {
                        let output_value = checked_m128_lane(&values, instruction.output)?;
                        let contribution =
                            _mm_div_pd(_mm_mul_pd(current, _mm_set1_pd(0.5)), output_value);
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, contribution),
                        )?;
                    }
                    OpCode::Relu => {
                        let input_value = checked_m128_lane(&values, instruction.left)?;
                        let mask = _mm_cmpgt_pd(input_value, _mm_setzero_pd());
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_and_pd(current, mask)),
                        )?;
                    }
                    OpCode::Sin
                    | OpCode::Cos
                    | OpCode::Tan
                    | OpCode::Tanh
                    | OpCode::Log1pExp
                    | OpCode::Exp
                    | OpCode::Ln => {
                        let input_value = checked_m128_lane(&values, instruction.left)?;
                        let output_value = checked_m128_lane(&values, instruction.output)?;
                        let derivative = simd_f64x2_scalar_unary_derivative(
                            input_value,
                            output_value,
                            instruction.opcode,
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, derivative)),
                        )?;
                    }
                    OpCode::Abs => {
                        let input_value = checked_m128_lane(&values, instruction.left)?;
                        let positive = _mm_cmpgt_pd(input_value, _mm_setzero_pd());
                        let negative = _mm_cmplt_pd(input_value, _mm_setzero_pd());
                        let sign = _mm_add_pd(
                            _mm_and_pd(_mm_set1_pd(1.0), positive),
                            _mm_and_pd(_mm_set1_pd(-1.0), negative),
                        );
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, sign)),
                        )?;
                    }
                }
            }
        }

        let output_value = checked_m128_lane(&values, output)?;
        let mut value_pair = [0.0_f64; 2];
        unsafe {
            _mm_storeu_pd(value_pair.as_mut_ptr(), output_value);
        }
        buffer.values.extend_from_slice(&value_pair);

        gradient_pairs.clear();
        for input_index in 0..graph.num_inputs {
            let gradient_lane = checked_m128_lane(&cotangents, input_index)?;
            let mut stored = [0.0_f64; 2];
            unsafe {
                _mm_storeu_pd(stored.as_mut_ptr(), gradient_lane);
            }
            gradient_pairs.push(stored);
        }
        for pair in &gradient_pairs {
            buffer.gradients.push(pair[0]);
        }
        for pair in &gradient_pairs {
            buffer.gradients.push(pair[1]);
        }

        row_index += 2;
    }

    append_scalar_gradient_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gradient_batch_simd_f64x4_impl(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_cmp_pd, _mm256_div_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_set_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_GT_OQ, _CMP_LT_OQ,
    };

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    buffer.reset(batch.batch_size, graph.num_inputs);

    let value_count = graph.num_inputs + graph.instructions.len();
    let mut values: Vec<__m256d> = Vec::with_capacity(value_count);
    let mut cotangents: Vec<__m256d> = Vec::with_capacity(value_count);
    let mut gradient_quads = Vec::with_capacity(graph.num_inputs);
    let mut row_index = 0;
    while row_index + 3 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        let third = batch.try_row(row_index + 2)?;
        let fourth = batch.try_row(row_index + 3)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            values.push(_mm256_set_pd(
                fourth[input_index],
                third[input_index],
                second[input_index],
                first[input_index],
            ));
        }

        simd_f64x4_forward_values(flat, &mut values)?;

        let Some(&output) = graph.output_nodes.first() else {
            buffer.values.extend_from_slice(&[0.0; 4]);
            buffer
                .gradients
                .resize(buffer.gradients.len() + graph.num_inputs * 4, 0.0);
            row_index += 4;
            continue;
        };

        cotangents.clear();
        cotangents.resize_with(value_count, || _mm256_setzero_pd());
        let max_index = cotangents.len().saturating_sub(1);
        *cotangents
            .get_mut(output)
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index,
            })? = _mm256_set1_pd(1.0);

        for instruction in flat.iter().rev() {
            let current = checked_m256_lane(&cotangents, instruction.output)?;
            match instruction.opcode {
                OpCode::Constant => {}
                OpCode::Add => {
                    let contribution = active_m256_contribution(current, current);
                    add_m256_cotangent(&mut cotangents, instruction.left, contribution)?;
                    add_m256_cotangent(&mut cotangents, instruction.right, contribution)?;
                }
                OpCode::Sub => {
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, current),
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(
                            current,
                            _mm256_sub_pd(_mm256_setzero_pd(), current),
                        ),
                    )?;
                }
                OpCode::Mul => {
                    let left = checked_m256_lane(&values, instruction.left)?;
                    let right = checked_m256_lane(&values, instruction.right)?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, right)),
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(current, _mm256_mul_pd(current, left)),
                    )?;
                }
                OpCode::Div => {
                    let left = checked_m256_lane(&values, instruction.left)?;
                    let right = checked_m256_lane(&values, instruction.right)?;
                    let right_squared = _mm256_mul_pd(right, right);
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_div_pd(current, right)),
                    )?;
                    let right_contribution = _mm256_sub_pd(
                        _mm256_setzero_pd(),
                        _mm256_div_pd(_mm256_mul_pd(current, left), right_squared),
                    );
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(current, right_contribution),
                    )?;
                }
                OpCode::Pow | OpCode::LogAddExp => {
                    let left = checked_m256_lane(&values, instruction.left)?;
                    let right = checked_m256_lane(&values, instruction.right)?;
                    let output_value = checked_m256_lane(&values, instruction.output)?;
                    let (left_derivative, right_derivative) = simd_f64x4_scalar_binary_derivatives(
                        left,
                        right,
                        output_value,
                        instruction.opcode,
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, left_derivative)),
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(current, _mm256_mul_pd(current, right_derivative)),
                    )?;
                }
                OpCode::Neg => {
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(
                            current,
                            _mm256_sub_pd(_mm256_setzero_pd(), current),
                        ),
                    )?;
                }
                OpCode::Sqrt => {
                    let output_value = checked_m256_lane(&values, instruction.output)?;
                    let contribution =
                        _mm256_div_pd(_mm256_mul_pd(current, _mm256_set1_pd(0.5)), output_value);
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, contribution),
                    )?;
                }
                OpCode::Relu => {
                    let input_value = checked_m256_lane(&values, instruction.left)?;
                    let mask = _mm256_cmp_pd(input_value, _mm256_setzero_pd(), _CMP_GT_OQ);
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_and_pd(current, mask)),
                    )?;
                }
                OpCode::Sin
                | OpCode::Cos
                | OpCode::Tan
                | OpCode::Tanh
                | OpCode::Log1pExp
                | OpCode::Exp
                | OpCode::Ln => {
                    let input_value = checked_m256_lane(&values, instruction.left)?;
                    let output_value = checked_m256_lane(&values, instruction.output)?;
                    let derivative = simd_f64x4_scalar_unary_derivative(
                        input_value,
                        output_value,
                        instruction.opcode,
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, derivative)),
                    )?;
                }
                OpCode::Abs => {
                    let input_value = checked_m256_lane(&values, instruction.left)?;
                    let positive = _mm256_cmp_pd(input_value, _mm256_setzero_pd(), _CMP_GT_OQ);
                    let negative = _mm256_cmp_pd(input_value, _mm256_setzero_pd(), _CMP_LT_OQ);
                    let sign = _mm256_add_pd(
                        _mm256_and_pd(_mm256_set1_pd(1.0), positive),
                        _mm256_and_pd(_mm256_set1_pd(-1.0), negative),
                    );
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, sign)),
                    )?;
                }
            }
        }

        let output_value = checked_m256_lane(&values, output)?;
        let mut value_quad = [0.0_f64; 4];
        _mm256_storeu_pd(value_quad.as_mut_ptr(), output_value);
        buffer.values.extend_from_slice(&value_quad);

        gradient_quads.clear();
        for input_index in 0..graph.num_inputs {
            let gradient_lane = checked_m256_lane(&cotangents, input_index)?;
            let mut stored = [0.0_f64; 4];
            _mm256_storeu_pd(stored.as_mut_ptr(), gradient_lane);
            gradient_quads.push(stored);
        }
        for lane_index in 0..4 {
            for quad in &gradient_quads {
                buffer.gradients.push(quad[lane_index]);
            }
        }

        row_index += 4;
    }

    append_scalar_gradient_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn gradient_batch_simd_f64x4(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    if !supports_simd_f64x4_runtime() {
        return Err(AutodiffError::InvalidGraph {
            reason: "simd f64x4 backend requires x86_64 AVX support",
        });
    }
    unsafe { gradient_batch_simd_f64x4_impl(graph, batch, buffer) }
}

#[cfg(not(target_arch = "x86_64"))]
fn gradient_batch_simd_f64x4(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd f64x4 backend requires x86_64 AVX support",
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn gradient_batch_simd_f64x2(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd backend requires x86_64 SSE2 support",
    })
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

#[cfg(feature = "backend-wgpu")]
#[inline]
fn wgpu_buffer_size_bytes(len: usize) -> u64 {
    let logical = len.saturating_mul(std::mem::size_of::<f64>()) as u64;
    logical.max(8)
}

#[cfg(feature = "backend-wgpu")]
fn encode_f64_slice(data: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().saturating_mul(std::mem::size_of::<f64>()));
    for value in data {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(feature = "backend-wgpu")]
fn decode_f64_bytes(bytes: &[u8], len: usize) -> Result<Vec<f64>> {
    let expected_len = len.saturating_mul(std::mem::size_of::<f64>());
    if bytes.len() < expected_len {
        return Err(AutodiffError::InvalidGraph {
            reason: "wgpu readback length does not match planned buffer length",
        });
    }
    let mut values = Vec::with_capacity(len);
    for chunk in bytes[..expected_len].chunks_exact(std::mem::size_of::<f64>()) {
        let mut array = [0_u8; std::mem::size_of::<f64>()];
        array.copy_from_slice(chunk);
        values.push(f64::from_ne_bytes(array));
    }
    Ok(values)
}

#[cfg(feature = "backend-wgpu")]
#[inline]
fn wgpu_buffer_size_bytes_f32(len: usize) -> u64 {
    let logical = len.saturating_mul(std::mem::size_of::<f32>()) as u64;
    logical.max(4)
}

#[cfg(feature = "backend-wgpu")]
fn encode_f32_slice(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().saturating_mul(std::mem::size_of::<f32>()));
    for value in data {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(feature = "backend-wgpu")]
fn decode_f32_bytes(bytes: &[u8], len: usize) -> Result<Vec<f32>> {
    let expected_len = len.saturating_mul(std::mem::size_of::<f32>());
    if bytes.len() < expected_len {
        return Err(AutodiffError::InvalidGraph {
            reason: "wgpu readback length does not match requested f32 buffer length",
        });
    }
    let mut values = Vec::with_capacity(len);
    for chunk in bytes[..expected_len].chunks_exact(std::mem::size_of::<f32>()) {
        let mut array = [0_u8; std::mem::size_of::<f32>()];
        array.copy_from_slice(chunk);
        values.push(f32::from_ne_bytes(array));
    }
    Ok(values)
}

#[cfg(feature = "backend-wgpu")]
fn encode_u32_slice(data: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().saturating_mul(std::mem::size_of::<u32>()));
    for value in data {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(feature = "backend-wgpu")]
/// Conservative opcode subset currently supported by the exact-safe native WGPU batch-compute path.
pub const WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES: [OpCode; 4] =
    [OpCode::Constant, OpCode::Neg, OpCode::Relu, OpCode::Abs];

#[cfg(feature = "backend-wgpu")]
fn checked_u32_from_usize(value: usize, reason: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| AutodiffError::InvalidGraph { reason })
}

#[cfg(feature = "backend-wgpu")]
#[inline]
fn is_exact_f32_roundtrip(value: f64) -> bool {
    if value.is_nan() {
        return false;
    }
    let narrowed = value as f32;
    let widened = f64::from(narrowed);
    widened == value && (value != 0.0 || widened.is_sign_negative() == value.is_sign_negative())
}

#[cfg(feature = "backend-wgpu")]
fn wgpu_native_exact_safe_supports_opcode(opcode: OpCode) -> bool {
    WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES.contains(&opcode)
}

#[cfg(feature = "backend-wgpu")]
fn wgpu_native_exact_safe_graph(graph: &CompiledGraph) -> bool {
    !graph.output_nodes().is_empty()
        && graph.flat_instructions_slice().iter().all(|instruction| {
            wgpu_native_exact_safe_supports_opcode(instruction.opcode)
                && (instruction.opcode != OpCode::Constant
                    || is_exact_f32_roundtrip(instruction.value))
        })
}

#[cfg(feature = "backend-wgpu")]
fn wgpu_native_exact_safe_batch(batch: BatchInputs<'_>) -> bool {
    batch.data.iter().copied().all(is_exact_f32_roundtrip)
}

#[cfg(feature = "backend-wgpu")]
const WGPU_NATIVE_WORDS_PER_INSTRUCTION: usize = 8;

#[cfg(feature = "backend-wgpu")]
const WGPU_NATIVE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> instruction_words: array<u32>;
@group(0) @binding(2) var<storage, read> output_nodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> value_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(5) var<storage, read> kernel_meta: array<u32>;

fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 {
        return x;
    }
    return 0.0;
}

fn log1p_exp_scalar(x: f32) -> f32 {
    if x > 0.0 {
        return x + log(1.0 + exp(-x));
    }
    return log(1.0 + exp(x));
}

fn log_add_exp_scalar(a: f32, b: f32) -> f32 {
    var max_value = a;
    var min_value = b;
    if a < b {
        max_value = b;
        min_value = a;
    }
    return max_value + log(1.0 + exp(min_value - max_value));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let num_inputs = kernel_meta[0u];
    let num_instructions = kernel_meta[1u];
    let num_outputs = kernel_meta[2u];
    let value_count = kernel_meta[3u];
    let batch_size = kernel_meta[4u];
    if row >= batch_size {
        return;
    }

    let input_base = row * num_inputs;
    let value_base = row * value_count;
    let output_base = row * num_outputs;

    var input_index = 0u;
    loop {
        if input_index >= num_inputs {
            break;
        }
        value_data[value_base + input_index] = input_data[input_base + input_index];
        input_index = input_index + 1u;
    }

    var instruction_index = 0u;
    loop {
        if instruction_index >= num_instructions {
            break;
        }
        let word_base = instruction_index * 8u;
        let opcode = instruction_words[word_base + 0u];
        let output_index = instruction_words[word_base + 1u];
        let left_index = instruction_words[word_base + 2u];
        let right_index = instruction_words[word_base + 3u];
        let value_bits = instruction_words[word_base + 4u];
        var left_value: f32 = 0.0;
        if left_index != 4294967295u {
            left_value = value_data[value_base + left_index];
        }
        var right_value: f32 = 0.0;
        if right_index != 4294967295u {
            right_value = value_data[value_base + right_index];
        }
        var result_value: f32 = 0.0;
        switch opcode {
            case 0u: {
                result_value = bitcast<f32>(value_bits);
            }
            case 1u: {
                result_value = left_value + right_value;
            }
            case 2u: {
                result_value = left_value - right_value;
            }
            case 3u: {
                result_value = left_value * right_value;
            }
            case 4u: {
                result_value = left_value / right_value;
            }
            case 5u: {
                result_value = pow(left_value, right_value);
            }
            case 6u: {
                result_value = sin(left_value);
            }
            case 7u: {
                result_value = cos(left_value);
            }
            case 8u: {
                result_value = tan(left_value);
            }
            case 9u: {
                result_value = tanh(left_value);
            }
            case 10u: {
                result_value = relu_scalar(left_value);
            }
            case 11u: {
                result_value = log1p_exp_scalar(left_value);
            }
            case 12u: {
                result_value = log_add_exp_scalar(left_value, right_value);
            }
            case 13u: {
                result_value = -left_value;
            }
            case 14u: {
                result_value = exp(left_value);
            }
            case 15u: {
                result_value = log(left_value);
            }
            case 16u: {
                result_value = sqrt(left_value);
            }
            case 17u: {
                result_value = abs(left_value);
            }
            default: {
                result_value = 0.0;
            }
        }
        value_data[value_base + output_index] = result_value;
        instruction_index = instruction_index + 1u;
    }

    var output_index = 0u;
    loop {
        if output_index >= num_outputs {
            break;
        }
        let node = output_nodes[output_index];
        output_data[output_base + output_index] = value_data[value_base + node];
        output_index = output_index + 1u;
    }
}
"#;

#[cfg(feature = "backend-wgpu")]
impl WgpuBackend {
    /// Create the default WGPU backend using device id 0 and automatic transfers.
    pub fn new_default() -> Result<Self> {
        Self::new(
            AcceleratorDeviceContext::wgpu(0),
            DeviceTransferPolicy::Automatic,
        )
    }

    /// Create the WGPU backend skeleton from a context and transfer policy.
    pub fn new(
        context: AcceleratorDeviceContext,
        transfer_policy: DeviceTransferPolicy,
    ) -> Result<Self> {
        Self::from_boundary(GpuBackendBoundary::new(context, transfer_policy))
    }

    /// Create the WGPU backend skeleton from a boundary descriptor.
    pub fn from_boundary(boundary: GpuBackendBoundary) -> Result<Self> {
        if boundary.context.kind != AcceleratorDeviceKind::Wgpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu backend requires a WGPU accelerator context",
            });
        }
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|_| AutodiffError::InvalidGraph {
            reason: "wgpu adapter request failed",
        })?;
        let adapter_info = adapter.get_info();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("petite-ad-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|_| AutodiffError::InvalidGraph {
            reason: "wgpu device request failed",
        })?;
        Ok(Self {
            boundary,
            device,
            queue,
            adapter_info,
        })
    }

    /// Return the boundary descriptor used for this backend.
    #[must_use]
    pub fn boundary(&self) -> &GpuBackendBoundary {
        &self.boundary
    }

    /// Return the accelerator context used for this backend.
    #[must_use]
    pub fn context(&self) -> &AcceleratorDeviceContext {
        &self.boundary.context
    }

    /// Return the configured transfer policy.
    #[must_use]
    pub fn transfer_policy(&self) -> DeviceTransferPolicy {
        self.boundary.transfer_policy
    }

    /// Return the resolved adapter name.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Return the conservative exact-safe opcode subset used by the native WGPU batch-compute path.
    #[must_use]
    pub fn native_batch_compute_supported_opcodes() -> &'static [OpCode] {
        &WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES
    }

    /// Return whether this graph is eligible for the restricted exact-safe native WGPU path.
    #[must_use]
    pub fn supports_native_batch_compute(&self, graph: &CompiledGraph) -> bool {
        wgpu_native_exact_safe_graph(graph)
    }

    /// Return whether a concrete batch is eligible for the exact-safe native WGPU path.
    #[must_use]
    pub fn supports_native_batch_compute_for_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
    ) -> bool {
        self.supports_native_batch_compute(graph) && wgpu_native_exact_safe_batch(batch)
    }

    fn create_initialized_storage_buffer(
        &self,
        label: &'static str,
        bytes: &[u8],
        min_size: u64,
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: min_size.max(bytes.len() as u64).max(4),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    }

    fn download_raw_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("petite-ad-wgpu-readback"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("petite-ad-wgpu-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        let submission = self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_for(submission));
        match rx.recv() {
            Ok(Ok(())) => {}
            _ => {
                return Err(AutodiffError::InvalidGraph {
                    reason: "wgpu readback mapping failed",
                });
            }
        }
        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    fn compute_batch_native(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
    ) -> Result<Vec<f64>> {
        let metadata = graph.metadata();
        let input_f32: Vec<f32> = batch.data.iter().map(|value| *value as f32).collect();
        let output_count = batch.batch_size.saturating_mul(metadata.num_outputs);
        let input_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-inputs",
            &encode_f32_slice(&input_f32),
            wgpu_buffer_size_bytes_f32(input_f32.len()),
        );
        let instruction_words = encode_u32_slice(&self.native_instruction_words(graph)?);
        let instruction_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-instructions",
            &instruction_words,
            instruction_words.len() as u64,
        );
        let output_nodes: Result<Vec<u32>> = graph
            .output_nodes()
            .iter()
            .map(|node| {
                checked_u32_from_usize(*node, "wgpu native output node index exceeds u32 range")
            })
            .collect();
        let output_nodes = output_nodes?;
        let output_node_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-output-nodes",
            &encode_u32_slice(&output_nodes),
            (output_nodes.len().max(1) * std::mem::size_of::<u32>()) as u64,
        );
        let value_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("petite-ad-wgpu-native-values"),
            size: wgpu_buffer_size_bytes_f32(batch.batch_size.saturating_mul(metadata.value_count)),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("petite-ad-wgpu-native-outputs"),
            size: wgpu_buffer_size_bytes_f32(output_count),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let meta = [
            checked_u32_from_usize(
                metadata.num_inputs,
                "wgpu native input dimension exceeds u32 range",
            )?,
            checked_u32_from_usize(
                metadata.num_instructions,
                "wgpu native instruction count exceeds u32 range",
            )?,
            checked_u32_from_usize(
                metadata.num_outputs,
                "wgpu native output dimension exceeds u32 range",
            )?,
            checked_u32_from_usize(
                metadata.value_count,
                "wgpu native value count exceeds u32 range",
            )?,
            checked_u32_from_usize(batch.batch_size, "wgpu native batch size exceeds u32 range")?,
        ];
        let meta_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-meta",
            &encode_u32_slice(&meta),
            (meta.len() * std::mem::size_of::<u32>()) as u64,
        );
        self.queue.submit([]);

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("petite-ad-wgpu-native-compute-shader"),
                source: wgpu::ShaderSource::Wgsl(WGPU_NATIVE_SHADER.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("petite-ad-wgpu-native-compute-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("petite-ad-wgpu-native-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instruction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: meta_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("petite-ad-wgpu-native-compute-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("petite-ad-wgpu-native-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = checked_u32_from_usize(
                batch.batch_size.div_ceil(64),
                "wgpu native workgroup count exceeds u32 range",
            )?;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        let raw_output =
            self.download_raw_buffer(&output_buffer, wgpu_buffer_size_bytes_f32(output_count))?;
        let output_f32 = decode_f32_bytes(&raw_output, output_count)?;
        Ok(output_f32.into_iter().map(f64::from).collect())
    }

    fn native_instruction_words(&self, graph: &CompiledGraph) -> Result<Vec<u32>> {
        let mut words = Vec::with_capacity(
            graph.flat_instructions_slice().len() * WGPU_NATIVE_WORDS_PER_INSTRUCTION,
        );
        for instruction in graph.flat_instructions_slice() {
            words.push(match instruction.opcode {
                OpCode::Constant => 0,
                OpCode::Add => 1,
                OpCode::Sub => 2,
                OpCode::Mul => 3,
                OpCode::Div => 4,
                OpCode::Pow => 5,
                OpCode::Sin => 6,
                OpCode::Cos => 7,
                OpCode::Tan => 8,
                OpCode::Tanh => 9,
                OpCode::Relu => 10,
                OpCode::Log1pExp => 11,
                OpCode::LogAddExp => 12,
                OpCode::Neg => 13,
                OpCode::Exp => 14,
                OpCode::Ln => 15,
                OpCode::Sqrt => 16,
                OpCode::Abs => 17,
            });
            words.push(checked_u32_from_usize(
                instruction.output,
                "wgpu native instruction output index exceeds u32 range",
            )?);
            words.push(if instruction.left == UNUSED_NODE_ID {
                u32::MAX
            } else {
                checked_u32_from_usize(
                    instruction.left,
                    "wgpu native instruction left index exceeds u32 range",
                )?
            });
            words.push(if instruction.right == UNUSED_NODE_ID {
                u32::MAX
            } else {
                checked_u32_from_usize(
                    instruction.right,
                    "wgpu native instruction right index exceeds u32 range",
                )?
            });
            words.push((instruction.value as f32).to_bits());
            words.extend_from_slice(&[0, 0, 0]);
        }
        Ok(words)
    }

    /// Allocate a real WGPU buffer set for a compiled graph and batch size.
    pub fn allocate_batch_buffers(
        &self,
        graph: &CompiledGraph,
        batch_size: usize,
    ) -> Result<WgpuBufferSet> {
        let plan = self.batch_plan(graph, batch_size);
        let mut buffers = Vec::with_capacity(plan.buffer_handles.len());
        for handle in &plan.buffer_handles {
            let label = format!("petite-ad-wgpu-{:?}", handle.kind);
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label.as_str()),
                size: wgpu_buffer_size_bytes(handle.len),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            buffers.push(WgpuBuffer {
                handle: *handle,
                buffer,
            });
        }
        Ok(WgpuBufferSet { plan, buffers })
    }

    fn upload_buffer(
        &self,
        buffers: &WgpuBufferSet,
        kind: DeviceBufferKind,
        data: &[f64],
    ) -> Result<()> {
        let buffer = buffers.buffer(kind)?;
        if buffer.handle.len != data.len() {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu upload length must match planned buffer length",
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let bytes = encode_f64_slice(data);
        self.queue.write_buffer(&buffer.buffer, 0, &bytes);
        self.queue.submit([]);
        Ok(())
    }

    fn download_buffer(&self, buffers: &WgpuBufferSet, kind: DeviceBufferKind) -> Result<Vec<f64>> {
        let buffer = buffers.buffer(kind)?;
        if buffer.handle.len == 0 {
            return Ok(Vec::new());
        }
        let raw =
            self.download_raw_buffer(&buffer.buffer, wgpu_buffer_size_bytes(buffer.handle.len))?;
        decode_f64_bytes(&raw, buffer.handle.len)
    }

    /// Execute batch value computation through explicit WGPU transfers.
    pub fn compute_batch_with_buffers(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        graph.check_batch(batch)?;
        if buffers.plan.backend != BackendKind::Wgpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu execution requires a WGPU buffer plan",
            });
        }
        if buffers.plan.batch_size != batch.batch_size || buffers.plan.input_dim != batch.input_dim
        {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch shape must match WGPU buffer plan",
            });
        }

        let transfers = buffers.plan.compute_transfer_plan.clone();
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::HostToDevice {
                if transfer.buffer != DeviceBufferKind::Inputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "wgpu compute supports host-to-device input transfers only",
                    });
                }
                buffers.upload(self, DeviceBufferKind::Inputs, batch.data)?;
            }
        }

        let used_native_kernel = self.supports_native_batch_compute_for_batch(graph, batch);
        let scratch_data = if used_native_kernel {
            self.compute_batch_native(graph, batch)?
        } else {
            let device_inputs = buffers.download(self, DeviceBufferKind::Inputs)?;
            let device_batch = BatchInputs::new(&device_inputs, batch.batch_size, batch.input_dim)?;
            let mut scratch = BatchValuesBuffer::new();
            ScalarBackend.compute_batch(graph, device_batch, &mut scratch)?;
            scratch.data
        };
        buffers.upload(self, DeviceBufferKind::Outputs, &scratch_data)?;
        output.reset(batch.batch_size, graph.output_nodes.len());
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::DeviceToHost {
                if transfer.buffer != DeviceBufferKind::Outputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "wgpu compute supports device-to-host output transfers only",
                    });
                }
                output
                    .data
                    .extend_from_slice(&buffers.download(self, DeviceBufferKind::Outputs)?);
            }
        }

        Ok(DeviceExecutionTrace {
            backend: BackendKind::Wgpu,
            mode: DeviceExecutionMode::ComputeBatch,
            transfers,
            used_native_kernel,
        })
    }

    /// Execute batch gradient computation through explicit WGPU transfers.
    pub fn gradient_batch_with_buffers(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        graph.check_batch(batch)?;
        if buffers.plan.backend != BackendKind::Wgpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu execution requires a WGPU buffer plan",
            });
        }
        if buffers.plan.batch_size != batch.batch_size || buffers.plan.input_dim != batch.input_dim
        {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch shape must match WGPU buffer plan",
            });
        }

        let transfers = buffers.plan.gradient_transfer_plan.clone();
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::HostToDevice {
                if transfer.buffer != DeviceBufferKind::Inputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "wgpu gradient supports host-to-device input transfers only",
                    });
                }
                buffers.upload(self, DeviceBufferKind::Inputs, batch.data)?;
            }
        }

        let device_inputs = buffers.download(self, DeviceBufferKind::Inputs)?;
        let device_batch = BatchInputs::new(&device_inputs, batch.batch_size, batch.input_dim)?;
        let mut scratch = BatchGradientsBuffer::new();
        ScalarBackend.gradient_batch(graph, device_batch, &mut scratch)?;
        buffers.upload(self, DeviceBufferKind::PrimaryValues, &scratch.values)?;
        buffers.upload(self, DeviceBufferKind::Gradients, &scratch.gradients)?;
        output.reset(batch.batch_size, graph.num_inputs);
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::DeviceToHost {
                match transfer.buffer {
                    DeviceBufferKind::PrimaryValues => output.values.extend_from_slice(
                        &buffers.download(self, DeviceBufferKind::PrimaryValues)?,
                    ),
                    DeviceBufferKind::Gradients => output
                        .gradients
                        .extend_from_slice(&buffers.download(self, DeviceBufferKind::Gradients)?),
                    _ => {
                        return Err(AutodiffError::InvalidGraph {
                            reason:
                                "wgpu gradient supports primary-value and gradient downloads only",
                        });
                    }
                }
            }
        }

        Ok(DeviceExecutionTrace {
            backend: BackendKind::Wgpu,
            mode: DeviceExecutionMode::GradientBatch,
            transfers,
            used_native_kernel: false,
        })
    }
}

impl DeviceBackend for ScalarBackend {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        BackendKind::Scalar
    }
}

impl DeviceBackend for MockDeviceBackend {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        BackendKind::MockDeviceCpu
    }
}

#[cfg(feature = "backend-wgpu")]
impl DeviceBackend for WgpuBackend {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        BackendKind::Wgpu
    }
}

impl DeviceBackend for SimdBackend {
    fn backend_kind(&self, graph: &CompiledGraph) -> BackendKind {
        graph
            .simd_support_report()
            .map(|report| report.backend)
            .unwrap_or(BackendKind::SimdF64x2)
    }
}

impl DeviceBackend for BackendKind {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        *self
    }
}

impl ExecutionBackend for SimdBackend {
    fn name(&self) -> &'static str {
        "simd"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::simd_f64x2()
    }

    fn compute(&self, _graph: &CompiledGraph, _inputs: &[f64]) -> Result<f64> {
        Err(AutodiffError::InvalidGraph {
            reason: "simd backend currently supports batch compute only",
        })
    }

    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        if let Ok(report) = graph.backend_support_report(BackendKind::SimdF64x4) {
            if report.can_compute_batch() {
                return compute_batch_simd_f64x4(graph, batch, buffer);
            }
        }
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_compute {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch compute on this target",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        compute_batch_simd_f64x2(graph, batch, buffer)
    }

    fn gradient(&self, _graph: &CompiledGraph, _inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        Err(AutodiffError::InvalidGraph {
            reason: "simd backend does not support reverse gradients yet",
        })
    }

    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        if let Ok(report) = graph.backend_support_report(BackendKind::SimdF64x4) {
            if report.can_gradient_batch() {
                return gradient_batch_simd_f64x4(graph, batch, buffer);
            }
        }
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_gradient {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch gradients on this target",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        gradient_batch_simd_f64x2(graph, batch, buffer)
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

#[cfg(feature = "backend-wgpu")]
impl ExecutionBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::wgpu_f64()
    }

    fn compute(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<f64> {
        let batch = BatchInputs::new(inputs, 1, inputs.len())?;
        let mut values = BatchValuesBuffer::new();
        self.compute_batch(graph, batch, &mut values)?;
        values
            .data
            .first()
            .copied()
            .ok_or(AutodiffError::InvalidGraph {
                reason: "wgpu compute did not produce an output value",
            })
    }

    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        graph.validate_backend_capabilities(&self.capabilities())?;
        let mut gpu_buffers = self.allocate_batch_buffers(graph, batch.batch_size)?;
        self.compute_batch_with_buffers(graph, batch, &mut gpu_buffers, buffer)?;
        Ok(())
    }

    fn gradient(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let batch = BatchInputs::new(inputs, 1, inputs.len())?;
        let mut gradients = BatchGradientsBuffer::new();
        self.gradient_batch(graph, batch, &mut gradients)?;
        let value = gradients
            .values
            .first()
            .copied()
            .ok_or(AutodiffError::InvalidGraph {
                reason: "wgpu gradient did not produce a primary output value",
            })?;
        Ok((value, gradients.gradients))
    }

    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        graph.validate_backend_capabilities(&self.capabilities())?;
        let mut gpu_buffers = self.allocate_batch_buffers(graph, batch.batch_size)?;
        self.gradient_batch_with_buffers(graph, batch, &mut gpu_buffers, buffer)?;
        Ok(())
    }
}

impl ExecutionBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::scalar_f64()
    }

    fn compute(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<f64> {
        graph.validate_backend_capabilities(&self.capabilities())?;
        graph.compute(inputs)
    }

    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_compute {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch compute",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        graph.compute_batch_into(batch, buffer)
    }

    fn gradient(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let capabilities = self.capabilities();
        if !capabilities.supports_reverse_gradient {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support reverse gradients",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        graph.gradient(inputs)
    }

    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_gradient {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch gradients",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        graph.gradient_batch_into(batch, buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::approx_eq_eps as approx_eq;

    // --- helpers ---

    fn make_simple_graph() -> CompiledGraph {
        let instructions = vec![Instruction::Binary {
            op: MultiAD::Add,
            left: 0,
            right: 1,
        }];
        CompiledGraph::new(2, instructions, vec![2]).unwrap()
    }

    fn make_multi_output_graph() -> CompiledGraph {
        // f(x) = x, outputs: [x, sin(x)]
        let instructions = vec![Instruction::Unary {
            op: MultiAD::Sin,
            arg: 0,
        }];
        CompiledGraph::new(1, instructions, vec![0, 1]).unwrap()
    }

    // ---- 1. compute / gradient input-length checks ----

    #[test]
    fn test_compiled_compute_wrong_input_length_error() {
        let graph = make_simple_graph();
        assert!(graph.compute(&[1.0]).is_err());
        assert!(graph.compute(&[1.0, 2.0, 3.0]).is_err());
        assert!(graph.compute(&[1.0, 2.0]).is_ok());
    }

    #[test]
    fn test_compiled_gradient_wrong_input_length_error() {
        let graph = make_simple_graph();
        assert!(graph.gradient(&[1.0]).is_err());
        assert!(graph.gradient(&[1.0, 2.0, 3.0]).is_err());
        assert!(graph.gradient(&[1.0, 2.0]).is_ok());
    }

    // ---- 2. batch with odd batch sizes ----

    #[test]
    fn test_compiled_compute_batch_odd_size() {
        let graph = make_simple_graph();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let batch = BatchInputs::new(&data, 3, 2).unwrap();
        let result = graph.compute_batch(batch).unwrap();
        assert_eq!(result.batch_size, 3);
        assert_eq!(result.output_dim, 1);
        assert_eq!(result.data.len(), 3);
        assert!(approx_eq(result.data[0], 3.0, 1e-10));
        assert!(approx_eq(result.data[1], 7.0, 1e-10));
        assert!(approx_eq(result.data[2], 11.0, 1e-10));
    }

    #[test]
    fn test_compiled_gradient_batch_odd_size() {
        let graph = make_simple_graph();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let batch = BatchInputs::new(&data, 3, 2).unwrap();
        let result = graph.gradient_batch(batch).unwrap();
        assert_eq!(result.batch_size, 3);
        assert_eq!(result.input_dim, 2);
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.gradients.len(), 6);
        // f(x,y) = x+y, gradients are always [1, 1]
        for i in 0..3 {
            assert!(approx_eq(result.gradients[i * 2], 1.0, 1e-10));
            assert!(approx_eq(result.gradients[i * 2 + 1], 1.0, 1e-10));
        }
    }

    // ---- 3. large batch ----

    #[test]
    fn test_compiled_compute_batch_large() {
        let graph = make_simple_graph();
        let batch_size = 100;
        let mut data = Vec::with_capacity(batch_size * 2);
        for i in 0..batch_size {
            let x = i as f64;
            data.push(x);
            data.push(x * 2.0);
        }
        let batch = BatchInputs::new(&data, batch_size, 2).unwrap();
        let result = graph.compute_batch(batch).unwrap();
        assert_eq!(result.batch_size, 100);
        assert_eq!(result.data.len(), 100);
        for i in 0..batch_size {
            assert!(approx_eq(result.data[i], (i as f64) * 3.0, 1e-10));
        }
    }

    // ---- 4. multi-output batch ----

    #[test]
    fn test_compiled_compute_batch_many() {
        let graph = make_multi_output_graph();
        let data = vec![0.0, 1.0, 2.0];
        let batch = BatchInputs::new(&data, 3, 1).unwrap();
        let result = graph.compute_batch(batch).unwrap();
        assert_eq!(result.batch_size, 3);
        assert_eq!(result.output_dim, 2);
        assert_eq!(result.data.len(), 6);
        // [x, sin(x)] for each row
        assert!(approx_eq(result.data[0], 0.0, 1e-10));
        assert!(approx_eq(result.data[1], 0.0_f64.sin(), 1e-10));
        assert!(approx_eq(result.data[2], 1.0, 1e-10));
        assert!(approx_eq(result.data[3], 1.0_f64.sin(), 1e-10));
        assert!(approx_eq(result.data[4], 2.0, 1e-10));
        assert!(approx_eq(result.data[5], 2.0_f64.sin(), 1e-10));
    }

    // ---- 5. CompiledWorkspace reuse ----

    #[test]
    fn test_workspace_reuse_compute() {
        let graph = make_simple_graph();
        let mut ws = graph.workspace();
        let v1 = graph.compute_with_workspace(&[1.0, 2.0], &mut ws).unwrap();
        assert!(approx_eq(v1, 3.0, 1e-10));
        let v2 = graph.compute_with_workspace(&[3.0, 4.0], &mut ws).unwrap();
        assert!(approx_eq(v2, 7.0, 1e-10));
    }

    #[test]
    fn test_workspace_reuse_gradient() {
        let graph = make_simple_graph();
        let mut ws = graph.workspace();
        let (v1, g1) = graph.gradient_with_workspace(&[1.0, 2.0], &mut ws).unwrap();
        assert!(approx_eq(v1, 3.0, 1e-10));
        assert!(approx_eq(g1[0], 1.0, 1e-10));
        assert!(approx_eq(g1[1], 1.0, 1e-10));
        let (v2, g2) = graph.gradient_with_workspace(&[3.0, 4.0], &mut ws).unwrap();
        assert!(approx_eq(v2, 7.0, 1e-10));
        assert!(approx_eq(g2[0], 1.0, 1e-10));
        assert!(approx_eq(g2[1], 1.0, 1e-10));
    }

    // ---- 6. BackendKind ----

    #[test]
    fn test_backend_kind_name_non_empty() {
        for kind in &[
            BackendKind::Scalar,
            BackendKind::MockDeviceCpu,
            BackendKind::Wgpu,
            BackendKind::SimdF64x4,
            BackendKind::SimdF64x2,
        ] {
            assert!(!kind.name().is_empty());
        }
    }

    #[test]
    fn test_backend_kind_capabilities() {
        for kind in &[
            BackendKind::Scalar,
            BackendKind::MockDeviceCpu,
            BackendKind::Wgpu,
            BackendKind::SimdF64x4,
            BackendKind::SimdF64x2,
        ] {
            let caps = kind.capabilities();
            // Every built-in backend should support f64.
            assert!(caps.supports_f64);
            // Every built-in backend should support at least one opcode.
            assert!(!caps.supported_opcodes.is_empty());
        }
    }

    // ---- 7. BatchInputs ----

    #[test]
    fn test_batch_inputs_try_row_valid() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let batch = BatchInputs::new(&data, 2, 2).unwrap();
        let row0 = batch.try_row(0).unwrap();
        assert_eq!(row0, &[1.0, 2.0]);
        let row1 = batch.try_row(1).unwrap();
        assert_eq!(row1, &[3.0, 4.0]);
    }

    #[test]
    fn test_batch_inputs_try_row_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let batch = BatchInputs::new(&data, 2, 2).unwrap();
        assert!(batch.try_row(2).is_err());
        assert!(batch.try_row(100).is_err());
    }

    #[test]
    fn test_batch_inputs_invalid_shape() {
        assert!(BatchInputs::new(&[1.0, 2.0], 2, 2).is_err());
        assert!(BatchInputs::new(&[1.0, 2.0, 3.0], 1, 2).is_err());
    }

    // ---- 8. BatchValues / BatchValuesBuffer ----

    #[test]
    fn test_batch_values_buffer_reuse() {
        let graph = make_simple_graph();
        let mut buffer = BatchValuesBuffer::new();

        let batch1 = BatchInputs::new(&[1.0, 2.0], 1, 2).unwrap();
        graph.compute_batch_into(batch1, &mut buffer).unwrap();
        let values1 = buffer.to_values();
        assert_eq!(values1.batch_size, 1);
        assert_eq!(values1.data, &[3.0]);

        let batch2 = BatchInputs::new(&[4.0, 5.0, 6.0, 7.0], 2, 2).unwrap();
        graph.compute_batch_into(batch2, &mut buffer).unwrap();
        let values2 = buffer.to_values();
        assert_eq!(values2.batch_size, 2);
        assert_eq!(values2.data, &[9.0, 13.0]);
    }

    // ---- 9. BatchGradients / BatchGradientsBuffer ----

    #[test]
    fn test_batch_gradients_buffer_reuse() {
        let graph = make_simple_graph();
        let mut buffer = BatchGradientsBuffer::new();

        let batch1 = BatchInputs::new(&[1.0, 2.0], 1, 2).unwrap();
        graph.gradient_batch_into(batch1, &mut buffer).unwrap();
        let grad1 = buffer.to_gradients();
        assert_eq!(grad1.batch_size, 1);
        assert!(approx_eq(grad1.values[0], 3.0, 1e-10));
        assert_eq!(grad1.gradients, &[1.0, 1.0]);

        let batch2 = BatchInputs::new(&[4.0, 5.0, 6.0, 7.0], 2, 2).unwrap();
        graph.gradient_batch_into(batch2, &mut buffer).unwrap();
        let grad2 = buffer.to_gradients();
        assert_eq!(grad2.batch_size, 2);
        assert!(approx_eq(grad2.values[0], 9.0, 1e-10));
        assert!(approx_eq(grad2.values[1], 13.0, 1e-10));
        assert_eq!(grad2.gradients, &[1.0, 1.0, 1.0, 1.0]);
    }

    // ---- 10. FlatInstruction and OpCode ----

    #[test]
    fn test_flat_instruction_debug() {
        let fi = FlatInstruction {
            opcode: OpCode::Add,
            output: 2,
            left: 0,
            right: 1,
            value: 0.0,
        };
        let s = format!("{:?}", fi);
        assert!(s.contains("Add"));
        assert!(s.contains("2"));
    }

    #[test]
    fn test_opcode_debug() {
        // Every variant should have a non-empty Debug representation.
        for op in &[
            OpCode::Constant,
            OpCode::Add,
            OpCode::Sub,
            OpCode::Mul,
            OpCode::Div,
            OpCode::Pow,
            OpCode::Sin,
            OpCode::Cos,
            OpCode::Tan,
            OpCode::Tanh,
            OpCode::Relu,
            OpCode::Log1pExp,
            OpCode::LogAddExp,
            OpCode::Neg,
            OpCode::Exp,
            OpCode::Ln,
            OpCode::Sqrt,
            OpCode::Abs,
        ] {
            assert!(!format!("{:?}", op).is_empty());
        }
    }

    #[test]
    fn test_opcode_arity() {
        assert_eq!(OpCode::Constant.arity(), 0);
        assert_eq!(OpCode::Sin.arity(), 1);
        assert_eq!(OpCode::Neg.arity(), 1);
        assert_eq!(OpCode::Add.arity(), 2);
        assert_eq!(OpCode::Mul.arity(), 2);
    }

    #[test]
    fn test_opcode_from_multi_ad() {
        assert_eq!(OpCode::from_multi_ad(MultiAD::Add).unwrap(), OpCode::Add);
        assert_eq!(OpCode::from_multi_ad(MultiAD::Sin).unwrap(), OpCode::Sin);
        assert!(OpCode::from_multi_ad(MultiAD::Inp).is_err());
    }

    // ---- 11. backend support reports ----

    #[test]
    fn test_backend_support_report_scalar() {
        let graph = make_simple_graph();
        let report = graph.backend_support_report(BackendKind::Scalar).unwrap();
        assert!(report.supports_f64);
        assert!(report.supports_required_opcodes);
        assert!(report.can_compute_batch());
        assert!(report.can_gradient_batch());
        assert_eq!(report.backend, BackendKind::Scalar);
    }

    #[test]
    fn test_simd_support_report() {
        let graph = make_simple_graph();
        let report = graph.simd_support_report().unwrap();
        // Report should belong to a SIMD backend.
        assert!(matches!(
            report.backend,
            BackendKind::SimdF64x4 | BackendKind::SimdF64x2
        ));
    }

    // ---- 12. Device types ----

    #[test]
    fn test_device_memory_location_variants() {
        let host = DeviceMemoryLocation::Host;
        let device = DeviceMemoryLocation::Device;
        assert_ne!(host, device);
        // Equality on same variant.
        assert_eq!(host, DeviceMemoryLocation::Host);
        assert_eq!(device, DeviceMemoryLocation::Device);
    }

    #[test]
    fn test_device_batch_plan_creation() {
        let graph = make_simple_graph();
        let plan = graph.device_batch_plan(BackendKind::Scalar, 10);
        assert_eq!(plan.backend, BackendKind::Scalar);
        assert_eq!(plan.batch_size, 10);
        assert_eq!(plan.input_dim, 2);
        assert_eq!(plan.output_dim, 1);
        assert!(!plan.buffers.is_empty());
        assert!(!plan.buffer_handles.is_empty());
    }

    #[test]
    fn test_mock_device_allocate_batch_buffers() {
        let graph = make_simple_graph();
        let buffers = graph.allocate_mock_device_buffers(5);
        assert_eq!(buffers.plan().backend, BackendKind::MockDeviceCpu);
        assert_eq!(buffers.plan().batch_size, 5);
        assert!(!buffers.buffers().is_empty());
    }

    #[test]
    fn test_device_buffer_set_upload_download() {
        let graph = make_simple_graph();
        let mut buffers = graph.allocate_mock_device_buffers(3);
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buffers
            .upload(DeviceBufferKind::Inputs, &input_data)
            .unwrap();
        let downloaded = buffers.download(DeviceBufferKind::Inputs).unwrap();
        assert_eq!(downloaded, input_data);
    }

    // ---- 13. graph with all ops ----

    #[test]
    fn test_compiled_graph_all_ops() {
        // Build a graph that uses many different ops.
        // Graph: 2 inputs (x, y)
        //  0: x (input)
        //  1: y (input)
        //  2: sin(x)            [Unary Sin]
        //  3: cos(y)            [Unary Cos]
        //  4: x + y            [Binary Add]
        //  5: x * y            [Binary Mul]
        //  6: -x               [Unary Neg]
        //  7: exp(x)           [Unary Exp]
        //  8: ln(y)            [Unary Ln]
        //  9: (x+y) / (x*y)    [Binary Div]
        // 10: x - y            [Binary Sub]
        // 11: tanh(x)          [Unary Tanh]
        // 12: sqrt(abs(x))     [Unary Sqrt] on abs(x)...
        //   Let's simplify: just use relu, abs on input
        let instructions = vec![
            Instruction::Unary {
                op: MultiAD::Sin,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Cos,
                arg: 1,
            },
            Instruction::Binary {
                op: MultiAD::Add,
                left: 0,
                right: 1,
            },
            Instruction::Binary {
                op: MultiAD::Mul,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Neg,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Exp,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Ln,
                arg: 1,
            },
            Instruction::Binary {
                op: MultiAD::Div,
                left: 3,
                right: 4,
            },
            Instruction::Binary {
                op: MultiAD::Sub,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Tanh,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Relu,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Abs,
                arg: 0,
            },
            Instruction::Constant(42.0),
            Instruction::Binary {
                op: MultiAD::Pow,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Log1pExp,
                arg: 0,
            },
            Instruction::Binary {
                op: MultiAD::LogAddExp,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Sqrt,
                arg: 4,
            },
            Instruction::Unary {
                op: MultiAD::Tan,
                arg: 0,
            },
        ];
        // Final output: use the Add node (index 4) as primary output.
        let graph = CompiledGraph::new(2, instructions, vec![4]).unwrap();

        // Compute with valid inputs.
        let result = graph.compute(&[0.5, 2.0]).unwrap();
        assert!(approx_eq(result, 2.5, 1e-10));

        // Gradient should also work.
        let (val, grad) = graph.gradient(&[0.5, 2.0]).unwrap();
        assert!(approx_eq(val, 2.5, 1e-10));
        assert_eq!(grad.len(), 2);
        // d/dx (x+y) = 1, d/dy (x+y) = 1
        assert!(approx_eq(grad[0], 1.0, 1e-10));
        assert!(approx_eq(grad[1], 1.0, 1e-10));
    }
}
