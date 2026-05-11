//! Backend dispatch types, ExecutionBackend trait, and auto-dispatch logic.

use crate::multi::backend::device::{DeviceBackend, DeviceMemoryLocation};
use crate::multi::backend::scalar::ScalarBackend;
use crate::multi::backend::simd::{
    compute_batch_simd_f64x2, compute_batch_simd_f64x4, gradient_batch_simd_f64x2,
    gradient_batch_simd_f64x4,
};
use crate::multi::backend::types::{
    supports_simd_f64x2_runtime, supports_simd_f64x4_runtime, supports_wgpu_runtime,
    BackendCapabilities, OpCode,
};
use crate::multi::compiled::{BatchGradientsBuffer, BatchInputs, BatchValuesBuffer, CompiledGraph};
use crate::{AutodiffError, Result};

#[cfg(feature = "backend-wgpu")]
use crate::multi::backend::wgpu::WgpuBackend;

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
    ///
    /// # Performance
    ///
    /// When `self` is [`BackendKind::Wgpu`] this method calls
    /// [`WgpuBackend::new_default`] on every invocation, which initialises a
    /// new GPU device and queue each time. For repeated batch calls, create a
    /// [`WgpuBackend`] once and call its [`ExecutionBackend::compute_batch`]
    /// method directly instead of routing through `BackendKind::Wgpu`.
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
    ///
    /// # Performance
    ///
    /// When `self` is [`BackendKind::Wgpu`] this method calls
    /// [`WgpuBackend::new_default`] on every invocation, which initialises a
    /// new GPU device and queue each time. For repeated batch calls, create a
    /// [`WgpuBackend`] once and call its [`ExecutionBackend::gradient_batch`]
    /// method directly instead of routing through `BackendKind::Wgpu`.
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

impl DeviceBackend for BackendKind {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        *self
    }
}
