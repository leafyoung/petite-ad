//! Reference scalar f64 backend.

use crate::multi::compiled::backend::device::DeviceBackend;
use crate::multi::compiled::backend::dispatch::{BackendKind, ExecutionBackend};
use crate::multi::compiled::backend::types::BackendCapabilities;
use crate::multi::compiled::{BatchGradientsBuffer, BatchInputs, BatchValuesBuffer, CompiledGraph};
use crate::{AutodiffError, Result};

/// Reference scalar backend for the execution-backend abstraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScalarBackend;

impl DeviceBackend for ScalarBackend {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        BackendKind::Scalar
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
