//! Closure-free compiled instruction IR for acceleration-ready graph execution.

pub mod backend;
pub mod batches;
pub mod ir;

pub use batches::{
    BatchGradients, BatchGradientsBuffer, BatchInputs, BatchValues, BatchValuesBuffer,
};
pub use ir::{CompiledGraph, CompiledGraphMetadata, CompiledWorkspace};
