//! Multi-variable automatic differentiation.
//!
//! This module provides functionality for computing gradients of
//! multi-variable functions using computational graphs.

pub mod compiled;
pub mod first_order;
pub mod func;
pub mod graph;
pub(crate) mod op_rules;
pub mod second_order;
pub mod types;

#[cfg(test)]
mod examples;
#[cfg(test)]
mod tests;

pub use first_order::MultiAD;
pub use func::MultiFn;

pub use second_order::fr::MultiAD2FR;
pub use second_order::rf::MultiAD2RF;
pub use second_order::rr::MultiAD2RR;

pub use compiled::backend::{
    AcceleratorDeviceContext, AcceleratorDeviceKind, BackendCapabilities, BackendKind,
    BackendRejectionReason, BackendSupportReport, BatchLayout, DeviceBackend, DeviceBatchPlan,
    DeviceBuffer, DeviceBufferHandle, DeviceBufferKind, DeviceBufferLayout, DeviceBufferSet,
    DeviceExecutionMode, DeviceExecutionTrace, DeviceMemoryLocation, DeviceTransferKind,
    DeviceTransferPlan, DeviceTransferPolicy, ExecutionBackend, FlatInstruction,
    GpuBackendBoundary, Instruction, MockDeviceBackend, OpCode, ScalarBackend, SimdBackend,
    UNUSED_NODE_ID,
};
#[cfg(feature = "backend-wgpu")]
pub use compiled::backend::{
    WgpuBackend, WgpuBuffer, WgpuBufferSet, WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES,
};
pub use compiled::{
    BatchGradients, BatchGradientsBuffer, BatchInputs, BatchValues, BatchValuesBuffer,
    CompiledGraph, CompiledGraphMetadata, CompiledWorkspace,
};
pub use graph::builder::GraphBuilder;
pub use graph::core::{
    DomainPolicy, GradientCheckEntry, GradientCheckReport, Graph, GraphNode, GraphStats, NodeId,
};
pub use graph::expr::{ExprGraph, ExprNode};
pub use graph::tape::{Tape, TapeWorkspace};
