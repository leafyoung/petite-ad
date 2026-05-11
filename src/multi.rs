//! Multi-variable automatic differentiation.
//!
//! This module provides functionality for computing gradients of
//! multi-variable functions using computational graphs.

#![allow(unused_imports)] // Re-exports used via lib.rs re-export path
                          // Example implementations - not part of public API
mod f1;
mod f2;
mod f3;

pub mod builder;
pub mod compiled;
pub mod graph;
mod multi_ad;
pub mod multi_ad_fr;
pub mod multi_ad_rf;
pub mod multi_ad_rr;
mod parser;

// Shared internal modules for multivariate derivative rules and Hessian computation.
mod multi_hessian_common;
pub(crate) mod op_rules;

mod multi_fn;
#[cfg(test)]
mod tests;
pub mod types;

pub use compiled::{
    AcceleratorDeviceContext, AcceleratorDeviceKind, BackendCapabilities, BackendKind,
    BackendRejectionReason, BackendSupportReport, BatchGradients, BatchGradientsBuffer,
    BatchInputs, BatchLayout, BatchValues, BatchValuesBuffer, CompiledGraph, CompiledGraphMetadata,
    CompiledWorkspace, DeviceBackend, DeviceBatchPlan, DeviceBuffer, DeviceBufferHandle,
    DeviceBufferKind, DeviceBufferLayout, DeviceBufferSet, DeviceExecutionMode,
    DeviceExecutionTrace, DeviceMemoryLocation, DeviceTransferKind, DeviceTransferPlan,
    DeviceTransferPolicy, ExecutionBackend, FlatInstruction, GpuBackendBoundary, Instruction,
    MockDeviceBackend, OpCode, ScalarBackend, SimdBackend, UNUSED_NODE_ID,
};
#[cfg(feature = "backend-wgpu")]
pub use compiled::{
    WgpuBackend, WgpuBuffer, WgpuBufferSet, WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES,
};
pub use graph::{
    DomainPolicy, ExprGraph, ExprNode, GradientCheckEntry, GradientCheckReport, Graph, GraphNode,
    GraphStats, NodeId, Tape, TapeWorkspace,
};
pub use multi_ad::MultiAD;
pub use multi_fn::MultiFn;

pub use multi_ad_fr::MultiAD2FR;
pub use multi_ad_rf::MultiAD2RF;
pub use multi_ad_rr::MultiAD2RR;
