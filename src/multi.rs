//! Multi-variable automatic differentiation.
//!
//! This module provides functionality for computing gradients of
//! multi-variable functions using computational graphs.

#![allow(unused_imports)] // Re-exports used via lib.rs re-export path
                          // Example implementations - not part of public API
mod f1;
mod f2;
mod f3;

pub mod backend;
pub mod builder;
pub mod compiled;
pub mod expr;
pub mod graph;
mod multi_ad;
pub mod multi_ad_fr;
pub mod multi_ad_rf;
pub mod multi_ad_rr;
mod parser;
pub mod tape;

// Shared internal modules for multivariate derivative rules and Hessian computation.
mod multi_hessian_common;
pub(crate) mod op_rules;

mod multi_fn;
#[cfg(test)]
mod tests;
pub mod types;

pub use backend::{
    AcceleratorDeviceContext, AcceleratorDeviceKind, BackendCapabilities, BackendKind,
    BackendRejectionReason, BackendSupportReport, BatchLayout, DeviceBackend, DeviceBatchPlan,
    DeviceBuffer, DeviceBufferHandle, DeviceBufferKind, DeviceBufferLayout, DeviceBufferSet,
    DeviceExecutionMode, DeviceExecutionTrace, DeviceMemoryLocation, DeviceTransferKind,
    DeviceTransferPlan, DeviceTransferPolicy, ExecutionBackend, FlatInstruction,
    GpuBackendBoundary, Instruction, MockDeviceBackend, OpCode, ScalarBackend, SimdBackend,
    UNUSED_NODE_ID,
};
#[cfg(feature = "backend-wgpu")]
pub use backend::{
    WgpuBackend, WgpuBuffer, WgpuBufferSet, WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES,
};
pub use compiled::{
    BatchGradients, BatchGradientsBuffer, BatchInputs, BatchValues, BatchValuesBuffer,
    CompiledGraph, CompiledGraphMetadata, CompiledWorkspace,
};
pub use expr::{ExprGraph, ExprNode};
pub use graph::{
    DomainPolicy, GradientCheckEntry, GradientCheckReport, Graph, GraphNode, GraphStats, NodeId,
};
pub use multi_ad::MultiAD;
pub use multi_fn::MultiFn;
pub use tape::{Tape, TapeWorkspace};

pub use multi_ad_fr::MultiAD2FR;
pub use multi_ad_rf::MultiAD2RF;
pub use multi_ad_rr::MultiAD2RR;
