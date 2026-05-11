//! # petite AD
//!
//! A pure Rust automatic differentiation library supporting both single-variable
//! and multi-variable functions with reverse-mode differentiation (backpropagation).
//!
//! ## Features
//!
//! - **Single-variable autodiff** - Chain operations like `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, and `abs`
//! - **Multi-variable autodiff** - Build computational graphs for multiple inputs
//! - **Reusable graph/tape API** - Build once with node handles, select explicit outputs, and evaluate repeatedly
//! - **Graph validation/export** - Check reusable graphs and export Mermaid/DOT views
//! - **Public forward-mode AD** - Compute derivatives, gradients, and directional derivatives directly
//! - **Opt-in checked mode** - Validate real-domain restrictions for sensitive operations
//! - **Zero-copy backward pass** - Efficient gradient computation through closure chains
//! - **Convenient macros** - Use `mono_ops![]` and `multi_ops![]` for concise notation
//!
//! ## Examples
//!
//! ### Single-variable function
//! ```
//! use petite_ad::{MonoAD, mono_ops};
//!
//! let ops = mono_ops![sin, cos, exp];
//! let (value, grad_fn) = MonoAD::compute_grad(&ops, 2.0);
//! println!("f(2.0) = {}", value);
//! println!("f'(2.0) = {}", grad_fn(1.0));
//! ```
//!
//! ### Multi-variable function
//! ```
//! use petite_ad::Graph;
//!
//! let mut graph = Graph::new(2);
//! let x = graph.input(0);
//! let y = graph.input(1);
//! let sum = graph.add(x, y);
//! let sin_x = graph.sin(x);
//! graph.mul(sum, sin_x);
//!
//! let (value, grad_fn) = graph.compute_grad(&[0.6, 1.4]).unwrap();
//! let gradients = grad_fn(1.0);
//! println!("f(0.6, 1.4) = {}", value);
//! println!("∇f = {:?}", gradients);
//!
//! // Reuse the same graph but choose a different output node.
//! graph.set_output(sum).unwrap();
//! assert!((graph.compute(&[0.6, 1.4]).unwrap() - 2.0).abs() < 1e-10);
//!
//! // Or expose multiple outputs and get a Jacobian directly.
//! graph.set_outputs(&[sum, sin_x]).unwrap();
//! let values = graph.compute_many(&[0.6, 1.4]).unwrap();
//! assert_eq!(values.len(), 2);
//! let jacobian = graph.jacobian(&[0.6, 1.4]).unwrap();
//! assert_eq!(jacobian.len(), 2);
//! ```
//!
//! ### Forward-mode derivative
//! ```
//! use petite_ad::{ForwardAD, MonoAD, mono_ops};
//!
//! let exprs = mono_ops![sin, exp];
//! let result = ForwardAD::differentiate(&exprs, 0.5);
//! assert!(result.value.is_finite());
//! assert!(result.tangent.is_finite());
//! ```
//!
//! ### Graph validation/export
//! ```
//! use petite_ad::Graph;
//!
//! let mut graph = Graph::new(1);
//! let x = graph.input(0);
//! let neg_x = graph.neg(x);
//! graph.exp(neg_x);
//!
//! graph.validate().unwrap();
//! assert!(graph.to_mermaid().contains("flowchart LR"));
//! assert!(graph.to_dot().contains("digraph Graph"));
//! ```
//!
//! ### Reusable tape workspace
//! ```
//! use petite_ad::Graph;
//!
//! let mut graph = Graph::new(2);
//! let x = graph.input(0);
//! let y = graph.input(1);
//! let sum = graph.add(x, y);
//! graph.mul(sum, y);
//!
//! let tape = graph.compile();
//! let mut workspace = tape.workspace();
//! let (value, grad) = tape.gradient_with_workspace(&[2.0, 3.0], &mut workspace).unwrap();
//! assert!(value.is_finite());
//! assert_eq!(grad.len(), 2);
//! ```
//!
//! ### Checked-domain evaluation
//! ```
//! use petite_ad::{Graph, MonoAD, MultiAD, multi_ops};
//!
//! let mono_exprs = [MonoAD::Sqrt];
//! assert!(MonoAD::compute_checked(&mono_exprs, 4.0).is_ok());
//! assert!(MonoAD::compute_checked(&mono_exprs, -1.0).is_err());
//!
//! let exprs = multi_ops![(ln, 0)];
//! assert!(MultiAD::compute_checked(&exprs, &[2.0]).is_ok());
//! assert!(MultiAD::compute_checked(&exprs, &[0.0]).is_err());
//!
//! let mut graph = Graph::new(1);
//! let x = graph.input(0);
//! graph.ln(x);
//! assert!(graph.compute_checked(&[2.0]).is_ok());
//! assert!(graph.compute_checked(&[0.0]).is_err());
//!
//! let mut multi_graph = Graph::new(2);
//! let x = multi_graph.input(0);
//! let y = multi_graph.input(1);
//! let ratio = multi_graph.div(x, y);
//! let log_y = multi_graph.ln(y);
//! multi_graph.set_outputs(&[ratio, log_y]).unwrap();
//! assert!(multi_graph.compute_many_checked(&[4.0, 2.0]).is_ok());
//! assert!(multi_graph.jacobian_checked(&[4.0, 2.0]).is_ok());
//! assert!(multi_graph.compute_many_checked(&[4.0, 0.0]).is_err());
//! ```

mod error;
pub mod forward;
mod macros;
pub mod optim;

#[cfg(test)]
mod test_utils;

#[cfg(test)]
mod tests_comprehensive;

mod mono;
mod multi;

// Core types
pub use mono::MonoAD;

// Higher-order autodiff methods (exact Hessian computation)
pub use mono::{MonoAD2FR, MonoAD2RF, MonoAD2RR};

pub use multi::{
    builder::GraphBuilder, multi_ad_fr::MultiAD2FR, multi_ad_rf::MultiAD2RF,
    multi_ad_rr::MultiAD2RR, AcceleratorDeviceContext, AcceleratorDeviceKind, BackendCapabilities,
    BackendKind, BackendRejectionReason, BackendSupportReport, BatchGradients,
    BatchGradientsBuffer, BatchInputs, BatchLayout, BatchValues, BatchValuesBuffer, CompiledGraph,
    CompiledGraphMetadata, CompiledWorkspace, DeviceBackend, DeviceBatchPlan, DeviceBuffer,
    DeviceBufferHandle, DeviceBufferKind, DeviceBufferLayout, DeviceBufferSet, DeviceExecutionMode,
    DeviceExecutionTrace, DeviceMemoryLocation, DeviceTransferKind, DeviceTransferPlan,
    DeviceTransferPolicy, DomainPolicy, ExecutionBackend, ExprGraph, ExprNode, FlatInstruction,
    GpuBackendBoundary, GradientCheckEntry, GradientCheckReport, Graph, GraphNode, GraphStats,
    Instruction, MockDeviceBackend, MultiAD, NodeId, OpCode, ScalarBackend, SimdBackend, Tape,
    TapeWorkspace, UNUSED_NODE_ID,
};
#[cfg(feature = "backend-wgpu")]
pub use multi::{
    WgpuBackend, WgpuBuffer, WgpuBufferSet, WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES,
};

// Error handling
pub use error::{AutodiffError, Result};
pub use forward::{ForwardAD, ForwardValue};
pub use optim::{Adam, GradientDescent};

/// Type definitions for autodiff results and gradient functions.
///
/// This module provides type aliases for working with gradient computation results.
pub mod types {
    pub use crate::mono::types::{
        BackwardResultArc as MonoResultArc, BackwardResultBox as MonoResultBox,
        DynMathFn as MonoGradientFn,
    };
    pub use crate::multi::types::{
        BackwardResultArc as MultiResultArc, BackwardResultBox as MultiResultBox,
        DynGradFn as MultiGradientFn,
    };
}

/// Traits for implementing custom differentiable functions.
///
/// These traits allow you to define your own mathematical functions
/// with analytical gradients for testing and comparison purposes.
pub mod traits {
    pub use crate::mono::MonoFn;
    pub use crate::multi::MultiFn;
}
