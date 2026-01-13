//! # petite AD
//!
//! A pure Rust automatic differentiation library supporting both single-variable
//! and multi-variable functions with reverse-mode differentiation (backpropagation).
//!
//! ## Features
//!
//! - **Single-variable autodiff** - Chain operations like `sin`, `cos`, `exp`
//! - **Multi-variable autodiff** - Build computational graphs for multiple inputs
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
//! use petite_ad::{MultiAD, multi_ops};
//!
//! let exprs = multi_ops![
//!     (inp, 0),    // x₁
//!     (inp, 1),    // x₂
//!     (add, 0, 1), // x₁ + x₂
//!     (sin, 0),    // sin(x₁)
//!     (mul, 2, 3), // sin(x₁) * (x₁ + x₂)
//! ];
//!
//! let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]).unwrap();
//! let gradients = grad_fn(1.0);
//! println!("f(0.6, 1.4) = {}", value);
//! println!("∇f = {:?}", gradients);
//! ```

mod error;
mod macros;

#[cfg(test)]
mod test_utils;

mod mono;
mod multi;

// Core types
pub use mono::MonoAD;
pub use multi::builder::GraphBuilder;
pub use multi::MultiAD;

// Error handling
pub use error::{AutodiffError, Result};

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
