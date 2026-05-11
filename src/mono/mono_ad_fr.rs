//! Exact second-order autodiff using Forward-over-Reverse (FR) mode.
//!
//! This module implements the **Forward-over-Reverse (FR)** method for computing exact
//! Hessians (second derivatives) of single-variable functions. It combines reverse-mode
//! gradient computation with forward-mode differentiation on the gradient function.
//!
//! # Comprehensive Documentation
//!
//! For complete mathematical theory, detailed derivations, and comparison with other methods, see:
//! **[`/docs/mono_ad_hessian.md`](../../docs/mono_ad_hessian.md)**
//!
//! # Supported Operations
//!
//! This type supports a subset of operations compared to [`crate::MonoAD`]:
//! `Sin`, `Cos`, `Tan`, `Exp`, `Neg`, `Ln`, `Sqrt`, `Abs`. See [`super::mono_hessian_common`] for details.
//!
//! # Mathematical Foundation
//!
//! The Forward-over-Reverse method computes the Hessian by treating the gradient function
//! itself as a differentiable function and applying forward-mode AD to it.
//!
//! For a function f(x):
//! 1. **Reverse pass**: Compute g(x) = f'(x) using reverse-mode AD
//! 2. **Forward pass**: Compute g'(x) = d/dx[f'(x)] = f''(x) using forward-mode AD with dual numbers
//!
//! # Algorithm
//!
//! ## For Single Operations
//!
//! Direct dual-number evaluation of the second derivative.
//!
//! ## For Multiple Operations (Compositions)
//!
//! Delegates to RR mode ([`crate::MonoAD2RR`]), which handles general compositions
//! correctly.
//!
//! # Accuracy
//!
//! - **Method**: Exact symbolic differentiation via dual numbers
//! - **Error source**: Only floating-point rounding (machine epsilon)
//! - **Typical relative error**: < 1e-14
//! - **No truncation error**: Mathematically exact (unlike finite differences)
//!
//! # Example Usage
//!
//! ```rust
//! use petite_ad::MonoAD2FR;
//!
//! // f(x) = sin(x), f''(x) = -sin(x)
//! let ops = [MonoAD2FR::Sin];
//! let x = 1.0;
//!
//! let value = MonoAD2FR::compute(&ops, x);
//! let hessian = MonoAD2FR::compute_hessian(&ops, x);
//!
//! println!("f({}) = {}", x, value);        // sin(1) ≈ 0.8414
//! println!("f''({}) = {}", x, hessian);    // -sin(1) ≈ -0.8414
//! ```
//!
//! # References
//!
//! - Griewank & Walther (2008): *Evaluating Derivatives*
//! - Naumann (2012): *The Art of Differentiating Computer Programs*
//! - Pearlmutter (1994): "Fast exact multiplication by the Hessian"

use crate::Result;

use super::mono_ad_rr::MonoAD2RR;
use super::mono_hessian_common::{self, MonoHessianOpKind};
use super::types::*;

/// Single-variable automatic differentiation operations for Forward-over-Reverse Hessian computation.
///
/// Supports: `Sin`, `Cos`, `Tan`, `Exp`, `Neg`, `Ln`, `Sqrt`, `Abs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonoAD2FR {
    Sin,
    Cos,
    Tan,
    Exp,
    Neg,
    Ln,
    Sqrt,
    Abs,
}

impl From<MonoAD2FR> for MonoHessianOpKind {
    fn from(op: MonoAD2FR) -> MonoHessianOpKind {
        match op {
            MonoAD2FR::Sin => MonoHessianOpKind::Sin,
            MonoAD2FR::Cos => MonoHessianOpKind::Cos,
            MonoAD2FR::Tan => MonoHessianOpKind::Tan,
            MonoAD2FR::Exp => MonoHessianOpKind::Exp,
            MonoAD2FR::Neg => MonoHessianOpKind::Neg,
            MonoAD2FR::Ln => MonoHessianOpKind::Ln,
            MonoAD2FR::Sqrt => MonoHessianOpKind::Sqrt,
            MonoAD2FR::Abs => MonoHessianOpKind::Abs,
        }
    }
}

impl MonoAD2FR {
    /// Compute forward pass only.
    pub fn compute(exprs: &[MonoAD2FR], x: f64) -> f64 {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        mono_hessian_common::compute_forward(&ops, x)
    }

    /// Compute forward pass with opt-in checked-domain validation.
    pub fn compute_checked(exprs: &[MonoAD2FR], x: f64) -> Result<f64> {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        mono_hessian_common::compute_forward_checked(&ops, x)
    }

    /// Compute forward pass and return gradient function using reverse-mode.
    pub fn compute_grad(exprs: &[MonoAD2FR], x: f64) -> BackwardResultBox {
        Self::compute_grad_generic::<Box<DynMathFn>>(exprs, x)
    }

    /// Compute forward pass and gradient function with checked-domain validation.
    pub fn compute_grad_checked(exprs: &[MonoAD2FR], x: f64) -> Result<BackwardResultBox> {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        mono_hessian_common::compute_grad_generic_checked::<Box<DynMathFn>>(&ops, x)
    }

    fn compute_grad_generic<W>(exprs: &[MonoAD2FR], x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
    {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        mono_hessian_common::compute_grad_generic::<W>(&ops, x)
    }

    /// Compute exact Hessian using Forward-over-Reverse mode.
    ///
    /// For single-operation expressions, uses direct dual-number differentiation.
    /// For multi-operation compositions, delegates to [`crate::MonoAD2RR`] (which
    /// handles compositions correctly via explicit second-order chain rule).
    ///
    /// # Arguments
    ///
    /// * `exprs` - Slice of operations to apply in sequence
    /// * `x` - Input value to evaluate at
    ///
    /// # Returns
    ///
    /// The exact second derivative f''(x) at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::MonoAD2FR;
    ///
    /// // f(x) = sin(x), f''(x) = -sin(x)
    /// let ops = [MonoAD2FR::Sin];
    /// let x = 0.5;
    /// let hessian = MonoAD2FR::compute_hessian(&ops, x);
    /// assert!((hessian - (-0.5_f64.sin())).abs() < 1e-12);
    ///
    /// // Composition f(x) = exp(sin(x)) — uses RR fallback
    /// let ops = [MonoAD2FR::Sin, MonoAD2FR::Exp];
    /// let x = 1.0;
    /// let hessian = MonoAD2FR::compute_hessian(&ops, x);
    /// let expected = x.sin().exp() * (x.cos().powi(2) - x.sin());
    /// assert!((hessian - expected).abs() < 1e-12);
    /// ```
    pub fn compute_hessian(exprs: &[MonoAD2FR], x: f64) -> f64 {
        if exprs.is_empty() {
            return 0.0;
        }

        // Single operation: direct dual-number evaluation
        if let [op] = exprs {
            return mono_hessian_common::compute_single_op_hessian((*op).into(), x)
                .expect("all single ops supported");
        }

        let rr_ops = Self::to_rr_ops(exprs);
        MonoAD2RR::compute_hessian(&rr_ops, x)
    }

    /// Compute exact Hessian with checked-domain validation.
    pub fn compute_hessian_checked(exprs: &[MonoAD2FR], x: f64) -> Result<f64> {
        let rr_ops = Self::to_rr_ops(exprs);
        MonoAD2RR::compute_hessian_checked(&rr_ops, x)
    }

    fn to_rr_ops(exprs: &[MonoAD2FR]) -> Vec<MonoAD2RR> {
        exprs
            .iter()
            .map(|&op| match op {
                MonoAD2FR::Sin => MonoAD2RR::Sin,
                MonoAD2FR::Cos => MonoAD2RR::Cos,
                MonoAD2FR::Tan => MonoAD2RR::Tan,
                MonoAD2FR::Exp => MonoAD2RR::Exp,
                MonoAD2FR::Neg => MonoAD2RR::Neg,
                MonoAD2FR::Ln => MonoAD2RR::Ln,
                MonoAD2FR::Sqrt => MonoAD2RR::Sqrt,
                MonoAD2FR::Abs => MonoAD2RR::Abs,
            })
            .collect()
    }
}
