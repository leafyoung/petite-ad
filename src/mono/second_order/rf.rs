//! Exact second-order autodiff using Reverse-over-Forward (RF) mode.
//!
//! This module implements the **Reverse-over-Forward (RF)** method for computing exact
//! Hessians (second derivatives) of single-variable functions.
//!
//! # Comprehensive Documentation
//!
//! For complete mathematical theory, detailed derivations, and comparison with other methods, see:
//! **[`/docs/mono_ad_hessian.md`](../../docs/mono_ad_hessian.md)**
//!
//! # Supported Operations
//!
//! This type supports a subset of operations compared to [`crate::MonoAD`]:
//! `Sin`, `Cos`, `Tan`, `Exp`, `Neg`, `Ln`, `Sqrt`, `Abs`. See [`super::common`] for details.
//!
//! For single-variable functions, RF and FR are mathematically equivalent (both use dual
//! numbers), but they differ in conceptual organization:
//!
//! - **FR**: Reverse-mode for gradient, then forward-mode on the gradient function
//! - **RF**: Forward-mode embedded within reverse-mode computation
//!
//! For a function f(x):
//! 1. Use dual numbers to track both value and derivative
//! 2. The tangent component propagates derivatives through operations
//! 3. Result: f''(x) from the final tangent
//!
//! ## Dual Number Arithmetic
//!
//! A dual number is a pair (val, tan) where:
//! - `val`: the function value
//! - `tan`: the tangent (derivative) value
//!
//! For an input x with seed tangent 1.0: Dual::variable(x) = (x, 1.0)
//!
//! Each operation propagates derivatives using the chain rule:
//!
//! ```text
//! Operation | Value         | Tangent (derivative)
//! ----------|---------------|----------------------
//! sin(x)    | sin(x.val)    | cos(x.val) · x.tan
//! cos(x)    | cos(x.val)    | -sin(x.val) · x.tan
//! tan(x)    | tan(x.val)    | sec²(x.val) · x.tan
//! exp(x)    | exp(x.val)    | exp(x.val) · x.tan
//! -x        | -x.val        | -x.tan
//! ln(x)     | ln(x.val)     | x.tan / x.val
//! sqrt(x)   | sqrt(x.val)   | x.tan / (2 sqrt(x.val))
//! abs(x)    | abs(x.val)    | sign_or_zero(x.val) · x.tan
//! ```
//!
//! ## Gradient Functions
//!
//! For elementary operations, the gradient functions are:
//!
//! | Original f(x) | Gradient g(x) = f'(x) | Second derivative g'(x) = f''(x) |
//! |---------------|----------------------|----------------------------------|
//! | sin(x)        | cos(x)               | -sin(x)                          |
//! | cos(x)        | -sin(x)              | -cos(x)                          |
//! | exp(x)        | exp(x)               | exp(x)                           |
//! | -x            | -1                   | 0                                |
//! | ln(x)         | 1/x                  | -1/x²                            |
//! | sqrt(x)       | 1/(2√x)              | -1/(4x√x)                        |
//! | abs(x)        | sign(x)              | 0 (raw convention at x = 0)       |
//!
//! ## RF vs FR: What's the Difference?
//!
//! For **single-variable** functions, RF and FR are computationally identical:
//! - Both use dual number arithmetic
//! - Both compute the same second derivatives
//! - Both have the same complexity
//!
//! The distinction becomes meaningful for **multi-variable** functions:
//! - **FR**: Compute full Jacobian (reverse), then differentiate each row (forward)
//! - **RF**: Differentiate function (forward) for each output (reverse)
//!
//! For MonoAD (single variable), this implementation is shared with FR via
//! [`super::common`].
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
//! use petite_ad::MonoAD2RF;
//!
//! // f(x) = cos(x), f''(x) = -cos(x)
//! let ops = [MonoAD2RF::Cos];
//! let x = 1.0;
//!
//! let value = MonoAD2RF::compute(&ops, x);
//! let hessian = MonoAD2RF::compute_hessian(&ops, x);
//!
//! println!("f({}) = {}", x, value);        // cos(1) ≈ 0.5403
//! println!("f''({}) = {}", x, hessian);    // -cos(1) ≈ -0.5403
//! ```
//!
//! # References
//!
//! - Griewank & Walther (2008): *Evaluating Derivatives*
//! - Naumann (2012): *The Art of Differentiating Computer Programs*
//! - Giles (2008): "Collected matrix derivative results for forward and reverse mode AD"

use crate::Result;

use super::common::{self, MonoHessianOpKind};
use super::rr::MonoAD2RR;
use crate::mono::types::*;

/// Single-variable automatic differentiation operations for Reverse-over-Forward Hessian computation.
///
/// Supports: `Sin`, `Cos`, `Tan`, `Exp`, `Neg`, `Ln`, `Sqrt`, `Abs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonoAD2RF {
    Sin,
    Cos,
    Tan,
    Exp,
    Neg,
    Ln,
    Sqrt,
    Abs,
}

impl From<MonoAD2RF> for MonoHessianOpKind {
    fn from(op: MonoAD2RF) -> MonoHessianOpKind {
        match op {
            MonoAD2RF::Sin => MonoHessianOpKind::Sin,
            MonoAD2RF::Cos => MonoHessianOpKind::Cos,
            MonoAD2RF::Tan => MonoHessianOpKind::Tan,
            MonoAD2RF::Exp => MonoHessianOpKind::Exp,
            MonoAD2RF::Neg => MonoHessianOpKind::Neg,
            MonoAD2RF::Ln => MonoHessianOpKind::Ln,
            MonoAD2RF::Sqrt => MonoHessianOpKind::Sqrt,
            MonoAD2RF::Abs => MonoHessianOpKind::Abs,
        }
    }
}

impl MonoAD2RF {
    /// Compute forward pass only.
    pub fn compute(exprs: &[MonoAD2RF], x: f64) -> f64 {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        common::compute_forward(&ops, x)
    }

    /// Compute forward pass with opt-in checked-domain validation.
    pub fn compute_checked(exprs: &[MonoAD2RF], x: f64) -> Result<f64> {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        common::compute_forward_checked(&ops, x)
    }

    /// Compute forward pass and return gradient function using reverse-mode.
    pub fn compute_grad(exprs: &[MonoAD2RF], x: f64) -> BackwardResultBox {
        Self::compute_grad_generic::<Box<DynMathFn>>(exprs, x)
    }

    /// Compute forward pass and gradient function with checked-domain validation.
    pub fn compute_grad_checked(exprs: &[MonoAD2RF], x: f64) -> Result<BackwardResultBox> {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        common::compute_grad_generic_checked::<Box<DynMathFn>>(&ops, x)
    }

    fn compute_grad_generic<W>(exprs: &[MonoAD2RF], x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
    {
        let ops: Vec<MonoHessianOpKind> = exprs.iter().map(|&op| op.into()).collect();
        common::compute_grad_generic::<W>(&ops, x)
    }

    /// Compute exact Hessian using Reverse-over-Forward mode.
    ///
    /// For single-variable functions, this is computationally identical to
    /// [`crate::MonoAD2FR`] but represents a different conceptual organization.
    ///
    /// For single-operation expressions, uses direct dual-number differentiation.
    /// For multi-operation compositions, delegates to [`crate::MonoAD2RR`].
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
    /// use petite_ad::MonoAD2RF;
    ///
    /// // f(x) = cos(x), f''(x) = -cos(x)
    /// let ops = [MonoAD2RF::Cos];
    /// let x = 0.5;
    /// let hessian = MonoAD2RF::compute_hessian(&ops, x);
    /// assert!((hessian - (-0.5_f64.cos())).abs() < 1e-12);
    ///
    /// // Composition f(x) = sin(sin(x)) — uses RR fallback
    /// let ops = [MonoAD2RF::Sin, MonoAD2RF::Sin];
    /// let x = 0.5;
    /// let hessian = MonoAD2RF::compute_hessian(&ops, x);
    /// let sin_x = x.sin();
    /// let cos_x = x.cos();
    /// let expected = -sin_x.sin() * cos_x.powi(2) - sin_x.cos() * x.sin();
    /// assert!((hessian - expected).abs() < 1e-12);
    /// ```
    pub fn compute_hessian(exprs: &[MonoAD2RF], x: f64) -> f64 {
        if exprs.is_empty() {
            return 0.0;
        }

        // Single operation: direct dual-number evaluation
        if let [op] = exprs {
            return common::compute_single_op_hessian((*op).into(), x)
                .expect("all single ops supported");
        }

        let rr_ops = Self::to_rr_ops(exprs);
        MonoAD2RR::compute_hessian(&rr_ops, x)
    }

    /// Compute exact Hessian with checked-domain validation.
    pub fn compute_hessian_checked(exprs: &[MonoAD2RF], x: f64) -> Result<f64> {
        let rr_ops = Self::to_rr_ops(exprs);
        MonoAD2RR::compute_hessian_checked(&rr_ops, x)
    }

    fn to_rr_ops(exprs: &[MonoAD2RF]) -> Vec<MonoAD2RR> {
        exprs
            .iter()
            .map(|&op| match op {
                MonoAD2RF::Sin => MonoAD2RR::Sin,
                MonoAD2RF::Cos => MonoAD2RR::Cos,
                MonoAD2RF::Tan => MonoAD2RR::Tan,
                MonoAD2RF::Exp => MonoAD2RR::Exp,
                MonoAD2RF::Neg => MonoAD2RR::Neg,
                MonoAD2RF::Ln => MonoAD2RR::Ln,
                MonoAD2RF::Sqrt => MonoAD2RR::Sqrt,
                MonoAD2RF::Abs => MonoAD2RR::Abs,
            })
            .collect()
    }
}
