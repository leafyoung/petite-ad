//! Shared utilities for mono-variable FR/RF Hessian computation.
//!
//! For single-variable functions, Forward-over-Reverse (FR) and Reverse-over-Forward (RF)
//! are mathematically equivalent: both use dual-number arithmetic to differentiate the
//! gradient function. This module provides the shared implementation to avoid code
//! duplication between `MonoAD2FR` and `MonoAD2RF`.
//!
//! # Supported Operations
//!
//! The Hessian types (`MonoAD2FR`, `MonoAD2RF`, `MonoAD2RR`) support a subset of the
//! operations available in [`crate::MonoAD`]:
//!
//! | Operation | Description       | First derivative  | Second derivative |
//! |-----------|-------------------|-------------------|-------------------|
//! | `Sin`     | sin(x)            | cos(x)            | -sin(x)           |
//! | `Cos`     | cos(x)            | -sin(x)           | -cos(x)           |
//! | `Tan`     | tan(x)            | sec²(x)           | 2 sec²(x) tan(x)  |
//! | `Exp`     | exp(x)            | exp(x)            | exp(x)            |
//! | `Neg`     | -x                | -1                | 0                 |
//! | `Ln`      | ln(x)             | 1/x               | -1/x²             |
//! | `Sqrt`    | sqrt(x)           | 1/(2 sqrt(x))     | -1/(4x sqrt(x))   |
//! | `Abs`     | abs(x)            | sign(x)           | 0                 |
//!
//! `Abs` is non-smooth at zero; exact Hessian types follow the same raw `f64`
//! convention as [`crate::MonoAD`] by using derivative `0` and curvature `0` at `x = 0`.
//! Operations like `Pow` are not yet supported in the
//! Hessian types. Use [`crate::MonoAD`] for first-order differentiation with the full
//! operation set, or `MultiAD::compute_hessian` for finite-difference Hessian
//! approximation with all operations.

use crate::{AutodiffError, Result};

use super::types::*;

/// Operation kind for mono-variable second-order AD (shared by FR and RF).
///
/// This enum abstracts over `MonoAD2FR` and `MonoAD2RF`, which have identical
/// variants. It is used by the shared [`compute_hessian_dual`] implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MonoHessianOpKind {
    Sin,
    Cos,
    Tan,
    Exp,
    Neg,
    Ln,
    Sqrt,
    Abs,
}

/// Validate real-domain restrictions for checked mono Hessian evaluation.
#[inline(always)]
pub(crate) fn check_domain(op: MonoHessianOpKind, x: f64) -> Result<()> {
    match op {
        MonoHessianOpKind::Ln if x <= 0.0 => {
            Err(AutodiffError::domain("Ln", "input must be positive"))
        }
        MonoHessianOpKind::Sqrt if x < 0.0 => {
            Err(AutodiffError::domain("Sqrt", "input must be non-negative"))
        }
        _ => Ok(()),
    }
}

/// Evaluate a single operation on a dual number.
#[inline(always)]
pub(crate) fn forward_dual(op: MonoHessianOpKind, x: Dual) -> Dual {
    match op {
        MonoHessianOpKind::Sin => Dual {
            val: x.val.sin(),
            tan: x.val.cos() * x.tan,
        },
        MonoHessianOpKind::Cos => Dual {
            val: x.val.cos(),
            tan: -x.val.sin() * x.tan,
        },
        MonoHessianOpKind::Tan => {
            let cos_x = x.val.cos();
            let sec_sq = 1.0 / cos_x.powi(2);
            Dual {
                val: x.val.tan(),
                tan: sec_sq * x.tan,
            }
        }
        MonoHessianOpKind::Exp => Dual {
            val: x.val.exp(),
            tan: x.val.exp() * x.tan,
        },
        MonoHessianOpKind::Neg => Dual {
            val: -x.val,
            tan: -x.tan,
        },
        MonoHessianOpKind::Ln => Dual {
            val: x.val.ln(),
            tan: x.tan / x.val,
        },
        MonoHessianOpKind::Sqrt => {
            let sqrt_x = x.val.sqrt();
            Dual {
                val: sqrt_x,
                tan: x.tan / (2.0 * sqrt_x),
            }
        }
        MonoHessianOpKind::Abs => Dual {
            val: x.val.abs(),
            tan: x.tan * sign_or_zero(x.val),
        },
    }
}

/// Evaluate a single operation's scalar value.
#[inline(always)]
pub(crate) fn eval_scalar(op: MonoHessianOpKind, x: f64) -> f64 {
    match op {
        MonoHessianOpKind::Sin => x.sin(),
        MonoHessianOpKind::Cos => x.cos(),
        MonoHessianOpKind::Tan => x.tan(),
        MonoHessianOpKind::Exp => x.exp(),
        MonoHessianOpKind::Neg => -x,
        MonoHessianOpKind::Ln => x.ln(),
        MonoHessianOpKind::Sqrt => x.sqrt(),
        MonoHessianOpKind::Abs => x.abs(),
    }
}

/// Evaluate a single operation's scalar value with checked-domain validation.
#[inline(always)]
pub(crate) fn eval_scalar_checked(op: MonoHessianOpKind, x: f64) -> Result<f64> {
    check_domain(op, x)?;
    Ok(eval_scalar(op, x))
}

/// Create a first-order backward closure for an operation at a given input value.
pub(crate) fn make_backward_fn(op: MonoHessianOpKind, x: f64) -> Box<DynMathFn> {
    match op {
        MonoHessianOpKind::Sin => Box::new(move |dy: f64| -> f64 { dy * x.cos() }),
        MonoHessianOpKind::Cos => Box::new(move |dy: f64| -> f64 { dy * -x.sin() }),
        MonoHessianOpKind::Tan => {
            let sec_sq = 1.0 / x.cos().powi(2);
            Box::new(move |dy: f64| -> f64 { dy * sec_sq })
        }
        MonoHessianOpKind::Exp => {
            let exp_val = x.exp();
            Box::new(move |dy: f64| -> f64 { dy * exp_val })
        }
        MonoHessianOpKind::Neg => Box::new(move |dy: f64| -> f64 { -dy }),
        MonoHessianOpKind::Ln => Box::new(move |dy: f64| -> f64 { dy / x }),
        MonoHessianOpKind::Sqrt => {
            let sqrt_x = x.sqrt();
            Box::new(move |dy: f64| -> f64 { dy / (2.0 * sqrt_x) })
        }
        MonoHessianOpKind::Abs => {
            let sign = sign_or_zero(x);
            Box::new(move |dy: f64| -> f64 { dy * sign })
        }
    }
}

/// Return the sign of `x`, using `0` at `x = 0` to match [`crate::MonoAD`].
#[inline]
pub(crate) fn sign_or_zero(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Compute the second derivative for a single-operation expression using dual numbers.
///
/// For multi-operation compositions, returns `None` (caller should delegate to RR).
pub(crate) fn compute_single_op_hessian(op: MonoHessianOpKind, x: f64) -> Option<f64> {
    match op {
        MonoHessianOpKind::Sin => Some(-x.sin()),
        MonoHessianOpKind::Cos => Some(-x.cos()),
        MonoHessianOpKind::Tan => {
            let sec_sq = 1.0 / x.cos().powi(2);
            Some(2.0 * sec_sq * x.tan())
        }
        MonoHessianOpKind::Exp => Some(x.exp()),
        MonoHessianOpKind::Neg => Some(0.0),
        MonoHessianOpKind::Ln => Some(-1.0 / x.powi(2)),
        MonoHessianOpKind::Sqrt => {
            let sqrt_x = x.sqrt();
            Some(-1.0 / (4.0 * x * sqrt_x))
        }
        MonoHessianOpKind::Abs => Some(0.0),
    }
}

/// Compute forward pass with checked-domain validation.
pub(crate) fn compute_forward_checked(ops: &[MonoHessianOpKind], x: f64) -> Result<f64> {
    let mut value = x;
    for &op in ops {
        value = eval_scalar_checked(op, value)?;
    }
    Ok(value)
}

/// Generic gradient computation using reverse-mode (first-order only).
///
/// Shared by `MonoAD2FR` and `MonoAD2RF` to avoid duplicating the backward pass logic.
pub(crate) fn compute_grad_generic<W>(ops: &[MonoHessianOpKind], x: f64) -> (f64, W)
where
    W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
{
    let mut value = x;
    let mut backprops: Vec<W> = Vec::new();

    for &op in ops {
        let new_value = eval_scalar(op, value);
        let backprop: W = W::from(make_backward_fn(op, value));
        value = new_value;
        backprops.push(backprop);
    }

    let backward_fn = Box::new(move |cotangent: f64| -> f64 {
        let mut grad = cotangent;
        for backprop in backprops.iter().rev() {
            grad = backprop(grad);
        }
        grad
    });

    (value, W::from(backward_fn))
}

/// Generic checked gradient computation using reverse-mode (first-order only).
pub(crate) fn compute_grad_generic_checked<W>(ops: &[MonoHessianOpKind], x: f64) -> Result<(f64, W)>
where
    W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
{
    let mut value = x;
    let mut backprops: Vec<W> = Vec::new();

    for &op in ops {
        check_domain(op, value)?;
        let new_value = eval_scalar(op, value);
        let backprop = W::from(make_backward_fn(op, value));
        value = new_value;
        backprops.push(backprop);
    }

    let backward_fn = Box::new(move |cotangent: f64| -> f64 {
        let mut grad = cotangent;
        for backprop in backprops.iter().rev() {
            grad = backprop(grad);
        }
        grad
    });

    Ok((value, W::from(backward_fn)))
}

/// Compute forward pass only using the shared operation kind.
pub(crate) fn compute_forward(ops: &[MonoHessianOpKind], x: f64) -> f64 {
    let mut dual = Dual::variable(x);
    for &op in ops {
        dual = forward_dual(op, dual);
    }
    dual.val
}
