//! Exact second-order autodiff for multivariate functions using Reverse-over-Forward (RF) mode.
//!
//! This module implements the RF method for computing exact Hessians of
//! multivariate functions f: ℝⁿ → ℝ.
//!
//! # Supported Operations
//!
//! `MultiAD2RF` supports the smooth operation subset shared by the exact Hessian
//! engines: `Inp`, `Sin`, `Cos`, `Tan`, `Neg`, `Exp`, `Ln`, `Sqrt`, `Log1pExp`,
//! `Add`, `Sub`, `Mul`, `Div`, and `Pow`. Non-smooth operations like `Abs` are not supported by
//! the exact Hessian types. For Hessian approximation with the full operation set,
//! use [`MultiAD::compute_hessian`](crate::MultiAD::compute_hessian)
//! (finite-difference based).
//!
//! # Algorithm
//!
//! RF composes reverse-mode AD (outer) with forward-mode AD (inner):
//!
//! 1. **Forward pass** — evaluate the computation graph with **dual numbers**
//!    whose tangent component is seeded in direction e_j. This computes the
//!    directional derivative D_{e_j} f for each node.
//!
//! 2. **Reverse pass** — propagate **dual adjoints** backward to differentiate
//!    the directional derivative computation. Each adjoint is a dual
//!    `(val, tan)` where:
//!    - `val = ∂f/∂node`  (standard first-order adjoint)
//!    - `tan = ∂²f/(∂node · ∂x_j)`  (second-order cross-derivative)
//!
//!    The dual adjoint at input node k gives Hessian entry `H[j][k]`.
//!
//! # FR vs RF for scalar functions
//!
//! For scalar functions f: ℝⁿ → ℝ, FR and RF produce **identical computations**:
//! both require a dual-number forward pass followed by a dual-adjoint reverse pass
//! for each seed direction. The distinction is conceptual:
//!
//! | Aspect | FR | RF |
//! |--------|----|----|
//! | Inner AD | Reverse (gradient) | Forward (directional derivative) |
//! | Outer AD | Forward (differentiate gradient) | Reverse (differentiate directional deriv) |
//! | Implementation | Dual forward + dual reverse | Dual forward + dual reverse |
//!
//! The distinction becomes meaningful for vector-valued functions f: ℝⁿ → ℝᵐ (m > 1).
//!
//! # Accuracy
//!
//! Machine-precision exact (~1e-14 relative error). No finite differences.
//!
//! # Complexity
//!
//! O(n · G) where n = number of inputs, G = graph size.
//!
//! See [docs/multi_ad_hessian.md](../../docs/multi_ad_hessian.md) for complete theory.

use std::fmt;

use crate::Result;

use super::common::{compute_hessian_dual, OpKind};

/// Stack-based operation for multivariate second-order AD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiAD2RF {
    /// Input variable by index
    Inp(usize),
    /// sin(x)
    Sin,
    /// cos(x)
    Cos,
    /// tan(x)
    Tan,
    /// -x
    Neg,
    /// exp(x)
    Exp,
    /// ln(x)
    Ln,
    /// sqrt(x)
    Sqrt,
    /// stable ln(1 + exp(x))
    Log1pExp,
    /// a + b
    Add,
    /// a - b
    Sub,
    /// a · b
    Mul,
    /// a / b
    Div,
    /// a ^ b
    Pow,
}

impl fmt::Display for MultiAD2RF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiAD2RF::Inp(idx) => write!(f, "Inp({})", idx),
            MultiAD2RF::Sin => write!(f, "Sin"),
            MultiAD2RF::Cos => write!(f, "Cos"),
            MultiAD2RF::Tan => write!(f, "Tan"),
            MultiAD2RF::Neg => write!(f, "Neg"),
            MultiAD2RF::Exp => write!(f, "Exp"),
            MultiAD2RF::Ln => write!(f, "Ln"),
            MultiAD2RF::Sqrt => write!(f, "Sqrt"),
            MultiAD2RF::Log1pExp => write!(f, "Log1pExp"),
            MultiAD2RF::Add => write!(f, "Add"),
            MultiAD2RF::Sub => write!(f, "Sub"),
            MultiAD2RF::Mul => write!(f, "Mul"),
            MultiAD2RF::Div => write!(f, "Div"),
            MultiAD2RF::Pow => write!(f, "Pow"),
        }
    }
}

impl From<MultiAD2RF> for OpKind {
    fn from(op: MultiAD2RF) -> OpKind {
        match op {
            MultiAD2RF::Inp(k) => OpKind::Inp(k),
            MultiAD2RF::Sin => OpKind::Sin,
            MultiAD2RF::Cos => OpKind::Cos,
            MultiAD2RF::Tan => OpKind::Tan,
            MultiAD2RF::Neg => OpKind::Neg,
            MultiAD2RF::Exp => OpKind::Exp,
            MultiAD2RF::Ln => OpKind::Ln,
            MultiAD2RF::Sqrt => OpKind::Sqrt,
            MultiAD2RF::Log1pExp => OpKind::Log1pExp,
            MultiAD2RF::Add => OpKind::Add,
            MultiAD2RF::Sub => OpKind::Sub,
            MultiAD2RF::Mul => OpKind::Mul,
            MultiAD2RF::Div => OpKind::Div,
            MultiAD2RF::Pow => OpKind::Pow,
        }
    }
}

impl MultiAD2RF {
    /// Compute exact Hessian using Reverse-over-Forward mode.
    ///
    /// Delegates to the shared dual-number forward + reverse algorithm,
    /// which implements the RF composition for scalar functions.
    ///
    /// # Arguments
    ///
    /// * `ops` — RPN operation sequence
    /// * `x`   — input vector (n-dimensional)
    ///
    /// # Returns
    ///
    /// Hessian matrix `H` where `H[i][j] = ∂²f/∂xᵢ∂xⱼ`.
    ///
    /// # Errors
    ///
    /// Returns `Err(AutodiffError)` if an input index is out of bounds or the
    /// RPN expression is malformed.
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::MultiAD2RF;
    ///
    /// // f(x, y) = x·y  →  Hessian = [[0, 1], [1, 0]]
    /// let ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Inp(1), MultiAD2RF::Mul];
    /// let h = MultiAD2RF::compute_hessian(&ops, &[2.0, 3.0]).unwrap();
    /// assert_eq!(h[0][1], 1.0);
    /// assert_eq!(h[1][0], 1.0);
    /// ```
    pub fn compute_hessian(ops: &[MultiAD2RF], x: &[f64]) -> Result<Vec<Vec<f64>>> {
        let ops_repr: Vec<OpKind> = ops.iter().map(|&op| OpKind::from(op)).collect();
        compute_hessian_dual(&ops_repr, x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1e-12;

    #[test]
    fn test_product_xy() {
        let ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Inp(1), MultiAD2RF::Mul];
        let h = MultiAD2RF::compute_hessian(&ops, &[2.0, 3.0]).unwrap();
        assert!((h[0][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
        assert!((h[0][1] - 1.0).abs() < TOL);
        assert!((h[1][0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_simple_quadratic() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Inp(0),
            MultiAD2RF::Mul,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Inp(1),
            MultiAD2RF::Mul,
            MultiAD2RF::Add,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < TOL);
        assert!((h[1][1] - 2.0).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
    }

    #[test]
    fn test_empty_ops() {
        let h = MultiAD2RF::compute_hessian(&[], &[1.0, 2.0]).unwrap();
        assert_eq!(h.len(), 2);
        assert!(h.iter().all(|row| row.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_sin() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Sin,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Add,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.sin())).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_cos() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Cos,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Add,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.cos())).abs() < TOL);
    }

    #[test]
    fn test_exp() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Exp,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Add,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 1.0_f64.exp()).abs() < TOL);
    }

    #[test]
    fn test_three_variables_quadratic() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Inp(0),
            MultiAD2RF::Mul,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Inp(1),
            MultiAD2RF::Mul,
            MultiAD2RF::Add,
            MultiAD2RF::Inp(2),
            MultiAD2RF::Inp(2),
            MultiAD2RF::Mul,
            MultiAD2RF::Add,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0, 3.0]).unwrap();
        for (i, row) in h.iter().enumerate().take(3) {
            assert!((row[i] - 2.0).abs() < TOL, "H[{}][{}] = {}", i, i, row[i]);
        }
        assert!((h[0][1]).abs() < TOL);
        assert!((h[0][2]).abs() < TOL);
        assert!((h[1][2]).abs() < TOL);
    }

    #[test]
    fn test_sin_plus_exp() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Sin,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Exp,
            MultiAD2RF::Add,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.sin())).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1] - 2.0_f64.exp()).abs() < TOL);
    }

    #[test]
    fn test_sin_times_cos() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Sin,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Cos,
            MultiAD2RF::Mul,
        ];
        let x = 1.0;
        let y = 2.0;
        let h = MultiAD2RF::compute_hessian(&ops, &[x, y]).unwrap();
        let ex00 = -x.sin() * y.cos();
        let ex01 = -x.cos() * y.sin();
        assert!((h[0][0] - ex00).abs() < TOL, "H[0][0] = {}", h[0][0]);
        assert!((h[0][1] - ex01).abs() < TOL, "H[0][1] = {}", h[0][1]);
        assert!((h[1][0] - ex01).abs() < TOL, "H[1][0] = {}", h[1][0]);
        assert!((h[1][1] - ex00).abs() < TOL, "H[1][1] = {}", h[1][1]);
    }

    #[test]
    fn test_exp_times_exp() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Exp,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Exp,
            MultiAD2RF::Mul,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        let expected = (1.0_f64.exp()) * (2.0_f64.exp());
        assert!((h[0][0] - expected).abs() < TOL);
        assert!((h[0][1] - expected).abs() < TOL);
        assert!((h[1][1] - expected).abs() < TOL);
    }

    #[test]
    fn test_exp_times_sin_single_var() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Exp,
            MultiAD2RF::Inp(0),
            MultiAD2RF::Sin,
            MultiAD2RF::Mul,
        ];
        let x = 1.0;
        let h = MultiAD2RF::compute_hessian(&ops, &[x]).unwrap();
        let expected = 2.0 * x.exp() * x.cos();
        assert!((h[0][0] - expected).abs() < TOL, "H[0][0] = {}", h[0][0]);
    }

    #[test]
    fn test_sum_squared() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Inp(1),
            MultiAD2RF::Add,
            MultiAD2RF::Inp(0),
            MultiAD2RF::Inp(1),
            MultiAD2RF::Add,
            MultiAD2RF::Mul,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < TOL);
        assert!((h[0][1] - 2.0).abs() < TOL);
        assert!((h[1][1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_tan_and_sub() {
        let x = 0.3_f64;
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Tan,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Sub,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[x, 2.0]).unwrap();
        let sec_sq = 1.0 / x.cos().powi(2);
        let expected = 2.0 * sec_sq * x.tan();
        assert!((h[0][0] - expected).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_div_ln_sqrt() {
        let x = 4.0_f64;
        let y = 2.0_f64;
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Sqrt,
            MultiAD2RF::Inp(1),
            MultiAD2RF::Ln,
            MultiAD2RF::Div,
        ];
        let h = MultiAD2RF::compute_hessian(&ops, &[x, y]).unwrap();
        let sqrt_x = x.sqrt();
        let expected_xx = -1.0 / (4.0 * x * sqrt_x * y.ln());
        let expected_xy = -1.0 / (2.0 * sqrt_x * y * y.ln().powi(2));
        let expected_yy =
            2.0 * sqrt_x / (y * y * y.ln().powi(3)) + sqrt_x / (y * y * y.ln().powi(2));
        assert!((h[0][0] - expected_xx).abs() < 1e-10);
        assert!((h[0][1] - expected_xy).abs() < 1e-10);
        assert!((h[1][0] - expected_xy).abs() < 1e-10);
        assert!((h[1][1] - expected_yy).abs() < 1e-10);
    }

    #[test]
    fn test_pow_and_neg() {
        let x = 2.0_f64;
        let y = 3.0_f64;
        let pow_ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Inp(1), MultiAD2RF::Pow];
        let h = MultiAD2RF::compute_hessian(&pow_ops, &[x, y]).unwrap();
        let expected_xx = y * (y - 1.0) * x.powf(y - 2.0);
        let expected_xy = x.powf(y - 1.0) * (1.0 + y * x.ln());
        let expected_yy = x.powf(y) * x.ln().powi(2);
        assert!((h[0][0] - expected_xx).abs() < 1e-10);
        assert!((h[0][1] - expected_xy).abs() < 1e-10);
        assert!((h[1][0] - expected_xy).abs() < 1e-10);
        assert!((h[1][1] - expected_yy).abs() < 1e-10);

        let neg_ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Sin, MultiAD2RF::Neg];
        let h_neg = MultiAD2RF::compute_hessian(&neg_ops, &[0.5]).unwrap();
        assert!((h_neg[0][0] - 0.5_f64.sin()).abs() < 1e-10);
    }

    // ---- Additional coverage tests ----

    #[test]
    fn test_display_format() {
        assert_eq!(format!("{}", MultiAD2RF::Inp(2)), "Inp(2)");
        assert_eq!(format!("{}", MultiAD2RF::Sin), "Sin");
        assert_eq!(format!("{}", MultiAD2RF::Cos), "Cos");
        assert_eq!(format!("{}", MultiAD2RF::Tan), "Tan");
        assert_eq!(format!("{}", MultiAD2RF::Neg), "Neg");
        assert_eq!(format!("{}", MultiAD2RF::Exp), "Exp");
        assert_eq!(format!("{}", MultiAD2RF::Ln), "Ln");
        assert_eq!(format!("{}", MultiAD2RF::Sqrt), "Sqrt");
        assert_eq!(format!("{}", MultiAD2RF::Log1pExp), "Log1pExp");
        assert_eq!(format!("{}", MultiAD2RF::Add), "Add");
        assert_eq!(format!("{}", MultiAD2RF::Sub), "Sub");
        assert_eq!(format!("{}", MultiAD2RF::Mul), "Mul");
        assert_eq!(format!("{}", MultiAD2RF::Div), "Div");
        assert_eq!(format!("{}", MultiAD2RF::Pow), "Pow");
    }

    #[test]
    fn test_display_all_variants() {
        let ops = vec![
            MultiAD2RF::Inp(0),
            MultiAD2RF::Sin,
            MultiAD2RF::Cos,
            MultiAD2RF::Tan,
            MultiAD2RF::Neg,
            MultiAD2RF::Exp,
            MultiAD2RF::Ln,
            MultiAD2RF::Sqrt,
            MultiAD2RF::Log1pExp,
            MultiAD2RF::Add,
            MultiAD2RF::Sub,
            MultiAD2RF::Mul,
            MultiAD2RF::Div,
            MultiAD2RF::Pow,
        ];
        for op in &ops {
            let _s = format!("{}", op);
        }
    }

    #[test]
    fn test_hessian_sub_op() {
        // f(x, y) = x - y  →  H = [[0, 0], [0, 0]]
        let ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Inp(1), MultiAD2RF::Sub];
        let h = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0]).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_log1p_exp_hessian() {
        // f(x) = log1p_exp(x) = ln(1+exp(x))
        // f''(x) = exp(x)/(1+exp(x))^2
        let ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Log1pExp];
        let x = 1.0_f64;
        let h = MultiAD2RF::compute_hessian(&ops, &[x]).unwrap();
        let sigmoid = x.exp() / (1.0 + x.exp());
        let expected = sigmoid * (1.0 - sigmoid);
        assert!((h[0][0] - expected).abs() < 1e-10, "H[0][0] = {}", h[0][0]);
    }

    #[test]
    fn test_input_index_out_of_bounds() {
        let ops = vec![MultiAD2RF::Inp(5)];
        let result = MultiAD2RF::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unary_missing_operand() {
        let ops = vec![MultiAD2RF::Sin];
        let result = MultiAD2RF::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_missing_right_operand() {
        let ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Add];
        let result = MultiAD2RF::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_missing_both_operands() {
        let ops = vec![MultiAD2RF::Add];
        let result = MultiAD2RF::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_items_on_stack() {
        let ops = vec![MultiAD2RF::Inp(0), MultiAD2RF::Inp(1)];
        let result = MultiAD2RF::compute_hessian(&ops, &[1.0, 2.0]);
        assert!(result.is_err());
    }
}
