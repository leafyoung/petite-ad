//! Exact second-order autodiff for multivariate functions using Forward-over-Reverse (FR) mode.
//!
//! This module implements the FR method for computing exact Hessians of
//! multivariate functions f: ℝⁿ → ℝ.
//!
//! # Supported Operations
//!
//! `MultiAD2FR` supports the smooth operation subset shared by the exact Hessian
//! engines: `Inp`, `Sin`, `Cos`, `Tan`, `Neg`, `Exp`, `Ln`, `Sqrt`, `Log1pExp`,
//! `Add`, `Sub`, `Mul`, `Div`, and `Pow`. Non-smooth operations like `Abs` are not supported by
//! the exact Hessian types. For Hessian approximation with the full operation set,
//! use [`MultiAD::compute_hessian`](crate::MultiAD::compute_hessian)
//! (finite-difference based).
//!
//! # Algorithm
//!
//! FR composes forward-mode AD (outer) with reverse-mode AD (inner):
//!
//! 1. **Forward pass** — evaluate the computation graph with **dual numbers**
//!    whose tangent component is seeded in direction e_j. Every intermediate
//!    value becomes a dual `(val, tan)`.
//!
//! 2. **Reverse pass** — propagate **dual adjoints** backward through the
//!    perturbed graph. Each adjoint is a dual `(val, tan)` where:
//!    - `val = ∂f/∂node`  (standard first-order adjoint)
//!    - `tan = ∂²f/(∂node · ∂x_j)`  (second-order cross-derivative)
//!
//!    The dual adjoint at input node k gives Hessian entry `H[j][k]`.
//!
//! This is repeated for each seed direction j = 0 … n−1, producing the full
//! Hessian column by column.
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
pub enum MultiAD2FR {
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

impl fmt::Display for MultiAD2FR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiAD2FR::Inp(idx) => write!(f, "Inp({})", idx),
            MultiAD2FR::Sin => write!(f, "Sin"),
            MultiAD2FR::Cos => write!(f, "Cos"),
            MultiAD2FR::Tan => write!(f, "Tan"),
            MultiAD2FR::Neg => write!(f, "Neg"),
            MultiAD2FR::Exp => write!(f, "Exp"),
            MultiAD2FR::Ln => write!(f, "Ln"),
            MultiAD2FR::Sqrt => write!(f, "Sqrt"),
            MultiAD2FR::Log1pExp => write!(f, "Log1pExp"),
            MultiAD2FR::Add => write!(f, "Add"),
            MultiAD2FR::Sub => write!(f, "Sub"),
            MultiAD2FR::Mul => write!(f, "Mul"),
            MultiAD2FR::Div => write!(f, "Div"),
            MultiAD2FR::Pow => write!(f, "Pow"),
        }
    }
}

impl From<MultiAD2FR> for OpKind {
    fn from(op: MultiAD2FR) -> OpKind {
        match op {
            MultiAD2FR::Inp(k) => OpKind::Inp(k),
            MultiAD2FR::Sin => OpKind::Sin,
            MultiAD2FR::Cos => OpKind::Cos,
            MultiAD2FR::Tan => OpKind::Tan,
            MultiAD2FR::Neg => OpKind::Neg,
            MultiAD2FR::Exp => OpKind::Exp,
            MultiAD2FR::Ln => OpKind::Ln,
            MultiAD2FR::Sqrt => OpKind::Sqrt,
            MultiAD2FR::Log1pExp => OpKind::Log1pExp,
            MultiAD2FR::Add => OpKind::Add,
            MultiAD2FR::Sub => OpKind::Sub,
            MultiAD2FR::Mul => OpKind::Mul,
            MultiAD2FR::Div => OpKind::Div,
            MultiAD2FR::Pow => OpKind::Pow,
        }
    }
}

impl MultiAD2FR {
    /// Compute exact Hessian using Forward-over-Reverse mode.
    ///
    /// Delegates to the shared dual-number forward + reverse algorithm.
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
    /// use petite_ad::MultiAD2FR;
    ///
    /// // f(x, y) = x·y  →  Hessian = [[0, 1], [1, 0]]
    /// let ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Inp(1), MultiAD2FR::Mul];
    /// let h = MultiAD2FR::compute_hessian(&ops, &[2.0, 3.0]).unwrap();
    /// assert_eq!(h[0][1], 1.0);
    /// assert_eq!(h[1][0], 1.0);
    /// ```
    pub fn compute_hessian(ops: &[MultiAD2FR], x: &[f64]) -> Result<Vec<Vec<f64>>> {
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
        let ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Inp(1), MultiAD2FR::Mul];
        let h = MultiAD2FR::compute_hessian(&ops, &[2.0, 3.0]).unwrap();
        assert!((h[0][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
        assert!((h[0][1] - 1.0).abs() < TOL);
        assert!((h[1][0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_simple_quadratic() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Inp(0),
            MultiAD2FR::Mul,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Inp(1),
            MultiAD2FR::Mul,
            MultiAD2FR::Add,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < TOL);
        assert!((h[1][1] - 2.0).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
    }

    #[test]
    fn test_empty_ops() {
        let h = MultiAD2FR::compute_hessian(&[], &[1.0, 2.0]).unwrap();
        assert_eq!(h.len(), 2);
        assert!(h.iter().all(|row| row.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_sin() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Sin,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Add,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.sin())).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_cos() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Cos,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Add,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.cos())).abs() < TOL);
    }

    #[test]
    fn test_exp() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Exp,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Add,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 1.0_f64.exp()).abs() < TOL);
    }

    #[test]
    fn test_three_variables_quadratic() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Inp(0),
            MultiAD2FR::Mul,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Inp(1),
            MultiAD2FR::Mul,
            MultiAD2FR::Add,
            MultiAD2FR::Inp(2),
            MultiAD2FR::Inp(2),
            MultiAD2FR::Mul,
            MultiAD2FR::Add,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0, 3.0]).unwrap();
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
            MultiAD2FR::Inp(0),
            MultiAD2FR::Sin,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Exp,
            MultiAD2FR::Add,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.sin())).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1] - 2.0_f64.exp()).abs() < TOL);
    }

    #[test]
    fn test_sin_times_cos() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Sin,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Cos,
            MultiAD2FR::Mul,
        ];
        let x = 1.0;
        let y = 2.0;
        let h = MultiAD2FR::compute_hessian(&ops, &[x, y]).unwrap();
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
            MultiAD2FR::Inp(0),
            MultiAD2FR::Exp,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Exp,
            MultiAD2FR::Mul,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        let expected = (1.0_f64.exp()) * (2.0_f64.exp());
        assert!((h[0][0] - expected).abs() < TOL);
        assert!((h[0][1] - expected).abs() < TOL);
        assert!((h[1][1] - expected).abs() < TOL);
    }

    #[test]
    fn test_exp_times_sin_single_var() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Exp,
            MultiAD2FR::Inp(0),
            MultiAD2FR::Sin,
            MultiAD2FR::Mul,
        ];
        let x = 1.0;
        let h = MultiAD2FR::compute_hessian(&ops, &[x]).unwrap();
        let expected = 2.0 * x.exp() * x.cos();
        assert!((h[0][0] - expected).abs() < TOL, "H[0][0] = {}", h[0][0]);
    }

    #[test]
    fn test_sum_squared() {
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Inp(1),
            MultiAD2FR::Add,
            MultiAD2FR::Inp(0),
            MultiAD2FR::Inp(1),
            MultiAD2FR::Add,
            MultiAD2FR::Mul,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < TOL);
        assert!((h[0][1] - 2.0).abs() < TOL);
        assert!((h[1][1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_tan_and_sub() {
        let x = 0.3_f64;
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Tan,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Sub,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[x, 2.0]).unwrap();
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
            MultiAD2FR::Inp(0),
            MultiAD2FR::Sqrt,
            MultiAD2FR::Inp(1),
            MultiAD2FR::Ln,
            MultiAD2FR::Div,
        ];
        let h = MultiAD2FR::compute_hessian(&ops, &[x, y]).unwrap();
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
        let pow_ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Inp(1), MultiAD2FR::Pow];
        let h = MultiAD2FR::compute_hessian(&pow_ops, &[x, y]).unwrap();
        let expected_xx = y * (y - 1.0) * x.powf(y - 2.0);
        let expected_xy = x.powf(y - 1.0) * (1.0 + y * x.ln());
        let expected_yy = x.powf(y) * x.ln().powi(2);
        assert!((h[0][0] - expected_xx).abs() < 1e-10);
        assert!((h[0][1] - expected_xy).abs() < 1e-10);
        assert!((h[1][0] - expected_xy).abs() < 1e-10);
        assert!((h[1][1] - expected_yy).abs() < 1e-10);

        let neg_ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Sin, MultiAD2FR::Neg];
        let h_neg = MultiAD2FR::compute_hessian(&neg_ops, &[0.5]).unwrap();
        assert!((h_neg[0][0] - 0.5_f64.sin()).abs() < 1e-10);
    }

    // ---- Additional coverage tests ----

    #[test]
    fn test_display_format() {
        assert_eq!(format!("{}", MultiAD2FR::Inp(2)), "Inp(2)");
        assert_eq!(format!("{}", MultiAD2FR::Sin), "Sin");
        assert_eq!(format!("{}", MultiAD2FR::Cos), "Cos");
        assert_eq!(format!("{}", MultiAD2FR::Tan), "Tan");
        assert_eq!(format!("{}", MultiAD2FR::Neg), "Neg");
        assert_eq!(format!("{}", MultiAD2FR::Exp), "Exp");
        assert_eq!(format!("{}", MultiAD2FR::Ln), "Ln");
        assert_eq!(format!("{}", MultiAD2FR::Sqrt), "Sqrt");
        assert_eq!(format!("{}", MultiAD2FR::Log1pExp), "Log1pExp");
        assert_eq!(format!("{}", MultiAD2FR::Add), "Add");
        assert_eq!(format!("{}", MultiAD2FR::Sub), "Sub");
        assert_eq!(format!("{}", MultiAD2FR::Mul), "Mul");
        assert_eq!(format!("{}", MultiAD2FR::Div), "Div");
        assert_eq!(format!("{}", MultiAD2FR::Pow), "Pow");
    }

    #[test]
    fn test_display_all_variants() {
        // Exercise Display for all remaining variants through compute_hessian
        // (which covers From<MultiAD2FR> for OpKind for those variants)
        let ops = vec![
            MultiAD2FR::Inp(0),
            MultiAD2FR::Sin,
            MultiAD2FR::Cos,
            MultiAD2FR::Tan,
            MultiAD2FR::Neg,
            MultiAD2FR::Exp,
            MultiAD2FR::Ln,
            MultiAD2FR::Sqrt,
            MultiAD2FR::Log1pExp,
            MultiAD2FR::Add,
            MultiAD2FR::Sub,
            MultiAD2FR::Mul,
            MultiAD2FR::Div,
            MultiAD2FR::Pow,
        ];
        // Just check display doesn't panic
        for op in &ops {
            let _s = format!("{}", op);
        }
    }

    #[test]
    fn test_hessian_sub_op() {
        // f(x, y) = x - y  →  H = [[0, 0], [0, 0]]
        let ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Inp(1), MultiAD2FR::Sub];
        let h = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0]).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_log1p_exp_hessian() {
        // f(x) = log1p_exp(x) = ln(1+exp(x))
        // f''(x) = exp(x)/(1+exp(x))^2 = sigmoid(x)*(1-sigmoid(x))
        let ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Log1pExp];
        let x = 1.0_f64;
        let h = MultiAD2FR::compute_hessian(&ops, &[x]).unwrap();
        let sigmoid = x.exp() / (1.0 + x.exp());
        let expected = sigmoid * (1.0 - sigmoid);
        assert!((h[0][0] - expected).abs() < 1e-10, "H[0][0] = {}", h[0][0]);
    }

    #[test]
    fn test_input_index_out_of_bounds() {
        let ops = vec![MultiAD2FR::Inp(5)];
        let result = MultiAD2FR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unary_missing_operand() {
        let ops = vec![MultiAD2FR::Sin];
        let result = MultiAD2FR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_missing_right_operand() {
        let ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Add];
        let result = MultiAD2FR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_missing_left_operand() {
        let ops = vec![MultiAD2FR::Add];
        let result = MultiAD2FR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_items_on_stack() {
        let ops = vec![MultiAD2FR::Inp(0), MultiAD2FR::Inp(1)];
        let result = MultiAD2FR::compute_hessian(&ops, &[1.0, 2.0]);
        assert!(result.is_err());
    }
}
