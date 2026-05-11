//! Exact second-order autodiff for multivariate functions using Reverse-over-Reverse (RR) mode.
//!
//! This module implements the RR method for computing exact Hessians of
//! multivariate functions f: ℝⁿ → ℝ.
//!
//! # Supported Operations
//!
//! `MultiAD2RR` supports the smooth operation subset shared by the exact Hessian
//! engines: `Inp`, `Sin`, `Cos`, `Tan`, `Neg`, `Exp`, `Ln`, `Sqrt`, `Add`, `Sub`,
//! `Mul`, `Div`, and `Pow`. Non-smooth operations like `Abs` are not supported by
//! the exact Hessian types. For Hessian approximation with the full operation set,
//! use [`MultiAD::compute_hessian`](crate::MultiAD::compute_hessian)
//! (finite-difference based).
//!
//! # Algorithm
//!
//! The RR method applies reverse-mode AD with second-order chain rule:
//!
//! 1. **Forward pass** — evaluate the computation graph, storing at each node:
//!    - The node's scalar value
//!    - The node's **gradient vector** `g = ∂node/∂x` (one entry per input variable)
//!    - Local first and second derivatives of the operation
//!
//! 2. **Reverse pass** — starting from the output, accumulate:
//!    - **Adjoint** (first-order): standard reverse-mode
//!    - **Hessian** (second-order): for each operation z = op(u, v):
//!      ```text
//!      H += a * ddy_uu * g_u (x) g_u^T
//!      H += a * ddy_vv * g_v (x) g_v^T
//!      H += a * ddy_uv * (g_u (x) g_v^T + g_v (x) g_u^T)
//!      ```
//!
//! See [docs/multi_ad_hessian.md](../../docs/multi_ad_hessian.md) for complete theory.

use std::fmt;

use crate::error::{AutodiffError, Result};

use super::multi_ad::MultiAD;
use super::op_rules::{self, LocalRule};

/// Stack-based operation for multivariate second-order AD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiAD2RR {
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

impl fmt::Display for MultiAD2RR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiAD2RR::Inp(idx) => write!(f, "Inp({})", idx),
            MultiAD2RR::Sin => write!(f, "Sin"),
            MultiAD2RR::Cos => write!(f, "Cos"),
            MultiAD2RR::Tan => write!(f, "Tan"),
            MultiAD2RR::Neg => write!(f, "Neg"),
            MultiAD2RR::Exp => write!(f, "Exp"),
            MultiAD2RR::Ln => write!(f, "Ln"),
            MultiAD2RR::Sqrt => write!(f, "Sqrt"),
            MultiAD2RR::Add => write!(f, "Add"),
            MultiAD2RR::Sub => write!(f, "Sub"),
            MultiAD2RR::Mul => write!(f, "Mul"),
            MultiAD2RR::Div => write!(f, "Div"),
            MultiAD2RR::Pow => write!(f, "Pow"),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

/// A node in the computation graph.
struct StackNode {
    value: f64,
    /// ∂(this node)/∂x_j for each input variable x_j
    grad: Vec<f64>,
}

/// Local derivative information stored during the forward pass.
enum LocalDerivs {
    Unary {
        parent: usize,
        dy: f64,  // dz/du
        ddy: f64, // d²z/du²
    },
    Binary {
        left: usize,
        right: usize,
        dy_left: f64,         // dz/du
        dy_right: f64,        // dz/dv
        ddy_left_left: f64,   // d²z/du²
        ddy_right_right: f64, // d²z/dv²
        ddy_left_right: f64,  // d²z/(du dv)
    },
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

#[inline(always)]
fn as_multiad(op: MultiAD2RR) -> Option<MultiAD> {
    match op {
        MultiAD2RR::Inp(_) => None,
        MultiAD2RR::Sin => Some(MultiAD::Sin),
        MultiAD2RR::Cos => Some(MultiAD::Cos),
        MultiAD2RR::Tan => Some(MultiAD::Tan),
        MultiAD2RR::Neg => Some(MultiAD::Neg),
        MultiAD2RR::Exp => Some(MultiAD::Exp),
        MultiAD2RR::Ln => Some(MultiAD::Ln),
        MultiAD2RR::Sqrt => Some(MultiAD::Sqrt),
        MultiAD2RR::Add => Some(MultiAD::Add),
        MultiAD2RR::Sub => Some(MultiAD::Sub),
        MultiAD2RR::Mul => Some(MultiAD::Mul),
        MultiAD2RR::Div => Some(MultiAD::Div),
        MultiAD2RR::Pow => Some(MultiAD::Pow),
    }
}

impl MultiAD2RR {
    /// Compute exact Hessian using Reverse-over-Reverse mode.
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
    /// # Accuracy
    ///
    /// Machine-precision exact (~1e-14 relative error).
    ///
    /// # Complexity
    ///
    /// - Time: O(G · n²) where G = graph size, n = number of inputs
    /// - Space: O(G · n) for per-node gradient vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::MultiAD2RR;
    ///
    /// // f(x, y) = x² + y²  →  Hessian = [[2, 0], [0, 2]]
    /// let ops = vec![
    ///     MultiAD2RR::Inp(0), MultiAD2RR::Inp(0), MultiAD2RR::Mul,
    ///     MultiAD2RR::Inp(1), MultiAD2RR::Inp(1), MultiAD2RR::Mul,
    ///     MultiAD2RR::Add,
    /// ];
    /// let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
    /// assert!((h[0][0] - 2.0).abs() < 1e-12);
    /// assert!((h[1][1] - 2.0).abs() < 1e-12);
    /// ```
    pub fn compute_hessian(ops: &[MultiAD2RR], x: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n_vars = x.len();
        if ops.is_empty() {
            return Ok(vec![vec![0.0; n_vars]; n_vars]);
        }

        // ================================================================
        // FORWARD PASS: value + per-node gradient vector
        // ================================================================
        let mut nodes: Vec<StackNode> = Vec::new();
        let mut local_derivs: Vec<Option<LocalDerivs>> = Vec::new();
        let mut eval_stack: Vec<usize> = Vec::new();

        for &op in ops {
            match op {
                MultiAD2RR::Inp(k) => {
                    if k >= n_vars {
                        return Err(AutodiffError::IndexOutOfBounds {
                            index: k,
                            max_index: n_vars.saturating_sub(1),
                        });
                    }
                    let idx = nodes.len();
                    let mut grad = vec![0.0; n_vars];
                    grad[k] = 1.0;
                    nodes.push(StackNode { value: x[k], grad });
                    local_derivs.push(None);
                    eval_stack.push(idx);
                }
                MultiAD2RR::Sin
                | MultiAD2RR::Cos
                | MultiAD2RR::Tan
                | MultiAD2RR::Neg
                | MultiAD2RR::Exp
                | MultiAD2RR::Ln
                | MultiAD2RR::Sqrt => {
                    let p = eval_stack.pop().ok_or(AutodiffError::InvalidGraph {
                        reason: "unary operation missing operand",
                    })?;
                    let op = as_multiad(op).expect("non-input op");
                    let args = [nodes[p].value];
                    let value = op_rules::forward_value(op, &args)?;
                    let rule = op_rules::local_rule(op, &args, value)?;
                    let idx = nodes.len();
                    let (dy, ddy) = match rule {
                        LocalRule::Unary { dy, ddy } => (dy, ddy),
                        LocalRule::Binary { .. } => unreachable!("unary op must have unary rule"),
                    };
                    let grad: Vec<f64> = nodes[p].grad.iter().map(|&g| dy * g).collect();
                    nodes.push(StackNode { value, grad });
                    local_derivs.push(Some(LocalDerivs::Unary { parent: p, dy, ddy }));
                    eval_stack.push(idx);
                }
                MultiAD2RR::Add
                | MultiAD2RR::Sub
                | MultiAD2RR::Mul
                | MultiAD2RR::Div
                | MultiAD2RR::Pow => {
                    let r = eval_stack.pop().ok_or(AutodiffError::InvalidGraph {
                        reason: "binary operation missing right operand",
                    })?;
                    let l = eval_stack.pop().ok_or(AutodiffError::InvalidGraph {
                        reason: "binary operation missing left operand",
                    })?;
                    let op = as_multiad(op).expect("non-input op");
                    let args = [nodes[l].value, nodes[r].value];
                    let value = op_rules::forward_value(op, &args)?;
                    let rule = op_rules::local_rule(op, &args, value)?;
                    let idx = nodes.len();
                    let (dy_left, dy_right, ddy_left_left, ddy_right_right, ddy_left_right) =
                        match rule {
                            LocalRule::Unary { .. } => {
                                unreachable!("binary op must have binary rule")
                            }
                            LocalRule::Binary {
                                dy_left,
                                dy_right,
                                ddy_left_left,
                                ddy_right_right,
                                ddy_left_right,
                            } => (
                                dy_left,
                                dy_right,
                                ddy_left_left,
                                ddy_right_right,
                                ddy_left_right,
                            ),
                        };
                    let grad: Vec<f64> = nodes[l]
                        .grad
                        .iter()
                        .zip(&nodes[r].grad)
                        .map(|(a, b)| dy_left * a + dy_right * b)
                        .collect();
                    nodes.push(StackNode { value, grad });
                    local_derivs.push(Some(LocalDerivs::Binary {
                        left: l,
                        right: r,
                        dy_left,
                        dy_right,
                        ddy_left_left,
                        ddy_right_right,
                        ddy_left_right,
                    }));
                    eval_stack.push(idx);
                }
            }
        }

        if eval_stack.len() != 1 {
            return Err(AutodiffError::InvalidGraph {
                reason: "RPN expression must leave exactly one output on the stack",
            });
        }

        // ================================================================
        // REVERSE PASS: adjoint + Hessian accumulation
        // ================================================================
        let mut adjoint: Vec<f64> = vec![0.0; nodes.len()];
        let mut hessian: Vec<Vec<f64>> = vec![vec![0.0; n_vars]; n_vars];

        adjoint[nodes.len() - 1] = 1.0;

        for i in (0..nodes.len()).rev() {
            let a = adjoint[i];

            if let Some(ref d) = local_derivs[i] {
                match d {
                    LocalDerivs::Unary { parent, dy, ddy } => {
                        adjoint[*parent] += a * dy;

                        // H += a · ddy · g_parent ⊗ g_parent
                        let g = &nodes[*parent].grad;
                        for vi in 0..n_vars {
                            let gi = g[vi];
                            if gi == 0.0 {
                                continue;
                            }
                            // Precompute factor outside the vj loop
                            let factor = a * ddy * gi;
                            for vj in vi..n_vars {
                                let contrib = factor * g[vj];
                                hessian[vi][vj] += contrib;
                                if vi != vj {
                                    hessian[vj][vi] += contrib;
                                }
                            }
                        }
                    }
                    LocalDerivs::Binary {
                        left,
                        right,
                        dy_left,
                        dy_right,
                        ddy_left_left,
                        ddy_right_right,
                        ddy_left_right,
                    } => {
                        adjoint[*left] += a * dy_left;
                        adjoint[*right] += a * dy_right;

                        let gl = &nodes[*left].grad;
                        let gr = &nodes[*right].grad;

                        // H += a · ddy_ll · g_left ⊗ g_left
                        if *ddy_left_left != 0.0 {
                            for vi in 0..n_vars {
                                let gi = gl[vi];
                                if gi == 0.0 {
                                    continue;
                                }
                                for vj in 0..n_vars {
                                    hessian[vi][vj] += a * ddy_left_left * gi * gl[vj];
                                }
                            }
                        }

                        // H += a · ddy_rr · g_right ⊗ g_right
                        if *ddy_right_right != 0.0 {
                            for vi in 0..n_vars {
                                let gi = gr[vi];
                                if gi == 0.0 {
                                    continue;
                                }
                                for vj in 0..n_vars {
                                    hessian[vi][vj] += a * ddy_right_right * gi * gr[vj];
                                }
                            }
                        }

                        // H += a · ddy_lr · (g_left ⊗ g_right + g_right ⊗ g_left)
                        if *ddy_left_right != 0.0 {
                            for vi in 0..n_vars {
                                for vj in 0..n_vars {
                                    hessian[vi][vj] +=
                                        a * ddy_left_right * (gl[vi] * gr[vj] + gr[vi] * gl[vj]);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(hessian)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    /// Absolute tolerance for exact (non-finite-difference) methods.
    const TOL: f64 = 1e-12;

    #[test]
    fn test_product_xy() {
        // f = x·y  →  H = [[0,1],[1,0]]
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Inp(1), MultiAD2RR::Mul];
        let h = MultiAD2RR::compute_hessian(&ops, &[2.0, 3.0]).unwrap();
        assert_eq!(h[0][0], 0.0);
        assert_eq!(h[1][1], 0.0);
        assert_eq!(h[0][1], 1.0);
        assert_eq!(h[1][0], 1.0);
    }

    #[test]
    fn test_simple_quadratic() {
        // f = x² + y²  →  H = [[2,0],[0,2]]
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Inp(0),
            MultiAD2RR::Mul,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Inp(1),
            MultiAD2RR::Mul,
            MultiAD2RR::Add,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < TOL);
        assert!((h[1][1] - 2.0).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
    }

    #[test]
    fn test_empty_ops() {
        let h = MultiAD2RR::compute_hessian(&[], &[1.0, 2.0]).unwrap();
        assert_eq!(h.len(), 2);
        assert!(h.iter().all(|row| row.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_sin() {
        // f = sin(x) + y  →  H[0][0] = -sin(x), rest 0
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Sin,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Add,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.sin())).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_cos() {
        // f = cos(x) + y  →  H[0][0] = -cos(x)
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Cos,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Add,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.cos())).abs() < TOL);
    }

    #[test]
    fn test_exp() {
        // f = exp(x) + y  →  H[0][0] = exp(x)
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Exp,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Add,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - 1.0_f64.exp()).abs() < TOL);
    }

    #[test]
    fn test_three_variables_quadratic() {
        // f = x²+y²+z²  →  H = diag(2,2,2)
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Inp(0),
            MultiAD2RR::Mul,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Inp(1),
            MultiAD2RR::Mul,
            MultiAD2RR::Add,
            MultiAD2RR::Inp(2),
            MultiAD2RR::Inp(2),
            MultiAD2RR::Mul,
            MultiAD2RR::Add,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0, 3.0]).unwrap();
        for (i, row) in h.iter().enumerate().take(3) {
            assert!((row[i] - 2.0).abs() < TOL, "H[{}][{}] = {}", i, i, row[i]);
        }
        assert!((h[0][1]).abs() < TOL);
        assert!((h[0][2]).abs() < TOL);
        assert!((h[1][2]).abs() < TOL);
    }

    #[test]
    fn test_sin_plus_exp() {
        // f = sin(x) + exp(y)  →  H = [[-sin(x), 0], [0, exp(y)]]
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Sin,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Exp,
            MultiAD2RR::Add,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0] - (-1.0_f64.sin())).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1] - 2.0_f64.exp()).abs() < TOL);
    }

    // ---- New correctness tests (non-trivial compositions) ----

    #[test]
    fn test_sin_times_cos() {
        // f = sin(x)·cos(y)
        // H[0][0] = -sin(x)·cos(y)
        // H[1][1] = -sin(x)·cos(y)
        // H[0][1] = H[1][0] = -cos(x)·sin(y)
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Sin,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Cos,
            MultiAD2RR::Mul,
        ];
        let x = 1.0;
        let y = 2.0;
        let h = MultiAD2RR::compute_hessian(&ops, &[x, y]).unwrap();

        let ex00 = -x.sin() * y.cos();
        let ex11 = ex00;
        let ex01 = -x.cos() * y.sin();

        assert!(
            (h[0][0] - ex00).abs() < TOL,
            "H[0][0] = {} (expected {})",
            h[0][0],
            ex00
        );
        assert!(
            (h[1][1] - ex11).abs() < TOL,
            "H[1][1] = {} (expected {})",
            h[1][1],
            ex11
        );
        assert!(
            (h[0][1] - ex01).abs() < TOL,
            "H[0][1] = {} (expected {})",
            h[0][1],
            ex01
        );
        assert!(
            (h[1][0] - ex01).abs() < TOL,
            "H[1][0] = {} (expected {})",
            h[1][0],
            ex01
        );
    }

    #[test]
    fn test_exp_times_exp() {
        // f = exp(x)·exp(y) = exp(x+y)
        // H[i][j] = exp(x+y) for all i,j
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Exp,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Exp,
            MultiAD2RR::Mul,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();
        let expected = (1.0_f64.exp()) * (2.0_f64.exp());

        assert!(
            (h[0][0] - expected).abs() < TOL,
            "H[0][0] = {} (expected {})",
            h[0][0],
            expected
        );
        assert!(
            (h[0][1] - expected).abs() < TOL,
            "H[0][1] = {} (expected {})",
            h[0][1],
            expected
        );
        assert!(
            (h[1][1] - expected).abs() < TOL,
            "H[1][1] = {} (expected {})",
            h[1][1],
            expected
        );
    }

    #[test]
    fn test_exp_times_sin_single_var() {
        // f(x) = exp(x)·sin(x)  (single variable)
        // f''(x) = 2·exp(x)·cos(x)
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Exp,
            MultiAD2RR::Inp(0),
            MultiAD2RR::Sin,
            MultiAD2RR::Mul,
        ];
        let x = 1.0;
        let h = MultiAD2RR::compute_hessian(&ops, &[x]).unwrap();
        let expected = 2.0 * x.exp() * x.cos();

        assert!(
            (h[0][0] - expected).abs() < TOL,
            "H[0][0] = {} (expected {})",
            h[0][0],
            expected
        );
    }

    #[test]
    fn test_sum_squared() {
        // f = (x+y)·(x+y) = x²+2xy+y²  →  H = [[2,2],[2,2]]
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Inp(1),
            MultiAD2RR::Add,
            MultiAD2RR::Inp(0),
            MultiAD2RR::Inp(1),
            MultiAD2RR::Add,
            MultiAD2RR::Mul,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]).unwrap();

        assert!(
            (h[0][0] - 2.0).abs() < TOL,
            "H[0][0] = {} (expected 2.0)",
            h[0][0]
        );
        assert!(
            (h[0][1] - 2.0).abs() < TOL,
            "H[0][1] = {} (expected 2.0)",
            h[0][1]
        );
        assert!(
            (h[1][1] - 2.0).abs() < TOL,
            "H[1][1] = {} (expected 2.0)",
            h[1][1]
        );
    }

    #[test]
    fn test_tan_and_sub() {
        let x = 0.3_f64;
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Tan,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Sub,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[x, 2.0]).unwrap();
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
            MultiAD2RR::Inp(0),
            MultiAD2RR::Sqrt,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Ln,
            MultiAD2RR::Div,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[x, y]).unwrap();
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
        let pow_ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Inp(1), MultiAD2RR::Pow];
        let h = MultiAD2RR::compute_hessian(&pow_ops, &[x, y]).unwrap();
        let expected_xx = y * (y - 1.0) * x.powf(y - 2.0);
        let expected_xy = x.powf(y - 1.0) * (1.0 + y * x.ln());
        let expected_yy = x.powf(y) * x.ln().powi(2);
        assert!((h[0][0] - expected_xx).abs() < 1e-10);
        assert!((h[0][1] - expected_xy).abs() < 1e-10);
        assert!((h[1][0] - expected_xy).abs() < 1e-10);
        assert!((h[1][1] - expected_yy).abs() < 1e-10);

        let neg_ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Sin, MultiAD2RR::Neg];
        let h_neg = MultiAD2RR::compute_hessian(&neg_ops, &[0.5]).unwrap();
        assert!((h_neg[0][0] - 0.5_f64.sin()).abs() < 1e-10);
    }

    // ---- Error path tests ----

    #[test]
    fn test_empty_ops_returns_zero_hessian() {
        let h = MultiAD2RR::compute_hessian(&[], &[]).unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn test_single_var_hessian() {
        // f(x) = x^2 => f''(x) = 2
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Inp(0), MultiAD2RR::Mul];
        let h = MultiAD2RR::compute_hessian(&ops, &[3.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < TOL, "H[0][0] = {}", h[0][0]);
    }

    #[test]
    fn test_input_index_out_of_bounds() {
        let ops = vec![MultiAD2RR::Inp(5)];
        let result = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unary_missing_operand() {
        let ops = vec![MultiAD2RR::Sin];
        let result = MultiAD2RR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_missing_operand() {
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Add];
        let result = MultiAD2RR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_items_on_stack() {
        // Two inputs pushed, no reduction => stack has 2 items
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Inp(1)];
        let result = MultiAD2RR::compute_hessian(&ops, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_format() {
        assert_eq!(format!("{}", MultiAD2RR::Inp(3)), "Inp(3)");
        assert_eq!(format!("{}", MultiAD2RR::Sin), "Sin");
        assert_eq!(format!("{}", MultiAD2RR::Cos), "Cos");
        assert_eq!(format!("{}", MultiAD2RR::Tan), "Tan");
        assert_eq!(format!("{}", MultiAD2RR::Neg), "Neg");
        assert_eq!(format!("{}", MultiAD2RR::Exp), "Exp");
        assert_eq!(format!("{}", MultiAD2RR::Ln), "Ln");
        assert_eq!(format!("{}", MultiAD2RR::Sqrt), "Sqrt");
        assert_eq!(format!("{}", MultiAD2RR::Add), "Add");
        assert_eq!(format!("{}", MultiAD2RR::Sub), "Sub");
        assert_eq!(format!("{}", MultiAD2RR::Mul), "Mul");
        assert_eq!(format!("{}", MultiAD2RR::Div), "Div");
        assert_eq!(format!("{}", MultiAD2RR::Pow), "Pow");
    }

    // ---- Additional operation tests for uncovered branches ----

    #[test]
    fn test_hessian_single_ln() {
        // f(x) = ln(x), f''(x) = -1/x²
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Ln];
        let x = 2.0_f64;
        let h = MultiAD2RR::compute_hessian(&ops, &[x]).unwrap();
        let expected = -1.0 / (x * x);
        assert!((h[0][0] - expected).abs() < TOL);
    }

    #[test]
    fn test_hessian_single_sqrt() {
        // f(x) = sqrt(x), f''(x) = -1/(4 * x * sqrt(x))
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Sqrt];
        let x = 4.0_f64;
        let h = MultiAD2RR::compute_hessian(&ops, &[x]).unwrap();
        let expected = -1.0 / (4.0 * x * x.sqrt());
        assert!((h[0][0] - expected).abs() < TOL);
    }

    #[test]
    fn test_hessian_single_neg() {
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Neg];
        let h = MultiAD2RR::compute_hessian(&ops, &[3.0]).unwrap();
        assert!((h[0][0]).abs() < TOL);
    }

    #[test]
    fn test_hessian_single_tan() {
        let x = 0.4_f64;
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Tan];
        let h = MultiAD2RR::compute_hessian(&ops, &[x]).unwrap();
        let sec_sq = 1.0 / x.cos().powi(2);
        let expected = 2.0 * sec_sq * x.tan();
        assert!((h[0][0] - expected).abs() < TOL);
    }

    #[test]
    fn test_hessian_single_cos() {
        let x = 1.0_f64;
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Cos];
        let h = MultiAD2RR::compute_hessian(&ops, &[x]).unwrap();
        assert!((h[0][0] - (-x.cos())).abs() < TOL);
    }

    #[test]
    fn test_hessian_composition_neg_sin() {
        // f(x) = -sin(x), f''(x) = sin(x)
        let ops = vec![MultiAD2RR::Inp(0), MultiAD2RR::Sin, MultiAD2RR::Neg];
        let x = 1.0_f64;
        let h = MultiAD2RR::compute_hessian(&ops, &[x]).unwrap();
        assert!((h[0][0] - x.sin()).abs() < TOL);
    }

    #[test]
    fn test_hessian_sub_and_ln() {
        // f(x, y) = x - ln(y) → H = [[0, 0], [0, 1/y²]]
        let y = 3.0_f64;
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Inp(1),
            MultiAD2RR::Ln,
            MultiAD2RR::Sub,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[2.0, y]).unwrap();
        assert!((h[0][0]).abs() < TOL);
        assert!((h[0][1]).abs() < TOL);
        assert!((h[1][0]).abs() < TOL);
        assert!((h[1][1] - 1.0 / (y * y)).abs() < TOL);
    }

    #[test]
    fn test_hessian_mul_and_cos() {
        // f(x, y) = cos(x) * y → H = [[-cos(x)*y, -sin(x)], [-sin(x), 0]]
        let x = 1.0_f64;
        let y = 2.0_f64;
        let ops = vec![
            MultiAD2RR::Inp(0),
            MultiAD2RR::Cos,
            MultiAD2RR::Inp(1),
            MultiAD2RR::Mul,
        ];
        let h = MultiAD2RR::compute_hessian(&ops, &[x, y]).unwrap();
        assert!((h[0][0] - (-x.cos() * y)).abs() < TOL);
        assert!((h[0][1] - (-x.sin())).abs() < TOL);
        assert!((h[1][0] - (-x.sin())).abs() < TOL);
        assert!((h[1][1]).abs() < TOL);
    }

    #[test]
    fn test_binary_missing_left_operand() {
        let ops = vec![MultiAD2RR::Mul];
        let result = MultiAD2RR::compute_hessian(&ops, &[1.0]);
        assert!(result.is_err());
    }
}
