//! Shared types and the dual-number Hessian algorithm for multivariate AD.
//!
//! This module provides:
//! - [`OpKind`]: an operation descriptor used by FR/RF to avoid enum duplication
//! - [`Dual`]: dual numbers for forward-mode AD
//! - [`compute_hessian_dual`]: the shared FR/RF Hessian algorithm
//!
//! For scalar functions f: ℝⁿ → ℝ, Forward-over-Reverse (FR) and Reverse-over-Forward (RF)
//! are equivalent in implementation. Both work by:
//! 1. Forward pass with dual numbers (value + tangent in seed direction e_j)
//! 2. Reverse pass with **dual adjoints** (val = ∂f/∂node, tan = ∂²f/∂node∂x_j)
//!
//! The dual adjoint at each input node's tangent gives a column of the Hessian.

use crate::error::{AutodiffError, Result};

use crate::multi::first_order::MultiAD;
use crate::multi::op_rules::{self, LocalRule};

/// An operation descriptor for the stack-based RPN computation graph.
/// Used by FR and RF Hessian methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Input variable by index
    Inp(usize),
    /// Unary sine
    Sin,
    /// Unary cosine
    Cos,
    /// Unary tangent
    Tan,
    /// Unary negation
    Neg,
    /// Unary exponential
    Exp,
    /// Unary natural logarithm
    Ln,
    /// Unary square root
    Sqrt,
    /// Unary stable softplus: ln(1 + exp(x))
    Log1pExp,
    /// Binary addition
    Add,
    /// Binary subtraction
    Sub,
    /// Binary multiplication
    Mul,
    /// Binary division
    Div,
    /// Binary power
    Pow,
}

// ---------------------------------------------------------------------------
// Dual number
// ---------------------------------------------------------------------------

/// A dual number `(val, tan)` for forward-mode automatic differentiation.
///
/// `val` is the function value; `tan` is the directional tangent
/// (∂/∂x_j in the current seed direction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Dual {
    pub val: f64,
    pub tan: f64,
}

impl Dual {
    /// Create a seed variable: `(x, 1)` — tangent = 1 in the seed direction.
    #[inline(always)]
    pub fn variable(val: f64) -> Self {
        Dual { val, tan: 1.0 }
    }

    /// Create a constant: `(x, 0)` — tangent = 0 (independent of seed direction).
    #[inline(always)]
    pub fn constant(val: f64) -> Self {
        Dual { val, tan: 0.0 }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    #[inline(always)]
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val + rhs.val,
            tan: self.tan + rhs.tan,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    #[inline(always)]
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val - rhs.val,
            tan: self.tan - rhs.tan,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    #[inline(always)]
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val * rhs.val,
            tan: self.val * rhs.tan + self.tan * rhs.val,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;
    #[inline(always)]
    fn div(self, rhs: Dual) -> Dual {
        let denom = rhs.val * rhs.val;
        Dual {
            val: self.val / rhs.val,
            tan: (self.tan * rhs.val - self.val * rhs.tan) / denom,
        }
    }
}

// ---------------------------------------------------------------------------
// Dual adjoint accumulation
// ---------------------------------------------------------------------------

/// Accumulate a dual adjoint: `target += source * local_deriv`.
///
/// Using the product rule on dual numbers:
/// `(a.val + a.tan·ε) × (b.val + b.tan·ε) = a.val·b.val + (a.val·b.tan + a.tan·b.val)·ε`
#[inline(always)]
pub(crate) fn dual_adj_accum(target: &mut Dual, source: Dual, local_deriv: Dual) {
    target.val += source.val * local_deriv.val;
    target.tan += source.val * local_deriv.tan + source.tan * local_deriv.val;
}

// ---------------------------------------------------------------------------
// Node metadata (internal to the forward pass)
// ---------------------------------------------------------------------------

/// Tracks how each node in the computation graph was created and the
/// associated local derivatives needed for the reverse pass.
#[derive(Clone, Copy)]
enum NodeKind {
    Input {
        var_idx: usize,
    },
    Unary {
        parent: usize,
        op: MultiAD,
    },
    Binary {
        left: usize,
        right: usize,
        op: MultiAD,
    },
}

// ---------------------------------------------------------------------------
// Core algorithm: dual forward + dual reverse
// ---------------------------------------------------------------------------

#[inline(always)]
fn opkind_to_multiad(op: OpKind) -> Option<MultiAD> {
    match op {
        OpKind::Inp(_) => None,
        OpKind::Sin => Some(MultiAD::Sin),
        OpKind::Cos => Some(MultiAD::Cos),
        OpKind::Tan => Some(MultiAD::Tan),
        OpKind::Neg => Some(MultiAD::Neg),
        OpKind::Exp => Some(MultiAD::Exp),
        OpKind::Ln => Some(MultiAD::Ln),
        OpKind::Sqrt => Some(MultiAD::Sqrt),
        OpKind::Log1pExp => Some(MultiAD::Log1pExp),
        OpKind::Add => Some(MultiAD::Add),
        OpKind::Sub => Some(MultiAD::Sub),
        OpKind::Mul => Some(MultiAD::Mul),
        OpKind::Div => Some(MultiAD::Div),
        OpKind::Pow => Some(MultiAD::Pow),
    }
}

#[inline(always)]
fn apply_dual(op: MultiAD, args: &[Dual]) -> Result<Dual> {
    let arg_values: Vec<f64> = args.iter().map(|arg| arg.val).collect();
    let arg_tangents: Vec<f64> = args.iter().map(|arg| arg.tan).collect();
    let value = op_rules::forward_value(op, &arg_values)?;
    let tangent = op_rules::directional_tangent(op, &arg_values, &arg_tangents, value)?;
    Ok(Dual {
        val: value,
        tan: tangent,
    })
}

/// Validate input indices and RPN stack shape before seed-direction evaluation.
fn validate_rpn_structure(ops: &[OpKind], n_vars: usize) -> Result<()> {
    let mut stack_size: usize = 0;

    for &op in ops {
        match op {
            OpKind::Inp(k) => {
                if k >= n_vars {
                    return Err(AutodiffError::IndexOutOfBounds {
                        index: k,
                        max_index: n_vars.saturating_sub(1),
                    });
                }
                stack_size += 1;
            }
            OpKind::Sin
            | OpKind::Cos
            | OpKind::Tan
            | OpKind::Neg
            | OpKind::Exp
            | OpKind::Ln
            | OpKind::Sqrt
            | OpKind::Log1pExp => {
                if stack_size == 0 {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "unary operation missing operand",
                    });
                }
            }
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow => {
                if stack_size < 2 {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "binary operation missing operand",
                    });
                }
                stack_size -= 1;
            }
        }
    }

    if stack_size != 1 {
        return Err(AutodiffError::InvalidGraph {
            reason: "RPN expression must leave exactly one output on the stack",
        });
    }

    Ok(())
}

/// Compute the exact Hessian using dual-number forward and reverse passes.
///
/// For each seed direction `e_j` (j = 0 … n−1):
///
/// 1. **Forward pass** — evaluate the graph with dual numbers whose tangent
///    component is seeded in direction `e_j`. Every intermediate value becomes
///    a [`Dual`].
///
/// 2. **Reverse pass** — propagate **dual adjoints** backward. Each adjoint
///    is a `Dual { val, tan }` where:
///    - `val = ∂f/∂node` (standard first-order adjoint)
///    - `tan = ∂²f/(∂node · ∂x_j)` (second-order cross-derivative)
///
///    At input nodes the tangent of the adjoint equals `H[j][k]`.
///
/// # Accuracy
///
/// Machine-precision exact (up to floating-point rounding, ~1e-14 relative error).
///
/// # Complexity
///
/// O(n · G) where n = number of inputs and G = graph size (number of ops).
/// Each of the n seed directions requires one forward pass and one reverse pass.
pub(crate) fn compute_hessian_dual(ops: &[OpKind], x: &[f64]) -> Result<Vec<Vec<f64>>> {
    let n_vars = x.len();
    if ops.is_empty() {
        return Ok(vec![vec![0.0; n_vars]; n_vars]);
    }

    validate_rpn_structure(ops, n_vars)?;

    let mut hessian = vec![vec![0.0; n_vars]; n_vars];

    for (j, hessian_row) in hessian.iter_mut().enumerate().take(n_vars) {
        // ================================================================
        // FORWARD PASS: dual values in direction e_j
        // ================================================================
        let mut dual_values: Vec<Dual> = Vec::new();
        let mut node_kinds: Vec<NodeKind> = Vec::new();
        let mut eval_stack: Vec<usize> = Vec::new();

        for &op in ops {
            match op {
                OpKind::Inp(k) => {
                    if k >= n_vars {
                        return Err(AutodiffError::IndexOutOfBounds {
                            index: k,
                            max_index: n_vars.saturating_sub(1),
                        });
                    }
                    let idx = dual_values.len();
                    dual_values.push(if k == j {
                        Dual::variable(x[k])
                    } else {
                        Dual::constant(x[k])
                    });
                    node_kinds.push(NodeKind::Input { var_idx: k });
                    eval_stack.push(idx);
                }
                OpKind::Sin
                | OpKind::Cos
                | OpKind::Tan
                | OpKind::Neg
                | OpKind::Exp
                | OpKind::Ln
                | OpKind::Sqrt
                | OpKind::Log1pExp => {
                    let p = eval_stack.pop().ok_or(AutodiffError::InvalidGraph {
                        reason: "unary operation missing operand",
                    })?;
                    let op = opkind_to_multiad(op).expect("non-input op");
                    let idx = dual_values.len();
                    dual_values.push(apply_dual(op, &[dual_values[p]])?);
                    node_kinds.push(NodeKind::Unary { parent: p, op });
                    eval_stack.push(idx);
                }
                OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow => {
                    let r = eval_stack.pop().ok_or(AutodiffError::InvalidGraph {
                        reason: "binary operation missing right operand",
                    })?;
                    let l = eval_stack.pop().ok_or(AutodiffError::InvalidGraph {
                        reason: "binary operation missing left operand",
                    })?;
                    let op = opkind_to_multiad(op).expect("non-input op");
                    let idx = dual_values.len();
                    dual_values.push(apply_dual(op, &[dual_values[l], dual_values[r]])?);
                    node_kinds.push(NodeKind::Binary {
                        left: l,
                        right: r,
                        op,
                    });
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
        // REVERSE PASS: dual adjoints
        // ================================================================
        let mut adj: Vec<Dual> = vec![Dual::constant(0.0); dual_values.len()];
        // Seed: ∂f/∂f = 1, ∂²f/(∂f·∂x_j) = 0
        adj[dual_values.len() - 1] = Dual::constant(1.0);

        for i in (0..dual_values.len()).rev() {
            let a = adj[i];
            let nk = node_kinds[i]; // Copy (NodeKind is Copy)
            match nk {
                NodeKind::Input { var_idx: k } => {
                    // Extract Hessian entry: ∂²f/∂x_j∂x_k
                    hessian_row[k] += a.tan;
                }
                NodeKind::Unary { parent, op } => {
                    let u = dual_values[parent];
                    let current = dual_values[i];
                    let rule = op_rules::local_rule(op, &[u.val], current.val)?;
                    let dual_deriv = match rule {
                        LocalRule::Unary { dy, ddy } => Dual {
                            val: dy,
                            tan: ddy * u.tan,
                        },
                        LocalRule::Binary { .. } => unreachable!("unary node must have unary rule"),
                    };
                    dual_adj_accum(&mut adj[parent], a, dual_deriv);
                }
                NodeKind::Binary { left, right, op } => {
                    let u = dual_values[left];
                    let v = dual_values[right];
                    let current = dual_values[i];
                    let rule = op_rules::local_rule(op, &[u.val, v.val], current.val)?;
                    match rule {
                        LocalRule::Unary { .. } => {
                            unreachable!("binary node must have binary rule")
                        }
                        LocalRule::Binary {
                            dy_left,
                            dy_right,
                            ddy_left_left,
                            ddy_right_right,
                            ddy_left_right,
                        } => {
                            let dz_du = Dual {
                                val: dy_left,
                                tan: ddy_left_left * u.tan + ddy_left_right * v.tan,
                            };
                            let dz_dv = Dual {
                                val: dy_right,
                                tan: ddy_left_right * u.tan + ddy_right_right * v.tan,
                            };
                            dual_adj_accum(&mut adj[left], a, dz_du);
                            dual_adj_accum(&mut adj[right], a, dz_dv);
                        }
                    }
                }
            }
        }
    }

    Ok(hessian)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Dual number arithmetic tests ----

    #[test]
    fn test_dual_variable() {
        let d = Dual::variable(3.0);
        assert_eq!(d.val, 3.0);
        assert_eq!(d.tan, 1.0);
    }

    #[test]
    fn test_dual_constant() {
        let d = Dual::constant(3.0);
        assert_eq!(d.val, 3.0);
        assert_eq!(d.tan, 0.0);
    }

    #[test]
    fn test_dual_add() {
        let a = Dual { val: 1.0, tan: 2.0 };
        let b = Dual { val: 3.0, tan: 4.0 };
        let c = a + b;
        assert_eq!(c.val, 4.0);
        assert_eq!(c.tan, 6.0);
    }

    #[test]
    fn test_dual_sub() {
        let a = Dual { val: 5.0, tan: 2.0 };
        let b = Dual { val: 3.0, tan: 1.0 };
        let c = a - b;
        assert_eq!(c.val, 2.0);
        assert_eq!(c.tan, 1.0);
    }

    #[test]
    fn test_dual_mul() {
        let a = Dual { val: 2.0, tan: 3.0 };
        let b = Dual { val: 4.0, tan: 5.0 };
        let c = a * b;
        assert_eq!(c.val, 8.0);
        assert_eq!(c.tan, 2.0 * 5.0 + 3.0 * 4.0); // a.val*b.tan + a.tan*b.val
    }

    #[test]
    fn test_dual_div() {
        let a = Dual { val: 6.0, tan: 2.0 };
        let b = Dual { val: 3.0, tan: 1.0 };
        let c = a / b;
        assert_eq!(c.val, 2.0);
        // tan = (a.tan*b.val - a.val*b.tan) / b.val^2
        let expected_tan = (2.0 * 3.0 - 6.0 * 1.0) / 9.0;
        assert!((c.tan - expected_tan).abs() < 1e-12);
    }

    #[test]
    fn test_dual_adj_accum() {
        let mut target = Dual { val: 1.0, tan: 2.0 };
        let source = Dual { val: 3.0, tan: 4.0 };
        let deriv = Dual { val: 5.0, tan: 6.0 };
        dual_adj_accum(&mut target, source, deriv);
        // val += source.val * deriv.val = 1 + 3*5 = 16
        assert_eq!(target.val, 16.0);
        // tan += source.val * deriv.tan + source.tan * deriv.val = 2 + 3*6 + 4*5 = 2+18+20 = 40
        assert_eq!(target.tan, 40.0);
    }

    // ---- OpKind to MultiAD conversion ----

    #[test]
    fn test_opkind_to_multiad() {
        assert!(opkind_to_multiad(OpKind::Inp(0)).is_none());
        assert_eq!(opkind_to_multiad(OpKind::Sin), Some(MultiAD::Sin));
        assert_eq!(opkind_to_multiad(OpKind::Cos), Some(MultiAD::Cos));
        assert_eq!(opkind_to_multiad(OpKind::Tan), Some(MultiAD::Tan));
        assert_eq!(opkind_to_multiad(OpKind::Neg), Some(MultiAD::Neg));
        assert_eq!(opkind_to_multiad(OpKind::Exp), Some(MultiAD::Exp));
        assert_eq!(opkind_to_multiad(OpKind::Ln), Some(MultiAD::Ln));
        assert_eq!(opkind_to_multiad(OpKind::Sqrt), Some(MultiAD::Sqrt));
        assert_eq!(opkind_to_multiad(OpKind::Log1pExp), Some(MultiAD::Log1pExp));
        assert_eq!(opkind_to_multiad(OpKind::Add), Some(MultiAD::Add));
        assert_eq!(opkind_to_multiad(OpKind::Sub), Some(MultiAD::Sub));
        assert_eq!(opkind_to_multiad(OpKind::Mul), Some(MultiAD::Mul));
        assert_eq!(opkind_to_multiad(OpKind::Div), Some(MultiAD::Div));
        assert_eq!(opkind_to_multiad(OpKind::Pow), Some(MultiAD::Pow));
    }

    // ---- apply_dual tests ----

    #[test]
    fn test_apply_dual_sin() {
        let result = apply_dual(MultiAD::Sin, &[Dual::variable(1.0)]).unwrap();
        assert!((result.val - 1.0_f64.sin()).abs() < 1e-12);
        assert!((result.tan - 1.0_f64.cos()).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_add() {
        let a = Dual::variable(1.0);
        let b = Dual::constant(2.0);
        let result = apply_dual(MultiAD::Add, &[a, b]).unwrap();
        assert_eq!(result.val, 3.0);
        assert_eq!(result.tan, 1.0); // Only a contributes to tangent
    }

    // ---- validate_rpn_structure tests ----

    #[test]
    fn test_validate_rpn_valid() {
        let ops = vec![OpKind::Inp(0), OpKind::Inp(1), OpKind::Mul];
        assert!(validate_rpn_structure(&ops, 2).is_ok());
    }

    #[test]
    fn test_validate_rpn_input_out_of_bounds() {
        let ops = vec![OpKind::Inp(5)];
        let result = validate_rpn_structure(&ops, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rpn_unary_missing_operand() {
        let ops = vec![OpKind::Sin];
        let result = validate_rpn_structure(&ops, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rpn_binary_missing_operand() {
        let ops = vec![OpKind::Inp(0), OpKind::Add];
        let result = validate_rpn_structure(&ops, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rpn_extra_stack_items() {
        let ops = vec![OpKind::Inp(0), OpKind::Inp(1)];
        let result = validate_rpn_structure(&ops, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rpn_empty_stack_at_end() {
        let ops: Vec<OpKind> = vec![];
        let result = validate_rpn_structure(&ops, 0);
        assert!(result.is_err());
    }

    // ---- compute_hessian_dual edge cases ----

    #[test]
    fn test_compute_hessian_dual_empty_ops() {
        let h = compute_hessian_dual(&[], &[1.0, 2.0]).unwrap();
        assert_eq!(h.len(), 2);
        assert!(h.iter().all(|row| row.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_compute_hessian_dual_empty_ops_no_vars() {
        let h = compute_hessian_dual(&[], &[]).unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn test_compute_hessian_dual_log1p_exp() {
        // f(x) = log1p_exp(x)
        let ops = vec![OpKind::Inp(0), OpKind::Log1pExp];
        let x = 1.0_f64;
        let h = compute_hessian_dual(&ops, &[x]).unwrap();
        let sigmoid = x.exp() / (1.0 + x.exp());
        let expected = sigmoid * (1.0 - sigmoid);
        assert!((h[0][0] - expected).abs() < 1e-10, "H[0][0] = {}", h[0][0]);
    }

    #[test]
    fn test_compute_hessian_dual_input_out_of_bounds() {
        let ops = vec![OpKind::Inp(10), OpKind::Sin];
        let result = compute_hessian_dual(&ops, &[1.0]);
        assert!(result.is_err());
    }

    // ---- apply_dual with more operations ----

    #[test]
    fn test_apply_dual_cos() {
        let x = 1.0_f64;
        let result = apply_dual(MultiAD::Cos, &[Dual::variable(x)]).unwrap();
        assert!((result.val - x.cos()).abs() < 1e-12);
        assert!((result.tan - (-x.sin())).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_exp() {
        let x = 1.0_f64;
        let result = apply_dual(MultiAD::Exp, &[Dual::variable(x)]).unwrap();
        assert!((result.val - x.exp()).abs() < 1e-12);
        assert!((result.tan - x.exp()).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_ln() {
        let x = 2.0_f64;
        let result = apply_dual(MultiAD::Ln, &[Dual::variable(x)]).unwrap();
        assert!((result.val - x.ln()).abs() < 1e-12);
        assert!((result.tan - 1.0 / x).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_neg() {
        let result = apply_dual(MultiAD::Neg, &[Dual::variable(5.0)]).unwrap();
        assert_eq!(result.val, -5.0);
        assert_eq!(result.tan, -1.0);
    }

    #[test]
    fn test_apply_dual_sqrt() {
        let x = 4.0_f64;
        let result = apply_dual(MultiAD::Sqrt, &[Dual::variable(x)]).unwrap();
        assert!((result.val - 2.0).abs() < 1e-12);
        assert!((result.tan - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_tan() {
        let x = 0.5_f64;
        let result = apply_dual(MultiAD::Tan, &[Dual::variable(x)]).unwrap();
        assert!((result.val - x.tan()).abs() < 1e-12);
        let expected_tan = 1.0 / x.cos().powi(2);
        assert!((result.tan - expected_tan).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_log1pexp() {
        let x = 1.0_f64;
        let result = apply_dual(MultiAD::Log1pExp, &[Dual::variable(x)]).unwrap();
        assert!((result.val - (1.0 + x.exp()).ln()).abs() < 1e-12);
        // derivative = sigmoid(x) = exp(x)/(1+exp(x))
        let sigmoid = x.exp() / (1.0 + x.exp());
        assert!((result.tan - sigmoid).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_sub() {
        let a = Dual::variable(5.0);
        let b = Dual::constant(3.0);
        let result = apply_dual(MultiAD::Sub, &[a, b]).unwrap();
        assert_eq!(result.val, 2.0);
        assert_eq!(result.tan, 1.0);
    }

    #[test]
    fn test_apply_dual_mul_two_vars() {
        let a = Dual::variable(3.0);
        let b = Dual::variable(4.0);
        let result = apply_dual(MultiAD::Mul, &[a, b]).unwrap();
        assert_eq!(result.val, 12.0);
        // d/dx[x*y] with both variable = y + x = 4 + 3 = 7
        assert!((result.tan - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_div() {
        let a = Dual::variable(6.0);
        let b = Dual::constant(3.0);
        let result = apply_dual(MultiAD::Div, &[a, b]).unwrap();
        assert_eq!(result.val, 2.0);
        assert!((result.tan - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dual_pow() {
        let a = Dual::variable(2.0);
        let b = Dual::constant(3.0);
        let result = apply_dual(MultiAD::Pow, &[a, b]).unwrap();
        assert!((result.val - 8.0).abs() < 1e-12);
        // d/dx[x^3] = 3*x^2 = 12
        assert!((result.tan - 12.0).abs() < 1e-12);
    }

    // ---- compute_hessian_dual with more operation combos ----

    #[test]
    fn test_compute_hessian_dual_sub() {
        let ops = vec![OpKind::Inp(0), OpKind::Inp(1), OpKind::Sub];
        let h = compute_hessian_dual(&ops, &[1.0, 2.0]).unwrap();
        assert!((h[0][0]).abs() < 1e-12);
        assert!((h[0][1]).abs() < 1e-12);
        assert!((h[1][0]).abs() < 1e-12);
        assert!((h[1][1]).abs() < 1e-12);
    }

    #[test]
    fn test_compute_hessian_dual_div() {
        // f(x, y) = x / y  →  H = [[0, -1/y²], [-1/y², 2*x/y³]]
        let x = 6.0_f64;
        let y = 3.0_f64;
        let ops = vec![OpKind::Inp(0), OpKind::Inp(1), OpKind::Div];
        let h = compute_hessian_dual(&ops, &[x, y]).unwrap();
        assert!((h[0][0]).abs() < TOL);
        assert!((h[0][1] - (-1.0 / (y * y))).abs() < TOL);
        assert!((h[1][0] - (-1.0 / (y * y))).abs() < TOL);
        assert!((h[1][1] - (2.0 * x / (y * y * y))).abs() < TOL);
    }

    #[test]
    fn test_compute_hessian_dual_ln() {
        // f(x) = ln(x), f''(x) = -1/x²
        let x = 2.0_f64;
        let ops = vec![OpKind::Inp(0), OpKind::Ln];
        let h = compute_hessian_dual(&ops, &[x]).unwrap();
        assert!((h[0][0] - (-1.0 / (x * x))).abs() < TOL);
    }

    #[test]
    fn test_compute_hessian_dual_pow() {
        // f(x, y) = x^y with y constant (=3)
        // We can't make y constant in OpKind directly, so test with Inp for both
        let x = 2.0_f64;
        let y = 3.0_f64;
        let ops = vec![OpKind::Inp(0), OpKind::Inp(1), OpKind::Pow];
        let h = compute_hessian_dual(&ops, &[x, y]).unwrap();
        let expected_xx = y * (y - 1.0) * x.powf(y - 2.0);
        assert!((h[0][0] - expected_xx).abs() < TOL);
    }

    #[test]
    fn test_compute_hessian_dual_sqrt() {
        let x = 4.0_f64;
        let ops = vec![OpKind::Inp(0), OpKind::Sqrt];
        let h = compute_hessian_dual(&ops, &[x]).unwrap();
        let expected = -1.0 / (4.0 * x * x.sqrt());
        assert!((h[0][0] - expected).abs() < TOL);
    }

    #[test]
    fn test_compute_hessian_dual_cos_sin() {
        // f(x) = cos(sin(x))
        let x = 1.0_f64;
        let ops = vec![OpKind::Inp(0), OpKind::Sin, OpKind::Cos];
        let h = compute_hessian_dual(&ops, &[x]).unwrap();
        assert!(h[0][0].is_finite());
    }

    // ---- validate_rpn_structure edge cases ----

    #[test]
    fn test_validate_rpn_binary_missing_both() {
        let ops = vec![OpKind::Mul];
        let result = validate_rpn_structure(&ops, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rpn_empty_vars() {
        let ops = vec![OpKind::Inp(0), OpKind::Sin];
        let result = validate_rpn_structure(&ops, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_hessian_dual_empty_single_var() {
        let h = compute_hessian_dual(&[], &[5.0]).unwrap();
        assert_eq!(h, vec![vec![0.0]]);
    }

    const TOL: f64 = 1e-12;
}
