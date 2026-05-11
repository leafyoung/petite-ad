//! Shared local operation rules for multivariate autodiff.
//!
//! This module centralizes scalar values plus first- and second-order local
//! derivatives for [`super::multi_ad::MultiAD`] operations so first-order,
//! forward-mode, and exact Hessian implementations can reuse the same formulas.

use super::multi_ad::MultiAD;
use crate::{AutodiffError, Result};

#[inline(always)]
fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[inline(always)]
fn stable_log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[inline(always)]
fn stable_logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY && right == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else if left == f64::INFINITY || right == f64::INFINITY {
        f64::INFINITY
    } else {
        let max_value = left.max(right);
        max_value + ((left - max_value).exp() + (right - max_value).exp()).ln()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LocalRule {
    Unary {
        dy: f64,
        ddy: f64,
    },
    Binary {
        dy_left: f64,
        dy_right: f64,
        ddy_left_left: f64,
        ddy_right_right: f64,
        ddy_left_right: f64,
    },
}

#[inline(always)]
pub(crate) fn op_name(op: MultiAD) -> &'static str {
    match op {
        MultiAD::Inp => "Inp",
        MultiAD::Add => "Add",
        MultiAD::Sub => "Sub",
        MultiAD::Mul => "Mul",
        MultiAD::Div => "Div",
        MultiAD::Pow => "Pow",
        MultiAD::Sin => "Sin",
        MultiAD::Cos => "Cos",
        MultiAD::Tan => "Tan",
        MultiAD::Tanh => "Tanh",
        MultiAD::Relu => "Relu",
        MultiAD::Log1pExp => "Log1pExp",
        MultiAD::LogAddExp => "LogAddExp",
        MultiAD::Neg => "Neg",
        MultiAD::Exp => "Exp",
        MultiAD::Ln => "Ln",
        MultiAD::Sqrt => "Sqrt",
        MultiAD::Abs => "Abs",
    }
}

#[inline(always)]
pub(crate) fn expected_arity(op: MultiAD) -> usize {
    match op {
        MultiAD::Inp
        | MultiAD::Sin
        | MultiAD::Cos
        | MultiAD::Tan
        | MultiAD::Tanh
        | MultiAD::Relu
        | MultiAD::Log1pExp
        | MultiAD::Neg
        | MultiAD::Exp
        | MultiAD::Ln
        | MultiAD::Sqrt
        | MultiAD::Abs => 1,
        MultiAD::Add
        | MultiAD::Sub
        | MultiAD::Mul
        | MultiAD::Div
        | MultiAD::Pow
        | MultiAD::LogAddExp => 2,
    }
}

#[inline(always)]
pub(crate) fn check_arity(op: MultiAD, actual: usize) -> Result<()> {
    AutodiffError::check_arity(op_name(op), expected_arity(op), actual)
}

#[inline(always)]
pub(crate) fn check_domain(op: MultiAD, args: &[f64]) -> Result<()> {
    check_arity(op, args.len())?;
    match op {
        MultiAD::Div if args[1] == 0.0 => {
            Err(AutodiffError::domain("Div", "denominator must be non-zero"))
        }
        MultiAD::Ln if args[0] <= 0.0 => Err(AutodiffError::domain("Ln", "input must be positive")),
        MultiAD::Sqrt if args[0] < 0.0 => {
            Err(AutodiffError::domain("Sqrt", "input must be non-negative"))
        }
        MultiAD::Pow if args[0] <= 0.0 => Err(AutodiffError::domain(
            "Pow",
            "base must be positive in checked mode",
        )),
        _ => Ok(()),
    }
}

#[inline(always)]
pub(crate) fn forward_value(op: MultiAD, args: &[f64]) -> Result<f64> {
    check_arity(op, args.len())?;
    Ok(match op {
        MultiAD::Inp => args[0],
        MultiAD::Sin => args[0].sin(),
        MultiAD::Cos => args[0].cos(),
        MultiAD::Tan => args[0].tan(),
        MultiAD::Tanh => args[0].tanh(),
        MultiAD::Relu => args[0].max(0.0),
        MultiAD::Log1pExp => stable_log1p_exp(args[0]),
        MultiAD::Neg => -args[0],
        MultiAD::Exp => args[0].exp(),
        MultiAD::Ln => args[0].ln(),
        MultiAD::Sqrt => args[0].sqrt(),
        MultiAD::Abs => args[0].abs(),
        MultiAD::Add => args[0] + args[1],
        MultiAD::Sub => args[0] - args[1],
        MultiAD::Mul => args[0] * args[1],
        MultiAD::Div => args[0] / args[1],
        MultiAD::Pow => args[0].powf(args[1]),
        MultiAD::LogAddExp => stable_logaddexp(args[0], args[1]),
    })
}

#[inline(always)]
pub(crate) fn forward_value_checked(op: MultiAD, args: &[f64]) -> Result<f64> {
    check_domain(op, args)?;
    forward_value(op, args)
}

#[inline(always)]
pub(crate) fn local_rule(op: MultiAD, args: &[f64], value: f64) -> Result<LocalRule> {
    check_arity(op, args.len())?;
    Ok(match op {
        MultiAD::Sin => LocalRule::Unary {
            dy: args[0].cos(),
            ddy: -args[0].sin(),
        },
        MultiAD::Cos => LocalRule::Unary {
            dy: -args[0].sin(),
            ddy: -args[0].cos(),
        },
        MultiAD::Tan => {
            let sec_sq = 1.0 / args[0].cos().powi(2);
            LocalRule::Unary {
                dy: sec_sq,
                ddy: 2.0 * sec_sq * args[0].tan(),
            }
        }
        MultiAD::Tanh => {
            let tanh = value;
            let dy = 1.0 - tanh * tanh;
            LocalRule::Unary {
                dy,
                ddy: -2.0 * tanh * dy,
            }
        }
        MultiAD::Relu => {
            let dy = if args[0] > 0.0 { 1.0 } else { 0.0 };
            LocalRule::Unary { dy, ddy: 0.0 }
        }
        MultiAD::Log1pExp => {
            let dy = stable_sigmoid(args[0]);
            LocalRule::Unary {
                dy,
                ddy: dy * (1.0 - dy),
            }
        }
        MultiAD::Neg => LocalRule::Unary { dy: -1.0, ddy: 0.0 },
        MultiAD::Exp => LocalRule::Unary {
            dy: value,
            ddy: value,
        },
        MultiAD::Ln => LocalRule::Unary {
            dy: 1.0 / args[0],
            ddy: -1.0 / args[0].powi(2),
        },
        MultiAD::Sqrt => LocalRule::Unary {
            dy: 1.0 / (2.0 * value),
            ddy: -1.0 / (4.0 * args[0] * value),
        },
        MultiAD::Abs => {
            let sign = if args[0] > 0.0 {
                1.0
            } else if args[0] < 0.0 {
                -1.0
            } else {
                0.0
            };
            LocalRule::Unary { dy: sign, ddy: 0.0 }
        }
        MultiAD::Add => LocalRule::Binary {
            dy_left: 1.0,
            dy_right: 1.0,
            ddy_left_left: 0.0,
            ddy_right_right: 0.0,
            ddy_left_right: 0.0,
        },
        MultiAD::Sub => LocalRule::Binary {
            dy_left: 1.0,
            dy_right: -1.0,
            ddy_left_left: 0.0,
            ddy_right_right: 0.0,
            ddy_left_right: 0.0,
        },
        MultiAD::Mul => LocalRule::Binary {
            dy_left: args[1],
            dy_right: args[0],
            ddy_left_left: 0.0,
            ddy_right_right: 0.0,
            ddy_left_right: 1.0,
        },
        MultiAD::Div => LocalRule::Binary {
            dy_left: 1.0 / args[1],
            dy_right: -args[0] / args[1].powi(2),
            ddy_left_left: 0.0,
            ddy_right_right: 2.0 * args[0] / args[1].powi(3),
            ddy_left_right: -1.0 / args[1].powi(2),
        },
        MultiAD::Pow => LocalRule::Binary {
            dy_left: args[1] * args[0].powf(args[1] - 1.0),
            dy_right: if args[0] == 0.0 {
                0.0
            } else {
                value * args[0].ln()
            },
            ddy_left_left: args[1] * (args[1] - 1.0) * args[0].powf(args[1] - 2.0),
            ddy_right_right: if args[0] == 0.0 {
                0.0
            } else {
                value * args[0].ln().powi(2)
            },
            ddy_left_right: if args[0] == 0.0 {
                args[0].powf(args[1] - 1.0)
            } else {
                args[0].powf(args[1] - 1.0) * (1.0 + args[1] * args[0].ln())
            },
        },
        MultiAD::LogAddExp => {
            let (left_weight, right_weight) = if value == f64::NEG_INFINITY {
                (0.5, 0.5)
            } else if value == f64::INFINITY {
                match (args[0] == f64::INFINITY, args[1] == f64::INFINITY) {
                    (true, true) => (0.5, 0.5),
                    (true, false) => (1.0, 0.0),
                    (false, true) => (0.0, 1.0),
                    (false, false) => (0.5, 0.5),
                }
            } else {
                ((args[0] - value).exp(), (args[1] - value).exp())
            };
            LocalRule::Binary {
                dy_left: left_weight,
                dy_right: right_weight,
                ddy_left_left: left_weight * (1.0 - left_weight),
                ddy_right_right: right_weight * (1.0 - right_weight),
                ddy_left_right: -left_weight * right_weight,
            }
        }
        MultiAD::Inp => {
            return Err(AutodiffError::InvalidGraph {
                reason: "input markers do not have local derivative rules",
            });
        }
    })
}

#[inline(always)]
pub(crate) fn first_derivatives(op: MultiAD, args: &[f64], value: f64) -> Result<Vec<f64>> {
    Ok(match local_rule(op, args, value)? {
        LocalRule::Unary { dy, .. } => vec![dy],
        LocalRule::Binary {
            dy_left, dy_right, ..
        } => vec![dy_left, dy_right],
    })
}

#[inline(always)]
pub(crate) fn directional_tangent(
    op: MultiAD,
    args: &[f64],
    tangents: &[f64],
    value: f64,
) -> Result<f64> {
    let first = first_derivatives(op, args, value)?;
    Ok(first.iter().zip(tangents.iter()).map(|(d, t)| d * t).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::approx_eq_eps as approx_eq;

    const EPS: f64 = 1e-10;

    // ─── forward_value ────────────────────────────────────────────────

    #[test]
    fn test_forward_inp() {
        assert_eq!(forward_value(MultiAD::Inp, &[2.5]).unwrap(), 2.5);
    }

    #[test]
    fn test_forward_sin() {
        let val = forward_value(MultiAD::Sin, &[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.sin(), EPS));
    }

    #[test]
    fn test_forward_cos() {
        let val = forward_value(MultiAD::Cos, &[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.cos(), EPS));
    }

    #[test]
    fn test_forward_tan() {
        let val = forward_value(MultiAD::Tan, &[0.5]).unwrap();
        assert!(approx_eq(val, 0.5_f64.tan(), EPS));
    }

    #[test]
    fn test_forward_tanh() {
        let val = forward_value(MultiAD::Tanh, &[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.tanh(), EPS));
    }

    #[test]
    fn test_forward_relu_positive() {
        assert_eq!(forward_value(MultiAD::Relu, &[3.0]).unwrap(), 3.0);
    }

    #[test]
    fn test_forward_relu_negative() {
        assert_eq!(forward_value(MultiAD::Relu, &[-2.0]).unwrap(), 0.0);
    }

    #[test]
    fn test_forward_relu_zero() {
        assert_eq!(forward_value(MultiAD::Relu, &[0.0]).unwrap(), 0.0);
    }

    #[test]
    fn test_forward_log1p_exp() {
        // x > 0 branch
        let val = forward_value(MultiAD::Log1pExp, &[5.0]).unwrap();
        assert!(approx_eq(val, stable_log1p_exp(5.0), EPS));
        // x <= 0 branch
        let val2 = forward_value(MultiAD::Log1pExp, &[-5.0]).unwrap();
        assert!(approx_eq(val2, stable_log1p_exp(-5.0), EPS));
    }

    #[test]
    fn test_forward_neg() {
        assert_eq!(forward_value(MultiAD::Neg, &[7.0]).unwrap(), -7.0);
    }

    #[test]
    fn test_forward_exp() {
        let val = forward_value(MultiAD::Exp, &[1.0]).unwrap();
        assert!(approx_eq(val, std::f64::consts::E, EPS));
    }

    #[test]
    fn test_forward_ln() {
        let val = forward_value(MultiAD::Ln, &[std::f64::consts::E]).unwrap();
        assert!(approx_eq(val, 1.0, EPS));
    }

    #[test]
    fn test_forward_sqrt() {
        let val = forward_value(MultiAD::Sqrt, &[4.0]).unwrap();
        assert!(approx_eq(val, 2.0, EPS));
    }

    #[test]
    fn test_forward_abs() {
        assert_eq!(forward_value(MultiAD::Abs, &[-3.0]).unwrap(), 3.0);
        assert_eq!(forward_value(MultiAD::Abs, &[3.0]).unwrap(), 3.0);
    }

    #[test]
    fn test_forward_add() {
        assert_eq!(forward_value(MultiAD::Add, &[2.0, 3.0]).unwrap(), 5.0);
    }

    #[test]
    fn test_forward_sub() {
        assert_eq!(forward_value(MultiAD::Sub, &[2.0, 3.0]).unwrap(), -1.0);
    }

    #[test]
    fn test_forward_mul() {
        assert_eq!(forward_value(MultiAD::Mul, &[2.0, 3.0]).unwrap(), 6.0);
    }

    #[test]
    fn test_forward_div() {
        let val = forward_value(MultiAD::Div, &[6.0, 3.0]).unwrap();
        assert!(approx_eq(val, 2.0, EPS));
    }

    #[test]
    fn test_forward_pow() {
        let val = forward_value(MultiAD::Pow, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(val, 8.0, EPS));
    }

    #[test]
    fn test_forward_logaddexp_normal() {
        let val = forward_value(MultiAD::LogAddExp, &[1.0, 2.0]).unwrap();
        assert!(approx_eq(val, (1.0_f64.exp() + 2.0_f64.exp()).ln(), 1e-9));
    }

    // ─── arity checks ────────────────────────────────────────────────

    #[test]
    fn test_forward_wrong_arity() {
        assert!(forward_value(MultiAD::Sin, &[1.0, 2.0]).is_err());
        assert!(forward_value(MultiAD::Add, &[1.0]).is_err());
    }

    // ─── forward_value_checked / check_domain ────────────────────────

    #[test]
    fn test_checked_div_by_zero() {
        assert!(forward_value_checked(MultiAD::Div, &[1.0, 0.0]).is_err());
    }

    #[test]
    fn test_checked_ln_non_positive() {
        assert!(forward_value_checked(MultiAD::Ln, &[0.0]).is_err());
        assert!(forward_value_checked(MultiAD::Ln, &[-1.0]).is_err());
    }

    #[test]
    fn test_checked_sqrt_negative() {
        assert!(forward_value_checked(MultiAD::Sqrt, &[-1.0]).is_err());
    }

    #[test]
    fn test_checked_pow_non_positive_base() {
        assert!(forward_value_checked(MultiAD::Pow, &[0.0, 2.0]).is_err());
        assert!(forward_value_checked(MultiAD::Pow, &[-1.0, 2.0]).is_err());
    }

    #[test]
    fn test_checked_ok() {
        assert!(forward_value_checked(MultiAD::Sin, &[1.0]).is_ok());
        assert!(forward_value_checked(MultiAD::Div, &[1.0, 2.0]).is_ok());
        assert!(forward_value_checked(MultiAD::Ln, &[1.0]).is_ok());
        assert!(forward_value_checked(MultiAD::Sqrt, &[0.0]).is_ok());
        assert!(forward_value_checked(MultiAD::Pow, &[2.0, 3.0]).is_ok());
    }

    // ─── local_rule — unary ops ─────────────────────────────────────

    #[test]
    fn test_local_rule_sin() {
        let rule = local_rule(MultiAD::Sin, &[1.0], 1.0_f64.sin()).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, 1.0_f64.cos(), EPS));
                assert!(approx_eq(ddy, -1.0_f64.sin(), EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_cos() {
        let rule = local_rule(MultiAD::Cos, &[1.0], 1.0_f64.cos()).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, -1.0_f64.sin(), EPS));
                assert!(approx_eq(ddy, -1.0_f64.cos(), EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_tan() {
        let x = 0.5;
        let rule = local_rule(MultiAD::Tan, &[x], x.tan()).unwrap();
        let sec_sq = 1.0 / x.cos().powi(2);
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, sec_sq, EPS));
                assert!(approx_eq(ddy, 2.0 * sec_sq * x.tan(), EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_tanh() {
        let x: f64 = 0.7;
        let tanh_x = x.tanh();
        let rule = local_rule(MultiAD::Tanh, &[x], tanh_x).unwrap();
        let dy = 1.0 - tanh_x * tanh_x;
        match rule {
            LocalRule::Unary { dy: dy_val, ddy } => {
                assert!(approx_eq(dy_val, dy, EPS));
                assert!(approx_eq(ddy, -2.0 * tanh_x * dy, EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_relu() {
        // x > 0
        let rule = local_rule(MultiAD::Relu, &[2.0], 2.0).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert_eq!(dy, 1.0);
                assert_eq!(ddy, 0.0);
            }
            _ => panic!("expected Unary"),
        }
        // x < 0
        let rule2 = local_rule(MultiAD::Relu, &[-1.0], 0.0).unwrap();
        match rule2 {
            LocalRule::Unary { dy, .. } => assert_eq!(dy, 0.0),
            _ => panic!("expected Unary"),
        }
        // x == 0
        let rule3 = local_rule(MultiAD::Relu, &[0.0], 0.0).unwrap();
        match rule3 {
            LocalRule::Unary { dy, .. } => assert_eq!(dy, 0.0),
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_log1p_exp() {
        let x = 2.0;
        let value = stable_log1p_exp(x);
        let rule = local_rule(MultiAD::Log1pExp, &[x], value).unwrap();
        let sigma = stable_sigmoid(x);
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, sigma, EPS));
                assert!(approx_eq(ddy, sigma * (1.0 - sigma), EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_neg() {
        let rule = local_rule(MultiAD::Neg, &[5.0], -5.0).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert_eq!(dy, -1.0);
                assert_eq!(ddy, 0.0);
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_exp() {
        let x: f64 = 1.5;
        let value = x.exp();
        let rule = local_rule(MultiAD::Exp, &[x], value).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, value, EPS));
                assert!(approx_eq(ddy, value, EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_ln() {
        let x: f64 = 3.0;
        let value = x.ln();
        let rule = local_rule(MultiAD::Ln, &[x], value).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, 1.0 / x, EPS));
                assert!(approx_eq(ddy, -1.0 / (x * x), EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_sqrt() {
        let x: f64 = 9.0;
        let value = x.sqrt(); // 3.0
        let rule = local_rule(MultiAD::Sqrt, &[x], value).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert!(approx_eq(dy, 1.0 / (2.0 * value), EPS));
                assert!(approx_eq(ddy, -1.0 / (4.0 * x * value), EPS));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_local_rule_abs() {
        // positive
        let rule = local_rule(MultiAD::Abs, &[3.0], 3.0).unwrap();
        match rule {
            LocalRule::Unary { dy, ddy } => {
                assert_eq!(dy, 1.0);
                assert_eq!(ddy, 0.0);
            }
            _ => panic!(),
        }
        // negative
        let rule2 = local_rule(MultiAD::Abs, &[-3.0], 3.0).unwrap();
        match rule2 {
            LocalRule::Unary { dy, .. } => assert_eq!(dy, -1.0),
            _ => panic!(),
        }
        // zero
        let rule3 = local_rule(MultiAD::Abs, &[0.0], 0.0).unwrap();
        match rule3 {
            LocalRule::Unary { dy, .. } => assert_eq!(dy, 0.0),
            _ => panic!(),
        }
    }

    #[test]
    fn test_local_rule_inp_errors() {
        assert!(local_rule(MultiAD::Inp, &[1.0], 1.0).is_err());
    }

    // ─── local_rule — binary ops ─────────────────────────────────────

    #[test]
    fn test_local_rule_add() {
        let rule = local_rule(MultiAD::Add, &[2.0, 3.0], 5.0).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert_eq!(dy_left, 1.0);
                assert_eq!(dy_right, 1.0);
                assert_eq!(ddy_left_left, 0.0);
                assert_eq!(ddy_right_right, 0.0);
                assert_eq!(ddy_left_right, 0.0);
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_sub() {
        let rule = local_rule(MultiAD::Sub, &[2.0, 3.0], -1.0).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert_eq!(dy_left, 1.0);
                assert_eq!(dy_right, -1.0);
                assert_eq!(ddy_left_left, 0.0);
                assert_eq!(ddy_right_right, 0.0);
                assert_eq!(ddy_left_right, 0.0);
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_mul() {
        let rule = local_rule(MultiAD::Mul, &[2.0, 3.0], 6.0).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert_eq!(dy_left, 3.0); // args[1]
                assert_eq!(dy_right, 2.0); // args[0]
                assert_eq!(ddy_left_left, 0.0);
                assert_eq!(ddy_right_right, 0.0);
                assert_eq!(ddy_left_right, 1.0);
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_div() {
        let a = 6.0;
        let b = 3.0;
        let rule = local_rule(MultiAD::Div, &[a, b], a / b).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert!(approx_eq(dy_left, 1.0 / b, EPS));
                assert!(approx_eq(dy_right, -a / b.powi(2), EPS));
                assert_eq!(ddy_left_left, 0.0);
                assert!(approx_eq(ddy_right_right, 2.0 * a / b.powi(3), EPS));
                assert!(approx_eq(ddy_left_right, -1.0 / b.powi(2), EPS));
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_pow_normal() {
        let base: f64 = 2.0;
        let exp: f64 = 3.0;
        let value = base.powf(exp); // 8.0
        let rule = local_rule(MultiAD::Pow, &[base, exp], value).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert!(approx_eq(dy_left, exp * base.powf(exp - 1.0), EPS));
                assert!(approx_eq(dy_right, value * base.ln(), EPS));
                assert!(approx_eq(
                    ddy_left_left,
                    exp * (exp - 1.0) * base.powf(exp - 2.0),
                    EPS
                ));
                assert!(approx_eq(ddy_right_right, value * base.ln().powi(2), EPS));
                assert!(approx_eq(
                    ddy_left_right,
                    base.powf(exp - 1.0) * (1.0 + exp * base.ln()),
                    EPS
                ));
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_pow_zero_base() {
        // base == 0.0 triggers special branches
        let rule = local_rule(MultiAD::Pow, &[0.0, 2.0], 0.0_f64.powf(2.0)).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                // dy_left: exp * base^(exp-1) = 2 * 0^1 = 0
                assert_eq!(dy_left, 0.0);
                assert_eq!(dy_right, 0.0); // base == 0.0 branch
                                           // ddy_left_left: exp*(exp-1)*base^(exp-2) = 2*1*0^0 = 2*1*1 = 2
                assert!(approx_eq(ddy_left_left, 2.0, EPS));
                assert_eq!(ddy_right_right, 0.0);
                // ddy_left_right: args[0].powf(args[1]-1) = 0^(1) = 0
                assert!(approx_eq(ddy_left_right, 0.0_f64.powf(2.0 - 1.0), EPS));
            }
            _ => panic!("expected Binary"),
        }
    }

    // ─── local_rule — LogAddExp edge cases ───────────────────────────

    #[test]
    fn test_local_rule_logaddexp_normal() {
        let a = 1.0;
        let b = 2.0;
        let value = stable_logaddexp(a, b);
        let rule = local_rule(MultiAD::LogAddExp, &[a, b], value).unwrap();
        let lw = (a - value).exp();
        let rw = (b - value).exp();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert!(approx_eq(dy_left, lw, 1e-9));
                assert!(approx_eq(dy_right, rw, 1e-9));
                assert!(approx_eq(ddy_left_left, lw * (1.0 - lw), 1e-9));
                assert!(approx_eq(ddy_right_right, rw * (1.0 - rw), 1e-9));
                assert!(approx_eq(ddy_left_right, -lw * rw, 1e-9));
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_logaddexp_neg_inf() {
        let rule = local_rule(
            MultiAD::LogAddExp,
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            f64::NEG_INFINITY,
        )
        .unwrap();
        match rule {
            LocalRule::Binary {
                dy_left,
                dy_right,
                ddy_left_left,
                ddy_right_right,
                ddy_left_right,
            } => {
                assert_eq!(dy_left, 0.5);
                assert_eq!(dy_right, 0.5);
                assert_eq!(ddy_left_left, 0.25);
                assert_eq!(ddy_right_right, 0.25);
                assert_eq!(ddy_left_right, -0.25);
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_local_rule_logaddexp_both_inf() {
        let rule = local_rule(
            MultiAD::LogAddExp,
            &[f64::INFINITY, f64::INFINITY],
            f64::INFINITY,
        )
        .unwrap();
        match rule {
            LocalRule::Binary {
                dy_left, dy_right, ..
            } => {
                assert_eq!(dy_left, 0.5);
                assert_eq!(dy_right, 0.5);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_local_rule_logaddexp_left_inf() {
        let rule = local_rule(MultiAD::LogAddExp, &[f64::INFINITY, 1.0], f64::INFINITY).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left, dy_right, ..
            } => {
                assert_eq!(dy_left, 1.0);
                assert_eq!(dy_right, 0.0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_local_rule_logaddexp_right_inf() {
        let rule = local_rule(MultiAD::LogAddExp, &[1.0, f64::INFINITY], f64::INFINITY).unwrap();
        match rule {
            LocalRule::Binary {
                dy_left, dy_right, ..
            } => {
                assert_eq!(dy_left, 0.0);
                assert_eq!(dy_right, 1.0);
            }
            _ => panic!(),
        }
    }

    // ─── directional_tangent ─────────────────────────────────────────

    #[test]
    fn test_directional_tangent_sin() {
        let x: f64 = 1.0;
        let value = x.sin();
        let t = directional_tangent(MultiAD::Sin, &[x], &[0.5], value).unwrap();
        assert!(approx_eq(t, x.cos() * 0.5, EPS));
    }

    #[test]
    fn test_directional_tangent_add() {
        let t = directional_tangent(MultiAD::Add, &[2.0, 3.0], &[1.0, -1.0], 5.0).unwrap();
        // dy_left * t_left + dy_right * t_right = 1*1 + 1*(-1) = 0
        assert!(approx_eq(t, 0.0, EPS));
    }

    #[test]
    fn test_directional_tangent_mul() {
        let t = directional_tangent(MultiAD::Mul, &[2.0, 3.0], &[1.0, 2.0], 6.0).unwrap();
        // dy_left*1 + dy_right*2 = 3*1 + 2*2 = 7
        assert!(approx_eq(t, 7.0, EPS));
    }

    #[test]
    fn test_directional_tangent_exp() {
        let x: f64 = 1.0;
        let value = x.exp();
        let t = directional_tangent(MultiAD::Exp, &[x], &[2.0], value).unwrap();
        assert!(approx_eq(t, value * 2.0, EPS));
    }

    #[test]
    fn test_directional_tangent_div() {
        let a = 6.0;
        let b = 3.0;
        let t = directional_tangent(MultiAD::Div, &[a, b], &[1.0, 1.0], a / b).unwrap();
        // (1/3)*1 + (-6/9)*1 = 1/3 - 2/3 = -1/3
        assert!(approx_eq(t, 1.0 / 3.0 - 2.0 / 3.0, EPS));
    }

    // ─── first_derivatives ──────────────────────────────────────────

    #[test]
    fn test_first_derivatives_unary() {
        let derivs = first_derivatives(MultiAD::Neg, &[5.0], -5.0).unwrap();
        assert_eq!(derivs, vec![-1.0]);
    }

    #[test]
    fn test_first_derivatives_binary() {
        let derivs = first_derivatives(MultiAD::Mul, &[2.0, 3.0], 6.0).unwrap();
        assert_eq!(derivs, vec![3.0, 2.0]);
    }

    // ─── op_name / expected_arity / check_arity ─────────────────────

    #[test]
    fn test_op_name_all() {
        assert_eq!(op_name(MultiAD::Inp), "Inp");
        assert_eq!(op_name(MultiAD::Add), "Add");
        assert_eq!(op_name(MultiAD::Sub), "Sub");
        assert_eq!(op_name(MultiAD::Mul), "Mul");
        assert_eq!(op_name(MultiAD::Div), "Div");
        assert_eq!(op_name(MultiAD::Pow), "Pow");
        assert_eq!(op_name(MultiAD::Sin), "Sin");
        assert_eq!(op_name(MultiAD::Cos), "Cos");
        assert_eq!(op_name(MultiAD::Tan), "Tan");
        assert_eq!(op_name(MultiAD::Tanh), "Tanh");
        assert_eq!(op_name(MultiAD::Relu), "Relu");
        assert_eq!(op_name(MultiAD::Log1pExp), "Log1pExp");
        assert_eq!(op_name(MultiAD::LogAddExp), "LogAddExp");
        assert_eq!(op_name(MultiAD::Neg), "Neg");
        assert_eq!(op_name(MultiAD::Exp), "Exp");
        assert_eq!(op_name(MultiAD::Ln), "Ln");
        assert_eq!(op_name(MultiAD::Sqrt), "Sqrt");
        assert_eq!(op_name(MultiAD::Abs), "Abs");
    }

    #[test]
    fn test_expected_arity() {
        assert_eq!(expected_arity(MultiAD::Sin), 1);
        assert_eq!(expected_arity(MultiAD::Cos), 1);
        assert_eq!(expected_arity(MultiAD::Tan), 1);
        assert_eq!(expected_arity(MultiAD::Tanh), 1);
        assert_eq!(expected_arity(MultiAD::Relu), 1);
        assert_eq!(expected_arity(MultiAD::Log1pExp), 1);
        assert_eq!(expected_arity(MultiAD::Neg), 1);
        assert_eq!(expected_arity(MultiAD::Exp), 1);
        assert_eq!(expected_arity(MultiAD::Ln), 1);
        assert_eq!(expected_arity(MultiAD::Sqrt), 1);
        assert_eq!(expected_arity(MultiAD::Abs), 1);
        assert_eq!(expected_arity(MultiAD::Inp), 1);
        assert_eq!(expected_arity(MultiAD::Add), 2);
        assert_eq!(expected_arity(MultiAD::Sub), 2);
        assert_eq!(expected_arity(MultiAD::Mul), 2);
        assert_eq!(expected_arity(MultiAD::Div), 2);
        assert_eq!(expected_arity(MultiAD::Pow), 2);
        assert_eq!(expected_arity(MultiAD::LogAddExp), 2);
    }

    #[test]
    fn test_check_arity_ok() {
        assert!(check_arity(MultiAD::Sin, 1).is_ok());
        assert!(check_arity(MultiAD::Add, 2).is_ok());
    }

    #[test]
    fn test_check_arity_fail() {
        assert!(check_arity(MultiAD::Sin, 2).is_err());
        assert!(check_arity(MultiAD::Add, 1).is_err());
    }

    // ─── stable helpers ─────────────────────────────────────────────

    #[test]
    fn test_stable_sigmoid() {
        // x >= 0
        assert!(approx_eq(stable_sigmoid(0.0), 0.5, EPS));
        assert!(approx_eq(
            stable_sigmoid(10.0),
            1.0 / (1.0 + (-10.0_f64).exp()),
            EPS
        ));
        // x < 0
        let exp_x = (-5.0_f64).exp();
        assert!(approx_eq(stable_sigmoid(-5.0), exp_x / (1.0 + exp_x), EPS));
    }

    #[test]
    fn test_stable_log1p_exp_positive() {
        // x > 0 branch
        let x: f64 = 5.0;
        let expected = x + (-x).exp().ln_1p();
        assert!(approx_eq(stable_log1p_exp(x), expected, EPS));
    }

    #[test]
    fn test_stable_log1p_exp_nonpositive() {
        // x <= 0 branch
        let x: f64 = -3.0;
        let expected = x.exp().ln_1p();
        assert!(approx_eq(stable_log1p_exp(x), expected, EPS));
    }

    #[test]
    fn test_stable_logaddexp_both_neg_inf() {
        assert_eq!(
            stable_logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_stable_logaddexp_one_inf() {
        assert_eq!(stable_logaddexp(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(stable_logaddexp(1.0, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn test_stable_logaddexp_normal() {
        let val = stable_logaddexp(1.0, 2.0);
        let max_v = 2.0_f64;
        let expected = max_v + ((1.0 - max_v).exp() + (2.0 - max_v).exp()).ln();
        assert!(approx_eq(val, expected, EPS));
    }

    // ─── forward_value edge cases ────────────────────────────────────

    #[test]
    fn test_forward_div_by_zero_unchecked() {
        // forward_value does NOT check domain; f64 division yields inf
        let val = forward_value(MultiAD::Div, &[1.0, 0.0]).unwrap();
        assert!(val.is_infinite());
    }

    #[test]
    fn test_forward_ln_zero_unchecked() {
        // forward_value does NOT check domain
        let val = forward_value(MultiAD::Ln, &[0.0]).unwrap();
        assert!(val.is_infinite() || val.is_nan());
    }

    #[test]
    fn test_forward_sqrt_zero() {
        let val = forward_value(MultiAD::Sqrt, &[0.0]).unwrap();
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_forward_log1p_exp_large_positive() {
        // For large x, log1p(exp(x)) ≈ x
        let val = forward_value(MultiAD::Log1pExp, &[100.0]).unwrap();
        assert!(approx_eq(val, 100.0, 1e-6));
    }

    #[test]
    fn test_forward_log1p_exp_large_negative() {
        // For very negative x, log1p(exp(x)) ≈ exp(x) ≈ 0
        let val = forward_value(MultiAD::Log1pExp, &[-100.0]).unwrap();
        assert!(val < 1e-40);
    }
}
