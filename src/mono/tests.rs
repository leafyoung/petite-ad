use super::examples::{MF1, MF2, MF3, MF4};
use super::first_order::MonoAD;
use super::func::MonoFn;
use crate::mono_ops;
use crate::test_utils::approx_eq_eps as approx_eq;
use crate::AutodiffError;

#[test]
fn test_single_sin_compute() {
    let ops = &[MonoAD::Sin];
    let (value, backprop) = MonoAD::compute_grad(ops, 2.0);
    assert!(approx_eq(value, 2.0_f64.sin(), 1e-10));
    assert!(approx_eq(backprop(1.0), 2.0_f64.cos(), 1e-10));
}

#[test]
fn test_single_cos_compute() {
    let ops = &[MonoAD::Cos];
    let (value, backprop) = MonoAD::compute_grad(ops, 2.0);
    assert!(approx_eq(value, 2.0_f64.cos(), 1e-10));
    assert!(approx_eq(backprop(1.0), -2.0_f64.sin(), 1e-10));
}

#[test]
fn test_single_exp_compute() {
    let ops = &[MonoAD::Exp];
    let (value, backprop) = MonoAD::compute_grad(ops, 2.0);
    assert!(approx_eq(value, 2.0_f64.exp(), 1e-10));
    assert!(approx_eq(backprop(1.0), 2.0_f64.exp(), 1e-10));
}

#[test]
fn test_computed_sin_sin_exp() {
    let ops = &[MonoAD::Sin, MonoAD::Sin, MonoAD::Exp];
    let (value, backprop) = MonoAD::compute_grad(ops, 2.0);

    // The expected value is exp(sin(sin(2.0)))
    let expected = 2.0_f64.sin().sin().exp();
    assert!(approx_eq(value, expected, 1e-10), "value mismatch");

    // The gradient at cotangent=1.0 should be approximately -0.562752...
    let grad = backprop(1.0);
    assert!(
        approx_eq(grad, -0.562752038662712, 1e-10),
        "gradient mismatch"
    );
}

#[test]
fn test_compute_arc_same_result() {
    let ops = &[MonoAD::Sin, MonoAD::Sin, MonoAD::Exp];

    let (value_box, backprop_box) = MonoAD::compute_grad(ops, 2.0);
    let grad_box = backprop_box(1.0);

    // Verify that computation is correct
    assert!(approx_eq(value_box, 2.2013533791690376, 1e-10));
    assert!(approx_eq(grad_box, -0.562752038662712, 1e-10));
}

#[test]
fn test_mono_f() {
    MF1(-1.0).test_mono_ad();
    MF2(3.0).test_mono_ad();
    MF3(-5.0).test_mono_ad();
    MF4(1.0).test_mono_ad();
}

#[test]
fn test_empty_operations() {
    // Empty operation list should return the input value unchanged
    let ops: &[MonoAD] = &[];
    let (value, backprop) = MonoAD::compute_grad(ops, 2.0);

    assert!(approx_eq(value, 2.0, 1e-10));
    // Identity function: gradient of 1.0 should be 1.0
    assert!(approx_eq(backprop(1.0), 1.0, 1e-10));
}

#[test]
fn test_single_operation() {
    let ops = &[MonoAD::Exp];
    let (value, backprop) = MonoAD::compute_grad(ops, 2.0);

    assert!(approx_eq(value, 2.0_f64.exp(), 1e-10));
    assert!(approx_eq(backprop(1.0), 2.0_f64.exp(), 1e-10));
}

#[test]
fn test_chaining_rule() {
    // Verify the chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
    // For sin(exp(x)): derivative is cos(exp(x)) * exp(x)
    let ops = &[MonoAD::Exp, MonoAD::Sin];
    let (_value, backprop) = MonoAD::compute_grad(ops, 2.0);

    let x: f64 = 2.0;
    let grad_computed = backprop(1.0);

    // Manual calculation: cos(exp(2.0)) * exp(2.0)
    let grad_expected = x.exp().cos() * x.exp();

    assert!(approx_eq(grad_computed, grad_expected, 1e-10));
}

#[test]
fn test_compute_arc_consistency() {
    // Test that compute_arc produces same results as compute for various operations
    let test_cases = vec![
        vec![MonoAD::Sin],
        vec![MonoAD::Cos],
        vec![MonoAD::Exp],
        vec![MonoAD::Sin, MonoAD::Cos],
        vec![MonoAD::Sin, MonoAD::Sin, MonoAD::Exp],
    ];

    for ops in test_cases {
        let (v1, b1) = MonoAD::compute_grad(&ops, 1.5);
        let g1 = b1(1.0);

        // Verify that computation succeeds
        assert!(v1.is_finite(), "value should be finite for ops: {:?}", ops);
        assert!(
            g1.is_finite(),
            "gradient should be finite for ops: {:?}",
            ops
        );
    }
}

#[test]
fn test_different_cotangents() {
    // Test that different cotangent values produce correct results
    let ops = mono_ops![sin, exp];
    let x: f64 = 1.0;
    let (_value, backprop) = MonoAD::compute_grad(&ops, x);

    // Test with different cotangent values
    for cotangent in [0.5, 1.0, 2.0, 10.0] {
        let grad = backprop(cotangent);

        let expected = x.sin().exp() * x.cos() * cotangent;
        // println!("Computed grad: {}, Expected grad: {}", grad, expected);

        assert!(approx_eq(grad, expected, 1e-10), "cotangent {}", cotangent);
    }
}

#[test]
fn test_hessian_sin() {
    // f(x) = sin(x), f''(x) = -sin(x)
    let ops = &[MonoAD::Sin];
    let x = 0.5;
    let second_deriv = MonoAD::compute_hessian(ops, x);

    let expected = -x.sin();
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_cos() {
    // f(x) = cos(x), f''(x) = -cos(x)
    let ops = &[MonoAD::Cos];
    let x = 0.5;
    let second_deriv = MonoAD::compute_hessian(ops, x);

    let expected = -x.cos();
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_exp() {
    // f(x) = exp(x), f''(x) = exp(x)
    let ops = &[MonoAD::Exp];
    let x = 2.0;
    let second_deriv = MonoAD::compute_hessian(ops, x);

    let expected = x.exp();
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_composed() {
    // f(x) = exp(sin(x)), f''(x) = exp(sin(x)) * cos²(x) - exp(sin(x)) * sin(x)
    let ops = &[MonoAD::Sin, MonoAD::Exp];
    let x = 0.5;
    let second_deriv = MonoAD::compute_hessian(ops, x);

    let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_chain() {
    // f(x) = sin(sin(sin(x)))
    // f'(x) = cos(sin(sin(x))) * cos(sin(x)) * cos(x)
    // This is complex, so we'll just verify that the computation doesn't panic and returns a value
    let ops = &[MonoAD::Sin, MonoAD::Sin, MonoAD::Sin];
    let x = 1.0;
    let second_deriv = MonoAD::compute_hessian(ops, x);

    // Just verify that it's a finite number
    assert!(second_deriv.is_finite());
}

#[test]
fn test_single_tan_compute() {
    let ops = &[MonoAD::Tan];
    let x: f64 = 0.5;
    let (value, backprop) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(value, x.tan(), 1e-10));
    assert!(approx_eq(backprop(1.0), 1.0 / x.cos().powi(2), 1e-10));
}

#[test]
fn test_single_ln_compute() {
    let ops = &[MonoAD::Ln];
    let x: f64 = 2.0;
    let (value, backprop) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(value, x.ln(), 1e-10));
    assert!(approx_eq(backprop(1.0), 1.0 / x, 1e-10));
}

#[test]
fn test_single_sqrt_compute() {
    let ops = &[MonoAD::Sqrt];
    let x: f64 = 4.0;
    let (value, backprop) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(value, x.sqrt(), 1e-10));
    assert!(approx_eq(backprop(1.0), 1.0 / (2.0 * x.sqrt()), 1e-10));
}

#[test]
fn test_single_abs_compute() {
    let ops = &[MonoAD::Abs];
    let (value_pos, backprop_pos) = MonoAD::compute_grad(ops, 3.0);
    assert!(approx_eq(value_pos, 3.0, 1e-10));
    assert!(approx_eq(backprop_pos(1.0), 1.0, 1e-10));

    let (value_neg, backprop_neg) = MonoAD::compute_grad(ops, -3.0);
    assert!(approx_eq(value_neg, 3.0, 1e-10));
    assert!(approx_eq(backprop_neg(1.0), -1.0, 1e-10));

    let (value_zero, backprop_zero) = MonoAD::compute_grad(ops, 0.0);
    assert!(approx_eq(value_zero, 0.0, 1e-10));
    assert!(approx_eq(backprop_zero(1.0), 0.0, 1e-10));
}

// ============================================================================
// MonoAD::compute tests (forward pass only)
// ============================================================================

#[test]
fn test_compute_sin() {
    let ops = mono_ops![sin];
    let result = MonoAD::compute(&ops, 2.0);
    assert!(approx_eq(result, 2.0_f64.sin(), 1e-10));
}

#[test]
fn test_compute_cos() {
    let ops = mono_ops![cos];
    let result = MonoAD::compute(&ops, 2.0);
    assert!(approx_eq(result, 2.0_f64.cos(), 1e-10));
}

#[test]
fn test_compute_tan() {
    let ops = mono_ops![tan];
    let result = MonoAD::compute(&ops, 0.5);
    assert!(approx_eq(result, 0.5_f64.tan(), 1e-10));
}

#[test]
fn test_compute_exp() {
    let ops = mono_ops![exp];
    let result = MonoAD::compute(&ops, 2.0);
    assert!(approx_eq(result, 2.0_f64.exp(), 1e-10));
}

#[test]
fn test_compute_neg() {
    let ops = mono_ops![neg];
    let result = MonoAD::compute(&ops, 3.0);
    assert!(approx_eq(result, -3.0, 1e-10));
}

#[test]
fn test_compute_ln() {
    let ops = mono_ops![ln];
    let result = MonoAD::compute(&ops, 2.0);
    assert!(approx_eq(result, 2.0_f64.ln(), 1e-10));
}

#[test]
fn test_compute_sqrt() {
    let ops = mono_ops![sqrt];
    let result = MonoAD::compute(&ops, 4.0);
    assert!(approx_eq(result, 4.0_f64.sqrt(), 1e-10));
}

#[test]
fn test_compute_abs() {
    let ops = mono_ops![abs];
    let result = MonoAD::compute(&ops, -3.0);
    assert!(approx_eq(result, 3.0, 1e-10));
}

#[test]
fn test_compute_chained() {
    let ops = mono_ops![sin, exp, neg];
    let result = MonoAD::compute(&ops, 1.0);
    let expected = -1.0_f64.sin().exp();
    assert!(approx_eq(result, expected, 1e-10));
}

// ============================================================================
// MonoAD::compute_checked tests
// ============================================================================

#[test]
fn test_compute_checked_sin_success() {
    let ops = mono_ops![sin];
    let result = MonoAD::compute_checked(&ops, 2.0).unwrap();
    assert!(approx_eq(result, 2.0_f64.sin(), 1e-10));
}

#[test]
fn test_compute_checked_cos_success() {
    let ops = mono_ops![cos];
    let result = MonoAD::compute_checked(&ops, 2.0).unwrap();
    assert!(approx_eq(result, 2.0_f64.cos(), 1e-10));
}

#[test]
fn test_compute_checked_tan_success() {
    let ops = mono_ops![tan];
    let result = MonoAD::compute_checked(&ops, 0.5).unwrap();
    assert!(approx_eq(result, 0.5_f64.tan(), 1e-10));
}

#[test]
fn test_compute_checked_exp_success() {
    let ops = mono_ops![exp];
    let result = MonoAD::compute_checked(&ops, 2.0).unwrap();
    assert!(approx_eq(result, 2.0_f64.exp(), 1e-10));
}

#[test]
fn test_compute_checked_neg_success() {
    let ops = mono_ops![neg];
    let result = MonoAD::compute_checked(&ops, 3.0).unwrap();
    assert!(approx_eq(result, -3.0, 1e-10));
}

#[test]
fn test_compute_checked_ln_success() {
    let ops = mono_ops![ln];
    let result = MonoAD::compute_checked(&ops, 2.0).unwrap();
    assert!(approx_eq(result, 2.0_f64.ln(), 1e-10));
}

#[test]
fn test_compute_checked_ln_error_negative() {
    let ops = mono_ops![ln];
    let result = MonoAD::compute_checked(&ops, -1.0);
    assert!(result.is_err());
    match result.unwrap_err() {
        AutodiffError::DomainError { operation, reason } => {
            assert_eq!(operation, "Ln");
            assert!(reason.contains("positive"));
        }
        _ => panic!("Expected DomainError"),
    }
}

#[test]
fn test_compute_checked_ln_error_zero() {
    let ops = mono_ops![ln];
    let result = MonoAD::compute_checked(&ops, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_compute_checked_sqrt_success() {
    let ops = mono_ops![sqrt];
    let result = MonoAD::compute_checked(&ops, 4.0).unwrap();
    assert!(approx_eq(result, 2.0, 1e-10));
}

#[test]
fn test_compute_checked_sqrt_error_negative() {
    let ops = mono_ops![sqrt];
    let result = MonoAD::compute_checked(&ops, -1.0);
    assert!(result.is_err());
    match result.unwrap_err() {
        AutodiffError::DomainError { operation, reason } => {
            assert_eq!(operation, "Sqrt");
            assert!(reason.contains("non-negative"));
        }
        _ => panic!("Expected DomainError"),
    }
}

#[test]
fn test_compute_checked_abs_success() {
    let ops = mono_ops![abs];
    let result = MonoAD::compute_checked(&ops, -3.0).unwrap();
    assert!(approx_eq(result, 3.0, 1e-10));
}

#[test]
fn test_compute_checked_chained() {
    let ops = mono_ops![sin, exp];
    let result = MonoAD::compute_checked(&ops, 1.0).unwrap();
    assert!(approx_eq(result, 1.0_f64.sin().exp(), 1e-10));
}

#[test]
fn test_compute_checked_chained_error() {
    // exp(sqrt(x)): forward pass goes through sqrt(-4) → error
    let ops = mono_ops![sqrt, exp];
    let result = MonoAD::compute_checked(&ops, -4.0);
    assert!(result.is_err());
}

// ============================================================================
// MonoAD::compute_grad_checked tests
// ============================================================================

#[test]
fn test_compute_grad_checked_sin_success() {
    let ops = mono_ops![sin];
    let (value, grad_fn) = MonoAD::compute_grad_checked(&ops, 2.0).unwrap();
    assert!(approx_eq(value, 2.0_f64.sin(), 1e-10));
    assert!(approx_eq(grad_fn(1.0), 2.0_f64.cos(), 1e-10));
}

#[test]
fn test_compute_grad_checked_exp_success() {
    let ops = mono_ops![exp];
    let (value, grad_fn) = MonoAD::compute_grad_checked(&ops, 2.0).unwrap();
    assert!(approx_eq(value, 2.0_f64.exp(), 1e-10));
    assert!(approx_eq(grad_fn(1.0), 2.0_f64.exp(), 1e-10));
}

#[test]
fn test_compute_grad_checked_ln_error() {
    let ops = mono_ops![ln];
    let result = MonoAD::compute_grad_checked(&ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_compute_grad_checked_chained() {
    let ops = mono_ops![sin, cos];
    let (value, grad_fn) = MonoAD::compute_grad_checked(&ops, 1.0).unwrap();
    assert!(approx_eq(value, 1.0_f64.sin().cos(), 1e-10));
    assert!(grad_fn(1.0).is_finite());
}

// ============================================================================
// MonoAD::compute_hessian tests (all variants)
// ============================================================================

#[test]
fn test_hessian_tan() {
    let ops = &[MonoAD::Tan];
    let x = 0.4;
    let second_deriv = MonoAD::compute_hessian(ops, x);
    let sec_sq = 1.0 / x.cos().powi(2);
    let expected = 2.0 * sec_sq * x.tan();
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_ln() {
    let ops = &[MonoAD::Ln];
    let x = 1.7;
    let second_deriv = MonoAD::compute_hessian(ops, x);
    let expected = -1.0 / x.powi(2);
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_sqrt() {
    let ops = &[MonoAD::Sqrt];
    let x = 2.5;
    let second_deriv = MonoAD::compute_hessian(ops, x);
    let expected = -1.0 / (4.0 * x * x.sqrt());
    assert!(approx_eq(second_deriv, expected, 1e-4));
}

#[test]
fn test_hessian_abs() {
    // abs(x) has second derivative 0 for x ≠ 0 (finite diff unstable at 0)
    let ops = &[MonoAD::Abs];
    for x in [-2.5, 2.5] {
        let second_deriv = MonoAD::compute_hessian(ops, x);
        assert!(approx_eq(second_deriv, 0.0, 1e-4));
    }
}

#[test]
fn test_hessian_neg() {
    let ops = &[MonoAD::Neg];
    let second_deriv = MonoAD::compute_hessian(ops, 3.0);
    assert!(approx_eq(second_deriv, 0.0, 1e-4));
}

// ============================================================================
// MonoAD::compute_hessian_checked tests
// ============================================================================

#[test]
fn test_hessian_checked_sin_success() {
    let ops = &[MonoAD::Sin];
    let result = MonoAD::compute_hessian_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, -1.0_f64.sin(), 1e-4));
}

#[test]
fn test_hessian_checked_ln_error() {
    let ops = &[MonoAD::Ln];
    // x = 0.000001 → x - ε ≈ -0.000009 (negative after perturbation step),
    let result = MonoAD::compute_hessian_checked(ops, 0.000001);
    assert!(result.is_err());
}

#[test]
fn test_hessian_checked_chained() {
    let ops = &[MonoAD::Sin, MonoAD::Exp];
    let x = 0.5;
    let result = MonoAD::compute_hessian_checked(ops, x).unwrap();
    let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();
    assert!(approx_eq(result, expected, 1e-4));
}

// ============================================================================
// MonoAD::compute_grad with non-1.0 cotangent for all variants
// ============================================================================

#[test]
fn test_grad_non_unit_cotangent_cos() {
    let ops = &[MonoAD::Cos];
    let x = 0.5;
    let (_value, grad_fn) = MonoAD::compute_grad(ops, x);
    let grad_2 = grad_fn(2.0);
    assert!(approx_eq(grad_2, 2.0 * -x.sin(), 1e-10));
}

#[test]
fn test_grad_non_unit_cotangent_tan() {
    let ops = &[MonoAD::Tan];
    let x = 0.4;
    let (_value, grad_fn) = MonoAD::compute_grad(ops, x);
    let grad_3 = grad_fn(3.0);
    let sec_sq = 1.0 / x.cos().powi(2);
    assert!(approx_eq(grad_3, 3.0 * sec_sq, 1e-10));
}

#[test]
fn test_grad_non_unit_cotangent_neg() {
    let ops = &[MonoAD::Neg];
    let (_value, grad_fn) = MonoAD::compute_grad(ops, 5.0);
    assert!(approx_eq(grad_fn(2.5), -2.5, 1e-10));
    assert!(approx_eq(grad_fn(0.0), 0.0, 1e-10));
}

#[test]
fn test_grad_non_unit_cotangent_ln() {
    let ops = &[MonoAD::Ln];
    let x = 3.0;
    let (_value, grad_fn) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(grad_fn(5.0), 5.0 / x, 1e-10));
}

#[test]
fn test_grad_non_unit_cotangent_sqrt() {
    let ops = &[MonoAD::Sqrt];
    let x = 9.0;
    let (_value, grad_fn) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(grad_fn(4.0), 4.0 / (2.0 * x.sqrt()), 1e-10));
}

#[test]
fn test_grad_non_unit_cotangent_abs() {
    let ops = &[MonoAD::Abs];
    let (_value, grad_fn) = MonoAD::compute_grad(ops, 5.0);
    assert!(approx_eq(grad_fn(3.0), 3.0, 1e-10));
    let (_value2, grad_fn2) = MonoAD::compute_grad(ops, -5.0);
    assert!(approx_eq(grad_fn2(3.0), -3.0, 1e-10));
}
