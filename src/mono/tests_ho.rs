//! Tests for higher-order autodiff methods (RR, FR, RF).

use super::second_order::fr::MonoAD2FR;
use super::second_order::rf::MonoAD2RF;
use super::second_order::rr::MonoAD2RR;
use crate::test_utils::approx_eq_eps as approx_eq;

// Common tolerance for exact autodiff (machine precision)
const EXACT_TOL: f64 = 1e-12;
const COMPARISON_TOL: f64 = 1e-10;

// ============================================================================
// MonoAD2RR (Reverse-over-Reverse) Tests
// ============================================================================

#[test]
fn test_rr_sin_hessian() {
    // f(x) = sin(x), f''(x) = -sin(x)
    let ops = &[MonoAD2RR::Sin];
    let x = 0.5;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    let expected = -x.sin();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rr_cos_hessian() {
    // f(x) = cos(x), f''(x) = -cos(x)
    let ops = &[MonoAD2RR::Cos];
    let x = 0.5;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    let expected = -x.cos();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rr_exp_hessian() {
    // f(x) = exp(x), f''(x) = exp(x)
    let ops = &[MonoAD2RR::Exp];
    let x = 2.0;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    let expected = x.exp();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rr_tan_hessian() {
    let ops = &[MonoAD2RR::Tan];
    let x: f64 = 0.4;
    let sec_sq = 1.0 / x.cos().powi(2);
    let expected = 2.0 * sec_sq * x.tan();
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rr_ln_hessian() {
    let ops = &[MonoAD2RR::Ln];
    let x: f64 = 1.7;
    let expected = -1.0 / x.powi(2);
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rr_sqrt_hessian() {
    let ops = &[MonoAD2RR::Sqrt];
    let x: f64 = 2.5;
    let expected = -1.0 / (4.0 * x * x.sqrt());
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rr_abs_hessian() {
    let ops = &[MonoAD2RR::Abs];
    for x in [-2.5, 0.0, 2.5] {
        let hessian = MonoAD2RR::compute_hessian(ops, x);
        assert!(approx_eq(hessian, 0.0, EXACT_TOL));
    }
}

#[test]
fn test_rr_neg_hessian() {
    // f(x) = -x, f''(x) = 0
    let ops = &[MonoAD2RR::Neg];
    let x = 3.0;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, 0.0, EXACT_TOL));
}

#[test]
fn test_rr_composed_hessian() {
    // f(x) = exp(sin(x)), f''(x) = exp(sin(x)) * cos²(x) - exp(sin(x)) * sin(x)
    let ops = &[MonoAD2RR::Sin, MonoAD2RR::Exp];
    let x = 0.5;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();
    assert!(approx_eq(hessian, expected, COMPARISON_TOL));
}

#[test]
fn test_rr_complex_chain() {
    // f(x) = -exp(sin(x))
    let ops = &[MonoAD2RR::Sin, MonoAD2RR::Exp, MonoAD2RR::Neg];
    let x = 1.0;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(hessian.is_finite());
}

// ============================================================================
// MonoAD2FR compute / compute_checked / compute_grad_checked tests
// ============================================================================

#[test]
fn test_fr_compute_sin() {
    let ops = &[MonoAD2FR::Sin];
    let result = MonoAD2FR::compute(ops, 1.0);
    assert!(approx_eq(result, 1.0_f64.sin(), COMPARISON_TOL));
}

#[test]
fn test_fr_compute_neg() {
    let ops = &[MonoAD2FR::Neg];
    let result = MonoAD2FR::compute(ops, 3.0);
    assert!(approx_eq(result, -3.0, COMPARISON_TOL));
}

#[test]
fn test_fr_compute_checked_success() {
    let ops = &[MonoAD2FR::Sin, MonoAD2FR::Exp];
    let result = MonoAD2FR::compute_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, 1.0_f64.sin().exp(), COMPARISON_TOL));
}

#[test]
fn test_fr_compute_checked_ln_error() {
    let ops = &[MonoAD2FR::Ln];
    let result = MonoAD2FR::compute_checked(ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_fr_compute_checked_sqrt_error() {
    let ops = &[MonoAD2FR::Sqrt];
    let result = MonoAD2FR::compute_checked(ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_fr_compute_grad_checked_success() {
    let ops = &[MonoAD2FR::Sin];
    let (val, grad_fn) = MonoAD2FR::compute_grad_checked(ops, 1.0).unwrap();
    assert!(approx_eq(val, 1.0_f64.sin(), COMPARISON_TOL));
    assert!(approx_eq(grad_fn(1.0), 1.0_f64.cos(), COMPARISON_TOL));
}

#[test]
fn test_fr_compute_grad_checked_error() {
    let ops = &[MonoAD2FR::Ln];
    let result = MonoAD2FR::compute_grad_checked(ops, -2.0);
    assert!(result.is_err());
}

#[test]
fn test_fr_hessian_empty() {
    let ops: &[MonoAD2FR] = &[];
    let hessian = MonoAD2FR::compute_hessian(ops, 1.0);
    assert!(approx_eq(hessian, 0.0, EXACT_TOL));
}

#[test]
fn test_fr_hessian_checked_success() {
    let ops = &[MonoAD2FR::Sin];
    let result = MonoAD2FR::compute_hessian_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, -1.0_f64.sin(), EXACT_TOL));
}

// ============================================================================
// MonoAD2FR (Forward-over-Reverse) Hessian Tests
// ============================================================================

#[test]
fn test_fr_sin_hessian() {
    let ops = &[MonoAD2FR::Sin];
    let x = 0.5;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    let expected = -x.sin();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_fr_cos_hessian() {
    let ops = &[MonoAD2FR::Cos];
    let x = 0.5;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    let expected = -x.cos();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_fr_exp_hessian() {
    let ops = &[MonoAD2FR::Exp];
    let x = 2.0;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    let expected = x.exp();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_fr_tan_hessian() {
    let ops = &[MonoAD2FR::Tan];
    let x: f64 = 0.4;
    let sec_sq = 1.0 / x.cos().powi(2);
    let expected = 2.0 * sec_sq * x.tan();
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_fr_ln_hessian() {
    let ops = &[MonoAD2FR::Ln];
    let x: f64 = 1.7;
    let expected = -1.0 / x.powi(2);
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_fr_sqrt_hessian() {
    let ops = &[MonoAD2FR::Sqrt];
    let x: f64 = 2.5;
    let expected = -1.0 / (4.0 * x * x.sqrt());
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_fr_abs_hessian() {
    let ops = &[MonoAD2FR::Abs];
    for x in [-2.5, 0.0, 2.5] {
        let hessian = MonoAD2FR::compute_hessian(ops, x);
        assert!(approx_eq(hessian, 0.0, EXACT_TOL));
    }
}

#[test]
fn test_fr_neg_hessian() {
    let ops = &[MonoAD2FR::Neg];
    let x = 3.0;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, 0.0, EXACT_TOL));
}

#[test]
fn test_fr_composed_hessian() {
    let ops = &[MonoAD2FR::Sin, MonoAD2FR::Exp];
    let x = 0.5;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();
    assert!(approx_eq(hessian, expected, COMPARISON_TOL));
}

// ============================================================================
// MonoAD2RF compute / compute_checked / compute_grad_checked tests
// ============================================================================

#[test]
fn test_rf_compute_sin() {
    let ops = &[MonoAD2RF::Sin];
    let result = MonoAD2RF::compute(ops, 1.0);
    assert!(approx_eq(result, 1.0_f64.sin(), COMPARISON_TOL));
}

#[test]
fn test_rf_compute_neg() {
    let ops = &[MonoAD2RF::Neg];
    let result = MonoAD2RF::compute(ops, 3.0);
    assert!(approx_eq(result, -3.0, COMPARISON_TOL));
}

#[test]
fn test_rf_compute_checked_success() {
    let ops = &[MonoAD2RF::Sin, MonoAD2RF::Exp];
    let result = MonoAD2RF::compute_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, 1.0_f64.sin().exp(), COMPARISON_TOL));
}

#[test]
fn test_rf_compute_checked_ln_error() {
    let ops = &[MonoAD2RF::Ln];
    let result = MonoAD2RF::compute_checked(ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_rf_compute_checked_sqrt_error() {
    let ops = &[MonoAD2RF::Sqrt];
    let result = MonoAD2RF::compute_checked(ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_rf_compute_grad_checked_success() {
    let ops = &[MonoAD2RF::Sin];
    let (val, grad_fn) = MonoAD2RF::compute_grad_checked(ops, 1.0).unwrap();
    assert!(approx_eq(val, 1.0_f64.sin(), COMPARISON_TOL));
    assert!(approx_eq(grad_fn(1.0), 1.0_f64.cos(), COMPARISON_TOL));
}

#[test]
fn test_rf_compute_grad_checked_error() {
    let ops = &[MonoAD2RF::Ln];
    let result = MonoAD2RF::compute_grad_checked(ops, -2.0);
    assert!(result.is_err());
}

#[test]
fn test_rf_hessian_empty() {
    let ops: &[MonoAD2RF] = &[];
    let hessian = MonoAD2RF::compute_hessian(ops, 1.0);
    assert!(approx_eq(hessian, 0.0, EXACT_TOL));
}

#[test]
fn test_rf_hessian_checked_success() {
    let ops = &[MonoAD2RF::Sin];
    let result = MonoAD2RF::compute_hessian_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, -1.0_f64.sin(), EXACT_TOL));
}

// ============================================================================
// MonoAD2RF (Reverse-over-Forward) Hessian Tests
// ============================================================================

#[test]
fn test_rf_sin_hessian() {
    let ops = &[MonoAD2RF::Sin];
    let x = 0.5;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    let expected = -x.sin();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rf_cos_hessian() {
    let ops = &[MonoAD2RF::Cos];
    let x = 0.5;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    let expected = -x.cos();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rf_exp_hessian() {
    let ops = &[MonoAD2RF::Exp];
    let x = 2.0;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    let expected = x.exp();
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rf_tan_hessian() {
    let ops = &[MonoAD2RF::Tan];
    let x: f64 = 0.4;
    let sec_sq = 1.0 / x.cos().powi(2);
    let expected = 2.0 * sec_sq * x.tan();
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rf_ln_hessian() {
    let ops = &[MonoAD2RF::Ln];
    let x: f64 = 1.7;
    let expected = -1.0 / x.powi(2);
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rf_sqrt_hessian() {
    let ops = &[MonoAD2RF::Sqrt];
    let x: f64 = 2.5;
    let expected = -1.0 / (4.0 * x * x.sqrt());
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, expected, EXACT_TOL));
}

#[test]
fn test_rf_abs_hessian() {
    let ops = &[MonoAD2RF::Abs];
    for x in [-2.5, 0.0, 2.5] {
        let hessian = MonoAD2RF::compute_hessian(ops, x);
        assert!(approx_eq(hessian, 0.0, EXACT_TOL));
    }
}

#[test]
fn test_rf_neg_hessian() {
    let ops = &[MonoAD2RF::Neg];
    let x = 3.0;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, 0.0, EXACT_TOL));
}

#[test]
fn test_rf_composed_hessian() {
    let ops = &[MonoAD2RF::Sin, MonoAD2RF::Exp];
    let x = 0.5;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();
    assert!(approx_eq(hessian, expected, COMPARISON_TOL));
}

// ============================================================================
// MonoAD2RR compute / compute_checked / compute_grad_checked tests
// ============================================================================

#[test]
fn test_rr_compute_sin() {
    let ops = &[MonoAD2RR::Sin];
    let result = MonoAD2RR::compute(ops, 1.0);
    assert!(approx_eq(result, 1.0_f64.sin(), COMPARISON_TOL));
}

#[test]
fn test_rr_compute_neg() {
    let ops = &[MonoAD2RR::Neg];
    let result = MonoAD2RR::compute(ops, 3.0);
    assert!(approx_eq(result, -3.0, COMPARISON_TOL));
}

#[test]
fn test_rr_compute_ln() {
    let ops = &[MonoAD2RR::Ln];
    let result = MonoAD2RR::compute(ops, 2.0);
    assert!(approx_eq(result, 2.0_f64.ln(), COMPARISON_TOL));
}

#[test]
fn test_rr_compute_sqrt() {
    let ops = &[MonoAD2RR::Sqrt];
    let result = MonoAD2RR::compute(ops, 4.0);
    assert!(approx_eq(result, 2.0, COMPARISON_TOL));
}

#[test]
fn test_rr_compute_abs() {
    let ops = &[MonoAD2RR::Abs];
    let result = MonoAD2RR::compute(ops, -5.0);
    assert!(approx_eq(result, 5.0, COMPARISON_TOL));
}

#[test]
fn test_rr_compute_checked_success() {
    let ops = &[MonoAD2RR::Sin, MonoAD2RR::Exp];
    let result = MonoAD2RR::compute_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, 1.0_f64.sin().exp(), COMPARISON_TOL));
}

#[test]
fn test_rr_compute_checked_ln_error() {
    let ops = &[MonoAD2RR::Ln];
    let result = MonoAD2RR::compute_checked(ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_rr_compute_checked_sqrt_error() {
    let ops = &[MonoAD2RR::Sqrt];
    let result = MonoAD2RR::compute_checked(ops, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_rr_compute_grad_non_unit_cotangent() {
    let ops = &[MonoAD2RR::Sin];
    let (_val, grad_fn) = MonoAD2RR::compute_grad(ops, 1.0);
    assert!(approx_eq(grad_fn(3.0), 3.0 * 1.0_f64.cos(), COMPARISON_TOL));
}

#[test]
fn test_rr_compute_grad_checked_success() {
    let ops = &[MonoAD2RR::Sin];
    let (val, grad_fn) = MonoAD2RR::compute_grad_checked(ops, 1.0).unwrap();
    assert!(approx_eq(val, 1.0_f64.sin(), COMPARISON_TOL));
    assert!(approx_eq(grad_fn(1.0), 1.0_f64.cos(), COMPARISON_TOL));
}

#[test]
fn test_rr_compute_grad_checked_error() {
    let ops = &[MonoAD2RR::Ln];
    let result = MonoAD2RR::compute_grad_checked(ops, -2.0);
    assert!(result.is_err());
}

#[test]
fn test_rr_hessian_checked_success() {
    let ops = &[MonoAD2RR::Sin];
    let result = MonoAD2RR::compute_hessian_checked(ops, 1.0).unwrap();
    assert!(approx_eq(result, -1.0_f64.sin(), EXACT_TOL));
}

#[test]
fn test_rr_hessian_checked_error() {
    // ln(-1.0) fails domain check
    let ops = &[MonoAD2RR::Ln];
    let result = MonoAD2RR::compute_hessian_checked(ops, -1.0);
    assert!(result.is_err());
}

// ============================================================================
// Consistency Tests: All three methods should produce identical results
// ============================================================================

#[test]
fn test_consistency_all_methods_sin() {
    let ops_rr = &[MonoAD2RR::Sin];
    let ops_fr = &[MonoAD2FR::Sin];
    let ops_rf = &[MonoAD2RF::Sin];
    let x = 0.5;

    let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
    let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
    let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

    let expected = -x.sin();
    assert!(approx_eq(h_rr, expected, EXACT_TOL));
    assert!(approx_eq(h_fr, expected, EXACT_TOL));
    assert!(approx_eq(h_rf, expected, EXACT_TOL));

    // All methods should agree
    assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
    assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
}

#[test]
fn test_consistency_all_methods_exp() {
    let ops_rr = &[MonoAD2RR::Exp];
    let ops_fr = &[MonoAD2FR::Exp];
    let ops_rf = &[MonoAD2RF::Exp];
    let x = 2.0;

    let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
    let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
    let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

    let expected = x.exp();
    assert!(approx_eq(h_rr, expected, EXACT_TOL));
    assert!(approx_eq(h_fr, expected, EXACT_TOL));
    assert!(approx_eq(h_rf, expected, EXACT_TOL));

    assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
    assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
}

#[test]
fn test_consistency_all_methods_tan() {
    let ops_rr = &[MonoAD2RR::Tan];
    let ops_fr = &[MonoAD2FR::Tan];
    let ops_rf = &[MonoAD2RF::Tan];
    let x: f64 = 0.4;

    let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
    let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
    let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

    let sec_sq = 1.0 / x.cos().powi(2);
    let expected = 2.0 * sec_sq * x.tan();
    assert!(approx_eq(h_rr, expected, EXACT_TOL));
    assert!(approx_eq(h_fr, expected, EXACT_TOL));
    assert!(approx_eq(h_rf, expected, EXACT_TOL));

    assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
    assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
}

#[test]
fn test_consistency_all_methods_ln() {
    let ops_rr = &[MonoAD2RR::Ln];
    let ops_fr = &[MonoAD2FR::Ln];
    let ops_rf = &[MonoAD2RF::Ln];
    let x: f64 = 1.7;

    let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
    let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
    let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

    let expected = -1.0 / x.powi(2);
    assert!(approx_eq(h_rr, expected, EXACT_TOL));
    assert!(approx_eq(h_fr, expected, EXACT_TOL));
    assert!(approx_eq(h_rf, expected, EXACT_TOL));

    assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
    assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
}

#[test]
fn test_consistency_all_methods_sqrt() {
    let ops_rr = &[MonoAD2RR::Sqrt];
    let ops_fr = &[MonoAD2FR::Sqrt];
    let ops_rf = &[MonoAD2RF::Sqrt];
    let x: f64 = 2.5;

    let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
    let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
    let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

    let expected = -1.0 / (4.0 * x * x.sqrt());
    assert!(approx_eq(h_rr, expected, EXACT_TOL));
    assert!(approx_eq(h_fr, expected, EXACT_TOL));
    assert!(approx_eq(h_rf, expected, EXACT_TOL));

    assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
    assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
}

#[test]
fn test_consistency_all_methods_abs() {
    let ops_rr = &[MonoAD2RR::Abs];
    let ops_fr = &[MonoAD2FR::Abs];
    let ops_rf = &[MonoAD2RF::Abs];

    for x in [-2.5, 0.0, 2.5] {
        let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
        let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
        let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

        assert!(approx_eq(h_rr, 0.0, EXACT_TOL));
        assert!(approx_eq(h_fr, 0.0, EXACT_TOL));
        assert!(approx_eq(h_rf, 0.0, EXACT_TOL));
        assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
        assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
    }
}

#[test]
fn test_consistency_all_methods_composed() {
    let ops_rr = &[MonoAD2RR::Sin, MonoAD2RR::Exp];
    let ops_fr = &[MonoAD2FR::Sin, MonoAD2FR::Exp];
    let ops_rf = &[MonoAD2RF::Sin, MonoAD2RF::Exp];
    let x = 0.5;

    let h_rr = MonoAD2RR::compute_hessian(ops_rr, x);
    let h_fr = MonoAD2FR::compute_hessian(ops_fr, x);
    let h_rf = MonoAD2RF::compute_hessian(ops_rf, x);

    let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();
    assert!(approx_eq(h_rr, expected, COMPARISON_TOL));
    assert!(approx_eq(h_fr, expected, COMPARISON_TOL));
    assert!(approx_eq(h_rf, expected, COMPARISON_TOL));

    assert!(approx_eq(h_rr, h_fr, COMPARISON_TOL));
    assert!(approx_eq(h_rr, h_rf, COMPARISON_TOL));
}
