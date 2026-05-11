//! Comprehensive tests for all features and edge cases.
//! This module ensures very high code coverage.

use crate::error::AutodiffError;
use crate::mono::types::Dual;
use crate::test_utils::approx_eq_eps as approx_eq;
use crate::{mono_ops, mono_ops_fr, mono_ops_rf, mono_ops_rr, multi_ops};
use crate::{
    GraphBuilder, MonoAD, MonoAD2FR, MonoAD2RF, MonoAD2RR, MultiAD, MultiAD2FR, MultiAD2RF,
    MultiAD2RR,
};

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_display() {
    let err = AutodiffError::arity("Sin", 1, 2);
    let msg = format!("{}", err);
    assert!(msg.contains("Sin"));
    assert!(msg.contains("expected 1"));
    assert!(msg.contains("got 2"));

    let err = AutodiffError::EmptyGraph;
    let msg = format!("{}", err);
    assert!(msg.contains("empty"));

    let err = AutodiffError::IndexOutOfBounds {
        index: 5,
        max_index: 3,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("5"));
    assert!(msg.contains("3"));

    let err = AutodiffError::InvalidGraph {
        reason: "missing operand",
    };
    let msg = format!("{}", err);
    assert!(msg.contains("invalid"));
    assert!(msg.contains("missing operand"));

    let err = AutodiffError::domain("Ln", "input must be positive");
    let msg = format!("{}", err);
    assert!(msg.contains("Domain error"));
    assert!(msg.contains("Ln"));
}

#[test]
fn test_error_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AutodiffError>();
}

#[test]
fn test_check_arity_ok() {
    assert!(AutodiffError::check_arity("Test", 2, 2).is_ok());
}

#[test]
fn test_check_arity_err() {
    let result = AutodiffError::check_arity("Test", 2, 3);
    assert!(result.is_err());
    match result {
        Err(AutodiffError::ArityError {
            operation,
            expected,
            actual,
        }) => {
            assert_eq!(operation, "Test");
            assert_eq!(expected, 2);
            assert_eq!(actual, 3);
        }
        _ => panic!("Expected ArityError"),
    }
}

// ============================================================================
// MonoAD Unary Operation Tests
// ============================================================================

#[test]
fn test_mono_neg_forward() {
    let ops = &[MonoAD::Neg];
    let result = MonoAD::compute(ops, 5.0);
    assert!(approx_eq(result, -5.0, 1e-10));

    let result = MonoAD::compute(ops, -3.0);
    assert!(approx_eq(result, 3.0, 1e-10));
}

#[test]
fn test_mono_neg_gradient() {
    let ops = &[MonoAD::Neg];
    let (_value, backprop) = MonoAD::compute_grad(ops, 5.0);
    assert!(approx_eq(backprop(1.0), -1.0, 1e-10));
    assert!(approx_eq(backprop(2.0), -2.0, 1e-10));
}

#[test]
fn test_mono_neg_chain() {
    let ops = &[MonoAD::Sin, MonoAD::Neg];
    let x: f64 = 1.0;
    let (value, backprop) = MonoAD::compute_grad(ops, x);

    assert!(approx_eq(value, -x.sin(), 1e-10));
    assert!(approx_eq(backprop(1.0), -x.cos(), 1e-10));
}

#[test]
fn test_mono_neg_hessian() {
    let ops = &[MonoAD::Neg];
    let hessian = MonoAD::compute_hessian(ops, 5.0);
    assert!(approx_eq(hessian, 0.0, 1e-10));
}

#[test]
fn test_mono_tan_forward_and_grad() {
    let x: f64 = 0.5;
    let ops = &[MonoAD::Tan];
    let (value, backprop) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(value, x.tan(), 1e-10));
    assert!(approx_eq(backprop(1.0), 1.0 / x.cos().powi(2), 1e-10));
}

#[test]
fn test_mono_ln_forward_and_grad() {
    let x: f64 = 2.0;
    let ops = &[MonoAD::Ln];
    let (value, backprop) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(value, x.ln(), 1e-10));
    assert!(approx_eq(backprop(1.0), 1.0 / x, 1e-10));
}

#[test]
fn test_mono_sqrt_forward_and_grad() {
    let x: f64 = 9.0;
    let ops = &[MonoAD::Sqrt];
    let (value, backprop) = MonoAD::compute_grad(ops, x);
    assert!(approx_eq(value, 3.0, 1e-10));
    assert!(approx_eq(backprop(1.0), 1.0 / 6.0, 1e-10));
}

#[test]
fn test_mono_abs_forward_and_grad() {
    let ops = &[MonoAD::Abs];
    let (value_pos, backprop_pos) = MonoAD::compute_grad(ops, 2.0);
    assert!(approx_eq(value_pos, 2.0, 1e-10));
    assert!(approx_eq(backprop_pos(1.0), 1.0, 1e-10));

    let (value_zero, backprop_zero) = MonoAD::compute_grad(ops, 0.0);
    assert!(approx_eq(value_zero, 0.0, 1e-10));
    assert!(approx_eq(backprop_zero(1.0), 0.0, 1e-10));
}

#[test]
fn test_mono_checked_domain_errors() {
    assert!(MonoAD::compute_checked(&[MonoAD::Ln], 2.0).is_ok());
    assert!(matches!(
        MonoAD::compute_checked(&[MonoAD::Ln], 0.0),
        Err(AutodiffError::DomainError {
            operation: "Ln",
            ..
        })
    ));
    assert!(MonoAD::compute_checked(&[MonoAD::Sqrt], 0.0).is_ok());
    assert!(matches!(
        MonoAD::compute_checked(&[MonoAD::Sqrt], -1.0),
        Err(AutodiffError::DomainError {
            operation: "Sqrt",
            ..
        })
    ));
}

#[test]
fn test_mono_checked_grad_and_hessian() {
    let (value, grad_fn) = MonoAD::compute_grad_checked(&[MonoAD::Ln], 2.0).unwrap();
    assert!(approx_eq(value, 2.0_f64.ln(), 1e-10));
    assert!(approx_eq(grad_fn(1.0), 0.5, 1e-10));

    assert!(MonoAD::compute_grad_checked(&[MonoAD::Sqrt], -1.0).is_err());
    assert!(MonoAD::compute_hessian_checked(&[MonoAD::Ln], 2.0).is_ok());
    assert!(MonoAD::compute_hessian_checked(&[MonoAD::Sqrt], 1.0).is_ok());
    assert!(MonoAD::compute_hessian_checked(&[MonoAD::Sqrt], 0.0).is_err());
}

#[test]
fn test_mono_exact_checked_domain_errors() {
    assert!(MonoAD2RR::compute_checked(&[MonoAD2RR::Ln], 2.0).is_ok());
    assert!(MonoAD2RR::compute_grad_checked(&[MonoAD2RR::Sqrt], 4.0).is_ok());
    assert!(MonoAD2RR::compute_hessian_checked(&[MonoAD2RR::Sqrt], 4.0).is_ok());
    assert!(MonoAD2FR::compute_checked(&[MonoAD2FR::Ln], 2.0).is_ok());
    assert!(MonoAD2RF::compute_checked(&[MonoAD2RF::Sqrt], 0.0).is_ok());

    assert!(matches!(
        MonoAD2RR::compute_checked(&[MonoAD2RR::Ln], 0.0),
        Err(AutodiffError::DomainError {
            operation: "Ln",
            ..
        })
    ));
    assert!(matches!(
        MonoAD2FR::compute_grad_checked(&[MonoAD2FR::Sqrt], -1.0),
        Err(AutodiffError::DomainError {
            operation: "Sqrt",
            ..
        })
    ));
    assert!(matches!(
        MonoAD2RF::compute_hessian_checked(&[MonoAD2RF::Ln], -1.0),
        Err(AutodiffError::DomainError {
            operation: "Ln",
            ..
        })
    ));
}

#[test]
fn test_mono_ops_macro_extended() {
    let ops = mono_ops![tan, neg, ln, sqrt, abs];
    assert_eq!(ops.len(), 5);
}

// ============================================================================
// MultiAD Missing Operations Tests (Div, Sub, Tan, Ln)
// ============================================================================

#[test]
fn test_multi_div_forward() {
    let exprs = &multi_ops![(inp, 0), (inp, 1), (div, 0, 1)];
    let inputs = &[10.0, 2.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 5.0, 1e-10));
}

#[test]
fn test_multi_div_gradient() {
    let exprs = &multi_ops![(inp, 0), (inp, 1), (div, 0, 1)];
    let x = 10.0;
    let y = 2.0;
    let inputs = &[x, y];

    let (_value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    assert!(approx_eq(grads[0], 1.0 / y, 1e-10));
    assert!(approx_eq(grads[1], -x / (y * y), 1e-10));
}

#[test]
fn test_multi_sub_forward() {
    let exprs = &multi_ops![(inp, 0), (inp, 1), (sub, 0, 1)];
    let inputs = &[5.0, 3.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 2.0, 1e-10));
}

#[test]
fn test_multi_sub_gradient() {
    let exprs = &multi_ops![(inp, 0), (inp, 1), (sub, 0, 1)];
    let inputs = &[5.0, 3.0];

    let (_value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    assert!(approx_eq(grads[0], 1.0, 1e-10));
    assert!(approx_eq(grads[1], -1.0, 1e-10));
}

#[test]
fn test_multi_tan_forward() {
    let exprs = &multi_ops![(tan, 0)];
    let x: f64 = 0.5;
    let inputs = &[x];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, x.tan(), 1e-10));
}

#[test]
fn test_multi_tan_gradient() {
    let exprs = &multi_ops![(tan, 0)];
    let x: f64 = 0.5;
    let inputs = &[x];

    let (_value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    let expected_grad = 1.0 / x.cos().powi(2);
    assert!(approx_eq(grads[0], expected_grad, 1e-10));
}

#[test]
fn test_multi_checked_domain_errors() {
    let ln_exprs = &multi_ops![(ln, 0)];
    let ln_error = MultiAD::compute_checked(ln_exprs, &[0.0]).unwrap_err();
    assert_eq!(
        ln_error,
        AutodiffError::DomainError {
            operation: "Ln",
            reason: "input must be positive",
        }
    );

    let div_exprs = &multi_ops![(inp, 0), (inp, 1), (div, 0, 1)];
    let div_error = MultiAD::compute_checked(div_exprs, &[1.0, 0.0]).unwrap_err();
    assert_eq!(
        div_error,
        AutodiffError::DomainError {
            operation: "Div",
            reason: "denominator must be non-zero",
        }
    );
}

#[test]
fn test_multi_checked_gradient_errors() {
    let exprs = &multi_ops![(inp, 0), (inp, 1), (pow, 0, 1)];
    let result = MultiAD::compute_grad_checked(exprs, &[-1.0, 2.0]);
    assert!(matches!(
        result,
        Err(AutodiffError::DomainError {
            operation: "Pow",
            reason: "base must be positive in checked mode",
        })
    ));
}

#[test]
fn test_multi_ln_forward() {
    let exprs = &multi_ops![(ln, 0)];
    let x: f64 = 2.0;
    let inputs = &[x];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, x.ln(), 1e-10));
}

#[test]
fn test_multi_ln_gradient() {
    let exprs = &multi_ops![(ln, 0)];
    let x: f64 = 2.0;
    let inputs = &[x];

    let (_value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    assert!(approx_eq(grads[0], 1.0 / x, 1e-10));
}

// ============================================================================
// MultiAD Edge Cases
// ============================================================================

#[test]
fn test_multi_empty_graph() {
    let exprs: &[(MultiAD, Vec<usize>)] = &[];
    let inputs: &[f64] = &[];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_multi_single_input() {
    // Inp operation alone returns the input value at that index.
    let exprs = &multi_ops![(inp, 0)];
    let inputs = &[5.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert_eq!(result, 5.0);
}

#[test]
fn test_multi_input_marker_selects_requested_input() {
    let exprs = &multi_ops![(inp, 0)];
    let inputs = &[5.0, 7.0];

    let (value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    assert_eq!(value, 5.0);
    assert_eq!(grads, vec![1.0, 0.0]);
}

#[test]
fn test_multi_complex_chain() {
    let exprs = &multi_ops![
        (inp, 0),
        (inp, 1),
        (sub, 0, 1),
        (sin, 2),
        (div, 0, 1),
        (ln, 4),
        (mul, 3, 5),
    ];

    let x: f64 = 3.0;
    let y: f64 = 2.0;
    let inputs = &[x, y];

    let (value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    let expected_value: f64 = (x - y).sin() * (x / y).ln();
    assert!(approx_eq(value, expected_value, 1e-10));

    assert!(grads[0].is_finite());
    assert!(grads[1].is_finite());
}

// ============================================================================
// GraphBuilder Comprehensive Tests
// ============================================================================

#[test]
fn test_builder_all_operations() {
    let graph = GraphBuilder::new(3)
        .input(0)
        .sin(0)
        .cos(0)
        .tan(0)
        .exp(0)
        .ln(0)
        .sqrt(0)
        .abs(0)
        .add(0, 1)
        .sub(0, 1)
        .mul(0, 1)
        .div(0, 1)
        .pow(0, 1)
        .build();

    assert!(!graph.is_empty());
    assert_eq!(graph.len(), 13);
}

#[test]
fn test_builder_chained_complex() {
    let graph = GraphBuilder::new(2)
        .sin(0)
        .cos(1)
        .tan(0)
        .exp(1)
        .ln(0)
        .sqrt(1)
        .abs(0)
        .sub(0, 1)
        .div(2, 3)
        .build();

    let inputs = &[0.5, 0.8];
    let result = MultiAD::compute(&graph, inputs);
    assert!(result.is_ok());
}

// ============================================================================
// Second-Order Methods Tests
// ============================================================================

#[test]
fn test_mono_ad2rr_compute() {
    let ops = &[MonoAD2RR::Sin, MonoAD2RR::Exp];
    let x: f64 = 0.5;
    let value = MonoAD2RR::compute(ops, x);
    assert!(approx_eq(value, x.sin().exp(), 1e-10));
}

#[test]
fn test_mono_ad2rr_tan_hessian() {
    let ops = &[MonoAD2RR::Tan];
    let x: f64 = 0.4;
    let sec_sq = 1.0 / x.cos().powi(2);
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, 2.0 * sec_sq * x.tan(), 1e-10));
}

#[test]
fn test_mono_ad2rr_ln_hessian() {
    let ops = &[MonoAD2RR::Ln];
    let x: f64 = 1.7;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, -1.0 / x.powi(2), 1e-10));
}

#[test]
fn test_mono_ad2rr_sqrt_hessian() {
    let ops = &[MonoAD2RR::Sqrt];
    let x: f64 = 2.5;
    let hessian = MonoAD2RR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, -1.0 / (4.0 * x * x.sqrt()), 1e-10));
}

#[test]
fn test_mono_ad2rr_abs_hessian() {
    let ops = &[MonoAD2RR::Abs];
    for x in [-2.5, 0.0, 2.5] {
        let hessian = MonoAD2RR::compute_hessian(ops, x);
        assert!(approx_eq(hessian, 0.0, 1e-10));
    }
}

#[test]
fn test_mono_ad2rr_compute_grad() {
    let ops = &[MonoAD2RR::Sin];
    let x: f64 = 0.5;
    let (value, backprop) = MonoAD2RR::compute_grad(ops, x);
    assert!(approx_eq(value, x.sin(), 1e-10));
    assert!(approx_eq(backprop(1.0), x.cos(), 1e-10));
}

#[test]
fn test_mono_ad2fr_compute() {
    let ops = &[MonoAD2FR::Sin, MonoAD2FR::Exp];
    let x: f64 = 0.5;
    let value = MonoAD2FR::compute(ops, x);
    assert!(approx_eq(value, x.sin().exp(), 1e-10));
}

#[test]
fn test_mono_ad2fr_tan_hessian() {
    let ops = &[MonoAD2FR::Tan];
    let x: f64 = 0.4;
    let sec_sq = 1.0 / x.cos().powi(2);
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, 2.0 * sec_sq * x.tan(), 1e-10));
}

#[test]
fn test_mono_ad2fr_ln_hessian() {
    let ops = &[MonoAD2FR::Ln];
    let x: f64 = 1.7;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, -1.0 / x.powi(2), 1e-10));
}

#[test]
fn test_mono_ad2fr_sqrt_hessian() {
    let ops = &[MonoAD2FR::Sqrt];
    let x: f64 = 2.5;
    let hessian = MonoAD2FR::compute_hessian(ops, x);
    assert!(approx_eq(hessian, -1.0 / (4.0 * x * x.sqrt()), 1e-10));
}

#[test]
fn test_mono_ad2fr_abs_hessian() {
    let ops = &[MonoAD2FR::Abs];
    for x in [-2.5, 0.0, 2.5] {
        let hessian = MonoAD2FR::compute_hessian(ops, x);
        assert!(approx_eq(hessian, 0.0, 1e-10));
    }
}

#[test]
fn test_mono_ad2fr_compute_grad() {
    let ops = &[MonoAD2FR::Sin];
    let x: f64 = 0.5;
    let (value, backprop) = MonoAD2FR::compute_grad(ops, x);
    assert!(approx_eq(value, x.sin(), 1e-10));
    assert!(approx_eq(backprop(1.0), x.cos(), 1e-10));
}

#[test]
fn test_mono_ad2rf_compute() {
    let ops = &[MonoAD2RF::Sin, MonoAD2RF::Exp];
    let x: f64 = 0.5;
    let value = MonoAD2RF::compute(ops, x);
    assert!(approx_eq(value, x.sin().exp(), 1e-10));
}

#[test]
fn test_mono_ad2rf_tan_hessian() {
    let ops = &[MonoAD2RF::Tan];
    let x: f64 = 0.4;
    let sec_sq = 1.0 / x.cos().powi(2);
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, 2.0 * sec_sq * x.tan(), 1e-10));
}

#[test]
fn test_mono_ad2rf_ln_hessian() {
    let ops = &[MonoAD2RF::Ln];
    let x: f64 = 1.7;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, -1.0 / x.powi(2), 1e-10));
}

#[test]
fn test_mono_ad2rf_sqrt_hessian() {
    let ops = &[MonoAD2RF::Sqrt];
    let x: f64 = 2.5;
    let hessian = MonoAD2RF::compute_hessian(ops, x);
    assert!(approx_eq(hessian, -1.0 / (4.0 * x * x.sqrt()), 1e-10));
}

#[test]
fn test_mono_ad2rf_abs_hessian() {
    let ops = &[MonoAD2RF::Abs];
    for x in [-2.5, 0.0, 2.5] {
        let hessian = MonoAD2RF::compute_hessian(ops, x);
        assert!(approx_eq(hessian, 0.0, 1e-10));
    }
}

#[test]
fn test_mono_ad2rf_compute_grad() {
    let ops = &[MonoAD2RF::Sin];
    let x: f64 = 0.5;
    let (value, backprop) = MonoAD2RF::compute_grad(ops, x);
    assert!(approx_eq(value, x.sin(), 1e-10));
    assert!(approx_eq(backprop(1.0), x.cos(), 1e-10));
}

// ============================================================================
// Dual Type Tests
// ============================================================================

#[test]
fn test_dual_variable() {
    let d = Dual::variable(3.0);
    assert!(approx_eq(d.val, 3.0, 1e-10));
    assert!(approx_eq(d.tan, 1.0, 1e-10));
}

#[test]
fn test_dual_constant() {
    let d = Dual::constant(3.0);
    assert!(approx_eq(d.val, 3.0, 1e-10));
    assert!(approx_eq(d.tan, 0.0, 1e-10));
}

// ============================================================================
// Arity Error Tests
// ============================================================================

#[test]
fn test_multi_arity_errors() {
    let exprs = &[(MultiAD::Sin, vec![0, 1])];
    let inputs = &[1.0, 2.0];
    let result = MultiAD::compute(exprs, inputs);
    assert!(result.is_err());

    let exprs = &[(MultiAD::Add, vec![0])];
    let result = MultiAD::compute(exprs, inputs);
    assert!(result.is_err());
}

// ============================================================================
// Macros Tests
// ============================================================================

#[test]
fn test_mono_ops_rr_macro() {
    let ops = mono_ops_rr![sin, cos, tan, exp, neg, ln, sqrt, abs];
    assert_eq!(ops.len(), 8);
}

#[test]
fn test_mono_ops_fr_macro() {
    let ops = mono_ops_fr![sin, cos, tan, exp, neg, ln, sqrt, abs];
    assert_eq!(ops.len(), 8);
}

#[test]
fn test_mono_ops_rf_macro() {
    let ops = mono_ops_rf![sin, cos, tan, exp, neg, ln, sqrt, abs];
    assert_eq!(ops.len(), 8);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_mono_empty_hessian() {
    let ops: &[MonoAD] = &[];
    let hessian = MonoAD::compute_hessian(ops, 5.0);
    assert_eq!(hessian, 0.0);
}

#[test]
fn test_mono2rr_empty_hessian() {
    let ops: &[MonoAD2RR] = &[];
    let hessian = MonoAD2RR::compute_hessian(ops, 5.0);
    assert_eq!(hessian, 0.0);
}

#[test]
fn test_mono2fr_empty_hessian() {
    let ops: &[MonoAD2FR] = &[];
    let hessian = MonoAD2FR::compute_hessian(ops, 5.0);
    assert_eq!(hessian, 0.0);
}

#[test]
fn test_mono2rf_empty_hessian() {
    let ops: &[MonoAD2RF] = &[];
    let hessian = MonoAD2RF::compute_hessian(ops, 5.0);
    assert_eq!(hessian, 0.0);
}

#[test]
fn test_multi_hessian_empty() {
    let exprs: &[(MultiAD, Vec<usize>)] = &[];
    let inputs = &[1.0, 2.0];
    let hessian = MultiAD::compute_hessian(exprs, inputs).unwrap();
    assert_eq!(hessian.len(), 2);
    assert_eq!(hessian[0][0], 0.0);
}

#[test]
fn test_multi_hessian_row() {
    let exprs = &multi_ops![(inp, 0), (inp, 1), (mul, 0, 0), (mul, 1, 1), (add, 2, 3)];
    let inputs = &[2.0, 3.0];

    let row0 = MultiAD::compute_hessian_row(exprs, inputs, 0).unwrap();
    assert!(approx_eq(row0[0], 2.0, 1e-6));
    assert!(approx_eq(row0[1], 0.0, 1e-6));

    let row1 = MultiAD::compute_hessian_row(exprs, inputs, 1).unwrap();
    assert!(approx_eq(row1[0], 0.0, 1e-6));
    assert!(approx_eq(row1[1], 2.0, 1e-6));
}

#[test]
fn test_deep_composition() {
    let ops = mono_ops![sin, sin, sin, sin, sin];
    let x: f64 = 0.5;

    let (value, backprop) = MonoAD::compute_grad(&ops, x);
    let expected = x.sin().sin().sin().sin().sin();
    assert!(approx_eq(value, expected, 1e-10));

    let grad = backprop(1.0);
    assert!(grad.is_finite());

    let hessian: f64 = MonoAD::compute_hessian(&ops, x);
    assert!(hessian.is_finite());
}

#[test]
fn test_abs_at_zero() {
    // Test abs at x=0 using the explicit subgradient convention sign(0) = 0.
    let exprs = &multi_ops![(abs, 0)];
    let inputs = &[0.0];

    let (_value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);
    assert_eq!(grads[0], 0.0);
}

#[test]
fn test_multi_invalid_index_returns_error() {
    let exprs = &[(MultiAD::Sin, vec![10])];
    let inputs = &[0.5];

    let compute_result = MultiAD::compute(exprs, inputs);
    assert!(matches!(
        compute_result,
        Err(AutodiffError::IndexOutOfBounds { index: 10, .. })
    ));

    let grad_result = MultiAD::compute_grad(exprs, inputs);
    assert!(matches!(
        grad_result,
        Err(AutodiffError::IndexOutOfBounds { index: 10, .. })
    ));
}

#[test]
fn test_multi_empty_compute_grad_zero_inputs_is_safe() {
    let exprs: &[(MultiAD, Vec<usize>)] = &[];
    let inputs: &[f64] = &[];

    let (value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    assert_eq!(value, 0.0);
    assert!(grads.is_empty());
}

#[test]
fn test_multi_hessian_row_invalid_index_returns_error() {
    let exprs = &multi_ops![(inp, 0), (mul, 0, 0)];
    let result = MultiAD::compute_hessian_row(exprs, &[2.0], 1);

    assert!(matches!(
        result,
        Err(AutodiffError::IndexOutOfBounds { index: 1, .. })
    ));
}

#[test]
fn test_multi_hessian_zero_inputs_validates_graph() {
    let exprs = &[(MultiAD::Sin, vec![0])];
    let result = MultiAD::compute_hessian(exprs, &[]);

    assert!(matches!(
        result,
        Err(AutodiffError::IndexOutOfBounds { index: 0, .. })
    ));
}

#[test]
fn test_multi_exact_hessian_invalid_input_returns_error() {
    let rr_result = MultiAD2RR::compute_hessian(&[MultiAD2RR::Inp(1)], &[1.0]);
    let fr_result = MultiAD2FR::compute_hessian(&[MultiAD2FR::Inp(1)], &[1.0]);
    let rf_result = MultiAD2RF::compute_hessian(&[MultiAD2RF::Inp(1)], &[1.0]);

    assert!(matches!(
        rr_result,
        Err(AutodiffError::IndexOutOfBounds { index: 1, .. })
    ));
    assert!(matches!(
        fr_result,
        Err(AutodiffError::IndexOutOfBounds { index: 1, .. })
    ));
    assert!(matches!(
        rf_result,
        Err(AutodiffError::IndexOutOfBounds { index: 1, .. })
    ));
}

#[test]
fn test_multi_exact_hessian_zero_inputs_validates_graph() {
    let rr_result = MultiAD2RR::compute_hessian(&[MultiAD2RR::Inp(0)], &[]);
    let fr_result = MultiAD2FR::compute_hessian(&[MultiAD2FR::Inp(0)], &[]);
    let rf_result = MultiAD2RF::compute_hessian(&[MultiAD2RF::Inp(0)], &[]);

    assert!(matches!(
        rr_result,
        Err(AutodiffError::IndexOutOfBounds { index: 0, .. })
    ));
    assert!(matches!(
        fr_result,
        Err(AutodiffError::IndexOutOfBounds { index: 0, .. })
    ));
    assert!(matches!(
        rf_result,
        Err(AutodiffError::IndexOutOfBounds { index: 0, .. })
    ));
}

#[test]
fn test_multi_exact_hessian_malformed_rpn_returns_error() {
    let rr_result = MultiAD2RR::compute_hessian(&[MultiAD2RR::Sin], &[1.0]);
    let fr_result = MultiAD2FR::compute_hessian(&[MultiAD2FR::Sin], &[1.0]);
    let rf_result = MultiAD2RF::compute_hessian(&[MultiAD2RF::Sin], &[1.0]);

    assert!(matches!(rr_result, Err(AutodiffError::InvalidGraph { .. })));
    assert!(matches!(fr_result, Err(AutodiffError::InvalidGraph { .. })));
    assert!(matches!(rf_result, Err(AutodiffError::InvalidGraph { .. })));
}

#[test]
fn test_builder_input_marker_does_not_shift_indices() {
    let graph = GraphBuilder::new(1).input(0).sin(0).build();
    let result = MultiAD::compute(&graph, &[0.5]).unwrap();

    assert!(approx_eq(result, 0.5_f64.sin(), 1e-10));
}

#[test]
fn test_mixed_operations() {
    let exprs = &multi_ops![
        (inp, 0),
        (inp, 1),
        (add, 0, 1),
        (sub, 0, 1),
        (mul, 2, 3),
        (div, 0, 1),
        (pow, 0, 1),
    ];

    let x: f64 = 2.0;
    let y: f64 = 1.5;
    let inputs = &[x, y];

    let (value, backprop) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop(1.0);

    let _value: f64 = value;
    assert!(value.is_finite());
    assert!(grads[0].is_finite());
    assert!(grads[1].is_finite());
}
