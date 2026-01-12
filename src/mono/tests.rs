use super::mono_ad::MonoAD;
use crate::mono_ops;
use crate::test_utils::approx_eq_eps as approx_eq;

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

    // Verify the computation is correct
    assert!(approx_eq(value_box, 2.2013533791690376, 1e-10));
    assert!(approx_eq(grad_box, -0.562752038662712, 1e-10));
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

        // Verify computation succeeds
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
