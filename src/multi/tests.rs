use super::*;
use crate::multi_ops;

// Helper function for floating point comparison
fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

#[test]
fn test_forward_single_op() {
    // Test sin(x) where x = 0.5
    let exprs = &[(MultiAD::Sin, vec![0])];
    let inputs = &[0.5];
    let result = MultiAD::compute(exprs, inputs);
    assert!(approx_eq(result, 0.5_f64.sin(), 1e-10));

    let exprs = &multi_ops![(sin, 0)];
    let inputs = &[0.5];
    let result = MultiAD::compute(exprs, inputs);
    assert!(approx_eq(result, 0.5_f64.sin(), 1e-10));
}

#[test]
fn test_forward_add() {
    // Test x + y where x = 2.0, y = 3.0
    let exprs = &multi_ops![(inp, 0), (inp, 1), (add, 0, 1),];
    let inputs = &[2.0, 3.0];
    let result = MultiAD::compute(exprs, inputs);
    assert!(approx_eq(result, 5.0, 1e-10));
}

#[test]
fn test_forward_mul() {
    // Test x * y where x = 2.0, y = 3.0
    let exprs = &multi_ops![(inp, 0), (inp, 1), (mul, 0, 1),];
    let inputs = &[2.0, 3.0];
    let result = MultiAD::compute(exprs, inputs);
    assert!(approx_eq(result, 6.0, 1e-10));
}

#[test]
fn test_backward_sin() {
    // Test gradient of sin(x): d(sin(x))/dx = cos(x)
    let exprs = &multi_ops![(sin, 0),];
    let inputs = &[0.5];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);
    assert_eq!(grads.len(), 1);
    assert!(approx_eq(grads[0], 0.5_f64.cos(), 1e-10));
}

#[test]
fn test_backward_exp() {
    // Test gradient of exp(x): d(exp(x))/dx = exp(x)
    let exprs = &[(MultiAD::Exp, vec![0])];
    let inputs = &[2.0];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);
    assert!(approx_eq(grads[0], 2.0_f64.exp(), 1e-10));
}

#[test]
fn test_backward_add() {
    // Test gradient of x + y: both partials are 1
    let exprs = &[
        (MultiAD::Inp, vec![0]),
        (MultiAD::Inp, vec![1]),
        (MultiAD::Add, vec![0, 1]),
    ];
    let inputs = &[2.0, 3.0];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);
    assert_eq!(grads.len(), 2);
    assert!(approx_eq(grads[0], 1.0, 1e-10));
    assert!(approx_eq(grads[1], 1.0, 1e-10));
}

#[test]
fn test_backward_mul() {
    // Test gradient of x * y: dx = y, dy = x
    let exprs = &[
        (MultiAD::Inp, vec![0]),
        (MultiAD::Inp, vec![1]),
        (MultiAD::Mul, vec![0, 1]),
    ];
    let inputs = &[3.0, 4.0];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);
    assert_eq!(grads.len(), 2);
    assert!(approx_eq(grads[0], 4.0, 1e-10)); // ∂(xy)/∂x = y
    assert!(approx_eq(grads[1], 3.0, 1e-10)); // ∂(xy)/∂y = x
}

#[test]
fn test_complex_function() {
    // Test f(x₁, x₂) = sin(x₁) * (x₁ + x₂)
    // This matches the analytical function f() and f_grad()
    let exprs = &multi_ops![
        (inp, 0),    // x₁ at index 0
        (inp, 1),    // x₂ at index 1
        (add, 0, 1), // x₁ + x₂ at index 2
        (sin, 0),    // sin(x₁) at index 3
        (mul, 2, 3), // sin(x₁) * (x₁ + x₂) at index 4
    ];

    let x1 = 0.6;
    let x2 = 1.4;
    let inputs = &[x1, x2];

    // Compute forward and backward
    let (value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);

    // Compare with analytical solution
    let f1 = F1(x1, x2);
    let expected_value = f1.f();
    let expected_grads = f1.grad();

    assert!(
        approx_eq(value, expected_value, 1e-10),
        "value mismatch: got {}, expected {}",
        value,
        expected_value
    );

    assert_eq!(grads.len(), 2);
    assert!(
        approx_eq(grads[0], expected_grads[0], 1e-10),
        "grad[0] mismatch: got {}, expected {}",
        grads[0],
        expected_grads[0]
    );
    assert!(
        approx_eq(grads[1], expected_grads[1], 1e-10),
        "grad[1] mismatch: got {}, expected {}",
        grads[1],
        expected_grads[1]
    );
}

#[test]
fn test_chain_rule() {
    // Test f(x) = exp(sin(x))
    // Chain rule: d(exp(sin(x)))/dx = exp(sin(x)) * cos(x)
    let exprs = &multi_ops![
        (inp, 0), // x at index 0
        (sin, 0), // sin(x) at index 1
        (exp, 1), // exp(sin(x)) at index 2
    ];

    let x: f64 = 0.5;
    let inputs = &[x];

    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);

    let expected_grad = x.sin().exp() * x.cos();

    assert!(
        approx_eq(grads[0], expected_grad, 1e-10),
        "gradient mismatch: got {}, expected {}",
        grads[0],
        expected_grad
    );
}

#[test]
fn test_cos_backward() {
    // Test gradient of cos(x): d(cos(x))/dx = -sin(x)
    let exprs = &[(MultiAD::Cos, vec![0])];
    let inputs = &[0.5];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
    let grads = backprop_fn(1.0);
    assert!(approx_eq(grads[0], -0.5_f64.sin(), 1e-10));
}

#[test]
fn test_compute_grad_arc_consistency() {
    // Test that compute_grad_arc produces the same results as compute_grad
    let exprs = &multi_ops![(inp, 0), (inp, 1), (add, 0, 1), (sin, 0), (mul, 2, 3),];

    let inputs = &[0.6, 1.4];

    // Test with cotangent = 1.0
    let (_value_box, backprop_box) = MultiAD::compute_grad(exprs, inputs);
    let grads_box = backprop_box(1.0);

    let (_value_arc, backprop_arc) = MultiAD::compute_grad_arc(exprs, inputs);
    let grads_arc = backprop_arc(1.0);

    assert_eq!(grads_box.len(), grads_arc.len());
    for i in 0..grads_box.len() {
        assert!(
            approx_eq(grads_box[i], grads_arc[i], 1e-10),
            "grad[{}] mismatch: Box={}, Arc={}",
            i,
            grads_box[i],
            grads_arc[i]
        );
    }
}

#[test]
fn test_different_cotangents() {
    // Test that different cotangent values produce correct results
    let exprs = &multi_ops![(inp, 0), (inp, 1), (add, 0, 1), (sin, 0), (mul, 2, 3),];

    let inputs = &[0.6, 1.4];

    // Test with different cotangent values
    for cotangent in [0.5, 1.0, 2.0, 10.0] {
        let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
        let grads = backprop_fn(cotangent);

        // Gradients should scale linearly with cotangent
        let (_value_base, backprop_fn_base) = MultiAD::compute_grad(exprs, inputs);
        let grads_base = backprop_fn_base(1.0);

        for i in 0..grads.len() {
            assert!(
                approx_eq(grads[i], grads_base[i] * cotangent, 1e-10),
                "cotangent {}: grads[{}] = {}, expected {}",
                cotangent,
                i,
                grads[i],
                grads_base[i] * cotangent
            );
        }
    }
}

#[test]
fn test_f() {
    F1(0.5, 1.0).demonstrate(true);
    F2(0.5, 1.0).demonstrate(true);
    F3(0.5, 1.0).demonstrate(true);
}
