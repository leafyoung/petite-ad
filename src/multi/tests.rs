use super::f1::F1;
use super::f2::F2;
use super::f3::F3;
use super::*;
use crate::multi_ops;
use crate::test_utils::approx_eq_eps as approx_eq;

#[test]
fn test_forward_single_op() {
    // Test sin(x) where x = 0.5
    let exprs = &[(MultiAD::Sin, vec![0])];
    let inputs = &[0.5];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 0.5_f64.sin(), 1e-10));

    let exprs = &multi_ops![(sin, 0)];
    let inputs = &[0.5];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 0.5_f64.sin(), 1e-10));
}

#[test]
fn test_forward_add() {
    // Test x + y where x = 2.0, y = 3.0
    let exprs = &multi_ops![(inp, 0), (inp, 1), (add, 0, 1),];
    let inputs = &[2.0, 3.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 5.0, 1e-10));
}

#[test]
fn test_forward_mul() {
    // Test x * y where x = 2.0, y = 3.0
    let exprs = &multi_ops![(inp, 0), (inp, 1), (mul, 0, 1),];
    let inputs = &[2.0, 3.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 6.0, 1e-10));
}

#[test]
fn test_backward_sin() {
    // Test gradient of sin(x): d(sin(x))/dx = cos(x)
    let exprs = &multi_ops![(sin, 0),];
    let inputs = &[0.5];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);
    assert_eq!(grads.len(), 1);
    assert!(approx_eq(grads[0], 0.5_f64.cos(), 1e-10));
}

#[test]
fn test_backward_exp() {
    // Test gradient of exp(x): d(exp(x))/dx = exp(x)
    let exprs = &[(MultiAD::Exp, vec![0])];
    let inputs = &[2.0];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
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
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
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
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
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
    let (value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);

    // Compare with analytical solution
    let f1 = F1(x1, x2);
    let expected_value = f1.expected_value();
    let expected_grads = f1.expected_gradients();

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

    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
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
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);
    assert!(approx_eq(grads[0], -0.5_f64.sin(), 1e-10));
}

#[test]
fn test_compute_grad_arc_consistency() {
    // Test that compute_grad produces consistent results
    let exprs = &multi_ops![(inp, 0), (inp, 1), (add, 0, 1), (sin, 0), (mul, 2, 3),];

    let inputs = &[0.6, 1.4];

    // Test with cotangent = 1.0
    let (_value_box, backprop_box) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads_box = backprop_box(1.0);

    let (_value_arc, backprop_arc) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads_arc = backprop_arc(1.0);
    // Arc conversion: let arc_fn = Arc::from(backprop_arc);

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
        let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
        let grads = backprop_fn(cotangent);

        // Gradients should scale linearly with cotangent
        let (_value_base, backprop_fn_base) = MultiAD::compute_grad(exprs, inputs).unwrap();
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
fn test_multi_f() {
    F1(0.5, 1.0).demonstrate(true);
    F2(0.5, 1.0).demonstrate(true);
    F3(0.5, 1.0).demonstrate(true);
}

#[test]
fn test_f() {
    F1(0.5, 1.0).demonstrate(true);
    F2(0.5, 1.0).demonstrate(true);
    F3(0.5, 1.0).demonstrate(true);
}
// Tests for new operations: Pow, Sqrt, Abs

#[test]
fn test_forward_pow() {
    // Test x^y where x = 2.0, y = 3.0
    let exprs = &multi_ops![(inp, 0), (inp, 1), (pow, 0, 1),];
    let inputs = &[2.0, 3.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 8.0, 1e-10));
}

#[test]
fn test_backward_pow() {
    // Test gradient of x^y:
    // d(x^y)/dx = y * x^(y-1)
    // d(x^y)/dy = x^y * ln(x)
    let exprs = &multi_ops![(inp, 0), (inp, 1), (pow, 0, 1),];
    let x = 2.0;
    let y = 3.0;
    let inputs = &[x, y];

    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);

    assert_eq!(grads.len(), 2);
    let expected_dx = y * x.powf(y - 1.0); // 3 * 2^2 = 12
    let expected_dy = x.powf(y) * x.ln(); // 8 * ln(2)
    assert!(approx_eq(grads[0], expected_dx, 1e-10));
    assert!(approx_eq(grads[1], expected_dy, 1e-10));
}

#[test]
fn test_forward_sqrt() {
    // Test sqrt(x) where x = 9.0
    let exprs = &multi_ops![(sqrt, 0),];
    let inputs = &[9.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 3.0, 1e-10));
}

#[test]
fn test_backward_sqrt() {
    // Test gradient of sqrt(x): d(sqrt(x))/dx = 1/(2*sqrt(x))
    let exprs = &multi_ops![(sqrt, 0),];
    let x = 4.0;
    let inputs = &[x];

    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);

    let expected_grad = 1.0 / (2.0 * x.sqrt()); // 1/(2*2) = 0.25
    assert!(approx_eq(grads[0], expected_grad, 1e-10));
}

#[test]
fn test_forward_abs() {
    // Test abs(x) with positive and negative values
    let exprs = &multi_ops![(abs, 0),];

    // Positive value
    let inputs = &[5.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 5.0, 1e-10));

    // Negative value
    let inputs = &[-5.0];
    let result = MultiAD::compute(exprs, inputs).unwrap();
    assert!(approx_eq(result, 5.0, 1e-10));
}

#[test]
fn test_backward_abs() {
    // Test gradient of abs(x): d(|x|)/dx = sign(x) where sign(0) = 0
    let exprs = &multi_ops![(abs, 0),];

    // Positive value: derivative is 1
    let inputs = &[5.0];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);
    assert!(approx_eq(grads[0], 1.0, 1e-10));

    // Negative value: derivative is -1
    let inputs = &[-5.0];
    let (_value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);
    assert!(approx_eq(grads[0], -1.0, 1e-10));
}

#[test]
fn test_complex_with_pow() {
    // Test f(x, y, z) = x^y + z where x=2, y=3, z=1
    // This computes: 2^3 + 1 = 9
    let exprs = &multi_ops![
        (inp, 0),    // x at index 0
        (inp, 1),    // y at index 1
        (inp, 2),    // z at index 2
        (pow, 0, 1), // x^y at index 3
        (add, 3, 2), // x^y + z at index 4
    ];

    let x = 2.0;
    let y = 3.0;
    let z = 1.0;
    let inputs = &[x, y, z];

    let (value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);

    // f(2, 3, 1) = 2^3 + 1 = 8 + 1 = 9
    assert!(approx_eq(value, 9.0, 1e-10));

    // ∂f/∂x = y * x^(y-1) = 3 * 2^2 = 12
    // ∂f/∂y = x^y * ln(x) = 8 * ln(2)
    // ∂f/∂z = 1
    assert!(approx_eq(grads[0], 12.0, 1e-10));
    assert!(approx_eq(grads[1], 8.0 * 2.0_f64.ln(), 1e-10));
    assert!(approx_eq(grads[2], 1.0, 1e-10));
}

#[test]
fn test_sqrt_and_mul_chain() {
    // Test f(x, y) = sqrt(x) * y
    // This tests composition of sqrt and mul operations
    let exprs = &multi_ops![
        (inp, 0),    // x at index 0
        (inp, 1),    // y at index 1
        (sqrt, 0),   // sqrt(x) at index 2
        (mul, 2, 1), // sqrt(x) * y at index 3
    ];

    let x = 16.0;
    let y = 5.0;
    let inputs = &[x, y];

    let (value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
    let grads = backprop_fn(1.0);

    // f(16, 5) = sqrt(16) * 5 = 4 * 5 = 20
    assert!(approx_eq(value, 20.0, 1e-10));

    // Using chain rule:
    // Let f = sqrt(x) * y = u * y where u = sqrt(x)
    // ∂f/∂x = y * (1/(2*sqrt(x))) = 5 * (1/8) = 0.625
    // ∂f/∂y = sqrt(x) = 4
    assert!(approx_eq(grads[0], 5.0 / (2.0 * 16.0_f64.sqrt()), 1e-10));
    assert!(approx_eq(grads[1], 4.0, 1e-10));
}
