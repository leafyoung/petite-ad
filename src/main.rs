use std::sync::Arc;

use petite_ad::{
    mono_ops, multi_ops, types::MonoResultBox, types::MultiResultBox, MonoAD, MultiAD,
};

fn main() {
    println!("=== Autodiff Library Demo ===\n");

    // Mono-variable automatic differentiation
    println!("--- Mono-variable automatic differentiation ---");
    // Example 1.
    println!("\n1. Obtain value and gradient for mono-variate function:");
    let exprs = mono_ops![sin, sin, exp];
    let (value, backprop): MonoResultBox = MonoAD::compute_grad(&exprs, 2.0);
    println!("f(2.0) = exp(sin(sin(2.0))) = {}", value);
    println!("f'(2.0) = {}", backprop(1.0));

    // Example 2: Converting Box to Arc for thread-safe sharing
    println!("\n2. Converting to Arc for multi-threaded use:");
    let (_, grad_fn_box) = MonoAD::compute_grad(&exprs, 2.0);
    let grad_fn_arc: Arc<dyn Fn(f64) -> f64> = Arc::from(grad_fn_box);
    let grad_fn_clone = grad_fn_arc.clone(); // Can clone Arc
    println!("   Arc gradient: {:.4}", grad_fn_arc(1.0));
    println!("   Cloned gradient: {:.4}", grad_fn_clone(1.0));

    // Example 3: Second derivative (Hessian for single-variable)
    println!("\n3. Second derivative computation:");
    let sin_exprs = mono_ops![sin];
    let x = 0.5;
    let second_deriv = MonoAD::compute_hessian(&sin_exprs, x);
    println!("f(x) = sin(x), f''(x) = -sin(x)");
    println!("At x = {:.1}, f''({:.1}) = {:.4}", x, x, second_deriv);
    println!("Expected: f''(0.5) = -sin(0.5) = {:.4}", -x.sin());

    let exp_sin_exprs = mono_ops![sin, exp];
    let x2 = 0.5;
    let second_deriv2 = MonoAD::compute_hessian(&exp_sin_exprs, x2);
    println!("\nf(x) = exp(sin(x))");
    println!("At x = {:.1}, f''({:.1}) = {:.4}", x2, x2, second_deriv2);
    let expected2 = x2.sin().exp() * x2.cos().powi(2) - x2.sin().exp() * x2.sin();
    println!("Expected: {:.4}", expected2);

    // Multi-variable automatic differentiation
    println!("\n--- Multi-variable automatic differentiation ---");

    // Example 1: f(x₁, x₂) = sin(x₁) * (x₁ + x₂)
    let exprs = multi_ops![
        (inp, 0),    // x₁ at index 0
        (inp, 1),    // x₂ at index 1
        (add, 0, 1), // x₁ + x₂ at index 2
        (sin, 0),    // sin(x₁) at index 3
        (mul, 2, 3), // sin(x₁) * (x₁ + x₂) at index 4
    ];

    let inputs = [0.4, 1.6];
    let (value, backprop_fn) = MultiAD::compute_grad(&exprs, &inputs).unwrap();
    let grads = backprop_fn(1.0);

    println!("\nf(x₁, x₂) = sin(x₁) * (x₁ + x₂)");
    println!("f({:?}) = {}", inputs, value);
    println!("∂f/∂x₁ = {}", grads[0]);
    println!("∂f/∂x₂ = {}", grads[1]);

    // Example 2: Multi-variable with type alias
    println!("\n2. Multi-variable with type aliases:");
    let result: MultiResultBox = MultiAD::compute_grad(&exprs, &[3.0, 4.0]).unwrap();
    let (value, grad_fn) = result;
    let grads = grad_fn(1.0);
    println!("   f(3.0, 4.0) = {:.1}", value);
    println!("   ∇f = [{:.1}, {:.1}]", grads[0], grads[1]);

    // Example 3: Demonstrate a more complex graph
    println!("\n=== Complex Graph Example ===\n");
    // f(x, y, z) = (x + y) * exp(z - sin(x))
    let complex_exprs = multi_ops![
        (inp, 0),    // x at index 0
        (inp, 1),    // y at index 1
        (inp, 2),    // z at index 2
        (add, 0, 1), // x + y at index 3
        (sin, 0),    // sin(x) at index 4
        (sub, 2, 4), // z - sin(x) at index 5
        (exp, 5),    // exp(z - sin(x)) at index 6
        (mul, 3, 6), // (x + y) * exp(z - sin(x)) at index 7
    ];

    let inputs2 = [1.0, 2.0, 0.5];
    let (value2, backprop_fn2) = MultiAD::compute_grad(&complex_exprs, &inputs2).unwrap();
    let grads2 = backprop_fn2(1.0);

    println!("Function: f(x, y, z) = (x + y) * exp(z - sin(x))");
    println!(
        "Inputs: x = {}, y = {}, z = {}",
        inputs2[0], inputs2[1], inputs2[2]
    );
    println!("Value: {}", value2);
    println!(
        "Gradients: ∂f/∂x = {:.4}, ∂f/∂y = {:.4}, ∂f/∂z = {:.4}",
        grads2[0], grads2[1], grads2[2]
    );

    // analytical gradients
    let x = inputs2[0];
    let y = inputs2[1];
    let z = inputs2[2];
    let exp_term = (z - x.sin()).exp();
    let analytical_dx = exp_term * (1.0 - (x + y) * x.cos()); // exp_term + (x + y) * exp_term * (-x.cos())
    let analytical_dy = exp_term;
    let analytical_dz = (x + y) * exp_term;

    println!(
        "Analytical Gradients: ∂f/∂x = {:.4}, ∂f/∂y = {:.4}, ∂f/∂z = {:.4}",
        analytical_dx, analytical_dy, analytical_dz
    );

    // Example 4: Hessian computation (second-order derivatives)
    println!("\n=== Hessian Computation (Second-Order Derivatives) ===\n");

    // Example 1: f(x, y) = x² + y²
    // Hessian: [[2, 0], [0, 2]]
    let quadratic_exprs = multi_ops![(inp, 0), (inp, 1), (mul, 0, 0), (mul, 1, 1), (add, 2, 3)];
    let inputs = [2.0, 3.0];
    let hessian = MultiAD::compute_hessian(&quadratic_exprs, &inputs).unwrap();

    println!("Function: f(x, y) = x² + y²");
    println!("Point: (x, y) = ({}, {})", inputs[0], inputs[1]);
    println!("Hessian matrix:");
    println!("  [[{:.1}, {:.1}],", hessian[0][0], hessian[0][1]);
    println!("   [{:.1}, {:.1}]]", hessian[1][0], hessian[1][1]);
    println!("Expected: [[2.0, 0.0], [0.0, 2.0]]");
    println!(
        "Interpretation: ∂²f/∂x² = {:.1}, ∂²f/∂x∂y = {:.1}, ∂²f/∂y∂x = {:.1}, ∂²f/∂y² = {:.1}",
        hessian[0][0], hessian[0][1], hessian[1][0], hessian[1][1]
    );

    // Example 2: f(x, y) = x * y
    // Hessian: [[0, 1], [1, 0]]
    println!("\nFunction: f(x, y) = x * y");
    let cross_exprs = multi_ops![(inp, 0), (inp, 1), (mul, 0, 1)];
    let cross_hessian = MultiAD::compute_hessian(&cross_exprs, &[2.0, 3.0]).unwrap();
    println!("Point: (x, y) = (2.0, 3.0)");
    println!("Hessian matrix:");
    println!(
        "  [[{:.1}, {:.1}],",
        cross_hessian[0][0], cross_hessian[0][1]
    );
    println!(
        "   [{:.1}, {:.1}]]",
        cross_hessian[1][0], cross_hessian[1][1]
    );
    println!("Expected: [[0.0, 1.0], [1.0, 0.0]]");

    // Example 3: Single-variable second derivative
    // f(x) = x³, f''(x) = 6x
    println!("\nFunction: f(x) = x³");
    let cubic_exprs = multi_ops![(inp, 0), (mul, 0, 0), (mul, 1, 0)];
    let x = 3.0;
    let cubic_hessian = MultiAD::compute_hessian(&cubic_exprs, &[x]).unwrap();
    println!("Point: x = {}", x);
    println!("Hessian (1x1 matrix): [[{:.4}]]", cubic_hessian[0][0]);
    println!("Expected: [[18.0]] (since f''(x) = 6x = 6*3 = 18)");
}
