use std::sync::Arc;

use autodiff::{mono_ops, multi_ops, types::MonoResultBox, types::MultiResultBox, MonoAD, MultiAD};

fn main() {
    println!("=== Autodiff Library Demo ===\n");

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
}
