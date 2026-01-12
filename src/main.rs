use autodiff::{mono_ops, multi_ops, MonoAD, MultiAD};

fn main() {
    println!("=== Autodiff Library Demo ===\n");

    // Mono-variable automatic differentiation
    println!("--- Mono-variable automatic differentiation ---");
    let exprs = mono_ops![sin, sin, exp];
    let (value, backprop) = MonoAD::compute_grad(&exprs, 2.0);
    println!("f(2.0) = exp(sin(sin(2.0))) = {}", value);
    println!("f'(2.0) = {}", backprop(1.0));

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
    let (value, backprop_fn) = MultiAD::compute_grad(&exprs, &inputs);
    let grads = backprop_fn(1.0);
    
    println!("\nf(x₁, x₂) = sin(x₁) * (x₁ + x₂)");
    println!("f({}, {}) = {}", inputs[0], inputs[1], value);
    println!("∂f/∂x₁ = {}", grads[0]);
    println!("∂f/∂x₂ = {}", grads[1]);
}
