/// Demo of the multi_ops! macro for clean computation graph syntax
use autodiff::{MultiAD, multi_ops};

fn main() {
    // Build computation graph: f(x₁, x₂) = sin(x₁) * (x₁ + x₂)
    // Using the new lowercase macro - much cleaner!
    let exprs = multi_ops![
        (inp, 0),    // x₁ at index 0
        (inp, 1),    // x₂ at index 1
        (add, 0, 1), // x₁ + x₂ at index 2
        (sin, 0),    // sin(x₁) at index 3
        (mul, 2, 3), // sin(x₁) * (x₁ + x₂) at index 4
    ];

    let inputs = [0.6, 1.4];

    // Compute gradients using Box version
    let (value, backprop_fn) = MultiAD::compute_grad(&exprs, &inputs);
    let grads_box = backprop_fn(1.0);

    // Compute gradients using Arc version
    let (_value_arc, backprop_fn_arc) = MultiAD::compute_grad_arc(&exprs, &inputs);
    let grads_arc = backprop_fn_arc(1.0);

    println!("=== multi_ops! Macro Demo ===\n");
    println!("Function: f(x₁, x₂) = sin(x₁) * (x₁ + x₂)");
    println!("Inputs: x₁ = {}, x₂ = {}", inputs[0], inputs[1]);
    println!("\nValue: {}", value);
    println!(
        "Gradients (Box): ∂f/∂x₁ = {:.6}, ∂f/∂x₂ = {:.6}",
        grads_box[0], grads_box[1]
    );
    println!(
        "Gradients (Arc):  ∂f/∂x₁ = {:.6}, ∂f/∂x₂ = {:.6}",
        grads_arc[0], grads_arc[1]
    );

    // Verify against analytical gradient
    // f(x, y) = sin(x) * (x + y)
    // ∂f/∂x = cos(x) * (x + y) + sin(x)
    // ∂f/∂y = sin(x)
    let x = inputs[0];
    let y = inputs[1];
    let analytical_dx = x.cos() * (x + y) + x.sin();
    let analytical_dy = x.sin();

    println!(
        "\nAnalytical:   ∂f/∂x₁ = {:.6}, ∂f/∂x₂ = {:.6}",
        analytical_dx, analytical_dy
    );
    println!(
        "Box error:    ∂f/∂x₁ = {:.2e}, ∂f/∂x₂ = {:.2e}",
        (grads_box[0] - analytical_dx).abs(),
        (grads_box[1] - analytical_dy).abs()
    );
    println!(
        "Arc error:    ∂f/∂x₁ = {:.2e}, ∂f/∂x₂ = {:.2e}",
        (grads_arc[0] - analytical_dx).abs(),
        (grads_arc[1] - analytical_dy).abs()
    );

    // Demonstrate a more complex graph
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
    let (value2, backprop_fn2) = MultiAD::compute_grad(&complex_exprs, &inputs2);
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
}
