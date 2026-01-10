mod f;
use autodiff::{compute, compute_arc, math_ops};
use f::{f, f_prime};

fn main() {
    println!("Hello, world!");
    println!("f(2.0) = {}, f_prime(2.0) = {}", f(2.0), f_prime(2.0));

    // Use the macro to convert function names to MathOp at compile time
    let (value, backprop) = compute(&math_ops![sin, sin, exp], 2.0);
    println!("backprop: {} {}", value, backprop(1.0));

    let (value, backprop) = compute_arc(&math_ops![sin, sin, exp], 2.0);
    println!("backprop: {} {}", value, backprop(1.0));
}
