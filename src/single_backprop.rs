/// Stable automatic differentiation implementation using enums.
/// Solves the function pointer hashing issue in release mode.
use crate::{BackwardResult, MathOp};

pub fn compute(exprs: &[MathOp], x: f64) -> BackwardResult {
    let mut value = x;
    let mut backprops = Vec::new();

    // Compute backward pass for each operation
    for &op in exprs {
        let (new_value, backprop) = op.backward(value);
        value = new_value;
        backprops.push(backprop);
    }

    // Chain all the backward functions
    let backward_fn = move |cotangent: f64| -> f64 {
        let mut grad = cotangent;
        for backprop in backprops.iter().rev() {
            grad = backprop(grad);
        }
        grad
    };

    (value, Box::new(backward_fn))
}
