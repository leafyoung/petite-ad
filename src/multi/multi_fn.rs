pub use super::multi_ad::MultiAD;
pub use super::types::{BackwardResultArc, BackwardResultBox};

#[allow(dead_code)]
pub type GraphType = [(MultiAD, Vec<usize>)];

#[allow(unused)]
pub trait MultiFn {
    fn to_vec(&self) -> Vec<f64>;
    fn f(&self) -> f64;
    fn graph(&self) -> &'static GraphType;

    fn grad(&self) -> Vec<f64>;

    fn compute(&self) -> f64 {
        MultiAD::compute(self.graph(), &self.to_vec())
    }

    fn auto_grad(&self) -> BackwardResultBox {
        MultiAD::compute_grad(self.graph(), &self.to_vec())
    }

    fn auto_grad_arc(&self) -> BackwardResultArc {
        MultiAD::compute_grad_arc(self.graph(), &self.to_vec())
    }

    fn demonstrate(&self) {
        // Forward pass only
        let result = self.compute();
        println!("\nForward pass only:");
        println!("f({:?}) = {}", &self.to_vec(), result);

        // Forward + backward (automatic differentiation)
        let (value, backprop_fn) = self.auto_grad();
        let grads = backprop_fn(1.0); // Call with cotangent = 1.0
        println!("\nForward + backward (automatic differentiation) with Box:");
        println!("f({:?}) = {}", &self.to_vec(), value);
        for (i, grad) in grads.iter().enumerate() {
            println!("∂f/∂x{} = {}", i + 1, grad);
        }

        let (value, backprop_fn) = self.auto_grad_arc();
        let grads = backprop_fn(1.0); // Call with cotangent = 1.0
        println!("\nForward + backward (automatic differentiation) with Arc:");
        println!("f({:?}) = {}", &self.to_vec(), value);
        for (i, grad) in grads.iter().enumerate() {
            println!("∂f/∂x{} = {}", i + 1, grad);
        }

        // Verify against analytical solution
        let expected_grad = self.grad().to_vec();
        println!("\nAnalytical gradients:");
        println!("∂f/∂x₁, ∂f/∂x₂, ... = {:?}", &expected_grad);

        println!("\nGradient differences:");
        for (expected, auto) in expected_grad.iter().zip(grads.iter()) {
            println!(
                "|∂f/∂x (auto) - ∂f/∂x (analytic)| = {}",
                (auto - expected).abs()
            );
        }
    }
}
