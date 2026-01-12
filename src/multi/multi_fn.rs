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

    fn demonstrate(&self, with_assert: bool) {
        // Forward pass only
        let result = self.compute();
        if with_assert {
            assert!((result - self.f()).abs() < 1e-10);
        }

        println!("\nForward pass only:");
        println!("f({:?}) = {}", &self.to_vec(), result);

        // Forward + backward (automatic differentiation)
        let (value, backprop_fn) = self.auto_grad();
        if with_assert {
            assert!((value - self.f()).abs() < 1e-10);
        }
        let grads = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation) with Box:");
        println!("f({:?}) = {}", &self.to_vec(), value);
        for (i, grad) in grads.iter().enumerate() {
            println!("∂f/∂x{} = {}", i + 1, grad);
        }

        let (value, backprop_fn) = self.auto_grad_arc();
        let grads_arc = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation) with Arc:");
        println!("f({:?}) = {}", &self.to_vec(), value);
        for ((i, grad_auto), grad) in grads_arc.iter().enumerate().zip(grads.iter()) {
            println!("∂f/∂x{} = {}", i + 1, grad);
            if with_assert {
                assert!((grad_auto - grad).abs() < 1e-10);
            }
        }

        // Verify against analytical solution
        let expected_grad = self.grad().to_vec();
        println!("\nAnalytical gradients:");
        println!("∂f/∂x₁, ∂f/∂x₂, ... = {:?}", &expected_grad);

        println!("\nGradient differences:");
        for (expected, auto_arc) in expected_grad.iter().zip(grads_arc.iter()) {
            println!(
                "|∂f/∂x (auto) - ∂f/∂x (analytic)| = {}",
                (auto_arc - expected).abs(),
            );
            if with_assert {
                assert!((auto_arc - expected).abs() < 1e-10);
            }
        }
    }
}
