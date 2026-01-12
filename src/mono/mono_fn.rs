pub use super::mono_ad::MonoAD;
pub use super::types::{BackwardResultArc, BackwardResultBox};

#[allow(dead_code)]
pub type GraphType = [MonoAD];

#[allow(unused)]
pub trait MonoFn {
    fn to_value(&self) -> f64;
    fn f(&self) -> f64;
    fn graph(&self) -> &'static GraphType;

    fn grad(&self) -> f64;

    fn compute(&self) -> f64 {
        MonoAD::compute(self.graph(), self.to_value())
    }

    fn auto_grad(&self) -> BackwardResultBox {
        MonoAD::compute_grad(self.graph(), self.to_value())
    }

    fn auto_grad_arc(&self) -> BackwardResultArc {
        MonoAD::compute_grad_arc(self.graph(), self.to_value())
    }

    fn demonstrate(&self, with_assert: bool) {
        // Forward pass only
        let result = self.compute();
        if with_assert {
            assert!((result - self.f()).abs() < 1e-10);
        }
        println!("\nForward pass only:");
        println!("f({:?}) = {}", &self.to_value(), result);

        // Forward + backward (automatic differentiation)
        let (value, backprop_fn) = self.auto_grad();
        if with_assert {
            assert!((value - self.f()).abs() < 1e-10);
        }
        let grad = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation) with Box:");
        println!("f({:?}) = {}", &self.to_value(), value);
        println!("∂f/∂x = {}", grad);

        let (value, backprop_fn) = self.auto_grad_arc();
        let grad_arc = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation) with Arc:");
        println!("f({:?}) = {}", &self.to_value(), value);
        println!("∂f/∂x = {}", grad_arc);
        if with_assert {
            assert!((grad_arc - grad).abs() < 1e-10);
        }
        // Verify against analytical solution
        let expected_grad = self.grad();
        println!("\nAnalytical gradients:");
        println!("∂f/∂x = {:?}", &expected_grad);

        println!("\nGradient differences:");
        println!(
            "|∂f/∂x (auto) - ∂f/∂x (analytic)| = {}",
            (grad_arc - expected_grad).abs()
        );
        if with_assert {
            assert!((grad_arc - expected_grad).abs() < 1e-10);
        }
    }
}
