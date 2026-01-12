pub use super::mono_ad::MonoAD;
pub use super::types::BackwardResultBox;

/// Type alias for a graph of mono operations (slice of MonoAD)
#[allow(dead_code)] // Public API for library extension
pub type GraphType = [MonoAD];

/// Trait for single-variable functions with analytical gradients.
///
/// Implement this trait to define custom mathematical functions that can be
/// compared against automatic differentiation results.
///
/// This trait is primarily intended for testing and demonstration purposes.
/// Most users will work directly with the `MonoAD` enum.
#[allow(dead_code)] // Public API for library extension
pub trait MonoFn {
    /// Returns the input value for this function.
    fn input(&self) -> f64;

    /// Returns the computation graph for this function.
    fn graph(&self) -> &'static GraphType;

    /// Computes the expected function value analytically.
    fn expected_value(&self) -> f64;

    /// Computes the expected gradient analytically.
    fn expected_gradient(&self) -> f64;

    /// Computes the function value using automatic differentiation (forward pass only).
    fn compute(&self) -> f64 {
        MonoAD::compute(self.graph(), self.input())
    }

    /// Computes both value and gradient using automatic differentiation.
    fn compute_with_gradient(&self) -> BackwardResultBox {
        MonoAD::compute_grad(self.graph(), self.input())
    }

    #[cfg(test)]
    fn test_mono_ad(&self) {
            use crate::test_utils::approx_eq_eps as approx_eq;
            let (value, backprop) = MonoAD::compute_grad(self.graph(), self.input());
        let expected_value = self.expected_value();
        assert!(approx_eq(value, expected_value, 1e-10), "value mismatch");

        let compute_value = self.compute();
        assert!(
            approx_eq(compute_value, expected_value, 1e-10),
            "compute value mismatch"
        );

        let grad = self.expected_gradient();
        let expected_grad = backprop(1.0);
        assert!(approx_eq(expected_grad, grad, 1e-10), "gradient mismatch");

        let compute_value = self.compute_with_gradient().1(1.0);
        assert!(
            approx_eq(compute_value, expected_grad, 1e-10),
            "compute gradient mismatch"
        );
    }

    fn demonstrate(&self, with_assert: bool) {
        // Forward pass only
        let result = self.compute();
        if with_assert {
            assert!((result - self.expected_value()).abs() < 1e-10);
        }
        println!("\nForward pass only:");
        println!("f({:?}) = {}", &self.input(), result);

        // Forward + backward (automatic differentiation)
        let (value, backprop_fn) = self.compute_with_gradient();
        if with_assert {
            assert!((value - self.expected_value()).abs() < 1e-10);
        }
        let grad = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation):");
        println!("f({:?}) = {}", &self.input(), value);
        println!("∂f/∂x = {}", grad);

        // Verify against analytical solution
        let expected_grad = self.expected_gradient();
        println!("\nAnalytical gradients:");
        println!("∂f/∂x = {:?}", &expected_grad);

        println!("\nGradient differences:");
        println!(
            "|∂f/∂x (auto) - ∂f/∂x (analytic)| = {}",
            (grad - expected_grad).abs()
        );
        if with_assert {
            assert!((grad - expected_grad).abs() < 1e-10);
        }
    }
}
