pub use super::multi_ad::MultiAD;
pub use super::types::BackwardResultBox;

/// Type alias for a multi-variable computation graph
#[allow(dead_code)] // Public API for library extension
pub type GraphType = [(MultiAD, Vec<usize>)];

/// Trait for multi-variable functions with analytical gradients.
///
/// Implement this trait to define custom mathematical functions that can be
/// compared against automatic differentiation results.
///
/// This trait is primarily intended for testing and demonstration purposes.
/// Most users will work directly with the `MultiAD` enum.
#[allow(dead_code)] // Public API for library extension
pub trait MultiFn {
    /// Returns the input values for this function.
    fn inputs(&self) -> Vec<f64>;

    /// Returns the computation graph for this function.
    fn graph(&self) -> &'static GraphType;

    /// Computes the expected function value analytically.
    fn expected_value(&self) -> f64;

    /// Computes the expected gradients analytically.
    fn expected_gradients(&self) -> Vec<f64>;

    /// Computes the function value using automatic differentiation (forward pass only).
    fn compute(&self) -> f64 {
        MultiAD::compute(self.graph(), &self.inputs())
    }

    /// Computes both value and gradients using automatic differentiation.
    fn compute_with_gradients(&self) -> BackwardResultBox {
        MultiAD::compute_grad(self.graph(), &self.inputs())
    }

    fn demonstrate(&self, with_assert: bool) {
        // Forward pass only
        let result = self.compute();
        if with_assert {
            assert!((result - self.expected_value()).abs() < 1e-10);
        }

        println!("\nForward pass only:");
        println!("f({:?}) = {}", &self.inputs(), result);

        // Forward + backward (automatic differentiation)
        let (value, backprop_fn) = self.compute_with_gradients();
        if with_assert {
            assert!((value - self.expected_value()).abs() < 1e-10);
        }
        let grads = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation):");
        println!("f({:?}) = {}", &self.inputs(), value);
        for (i, grad) in grads.iter().enumerate() {
            println!("∂f/∂x{} = {}", i + 1, grad);
        }

        // Verify against analytical solution
        let expected_grad = self.expected_gradients();
        println!("\nAnalytical gradients:");
        println!("∂f/∂x₁, ∂f/∂x₂, ... = {:?}", &expected_grad);

        println!("\nGradient differences:");
        for (expected, auto) in expected_grad.iter().zip(grads.iter()) {
            println!(
                "|∂f/∂x (auto) - ∂f/∂x (analytic)| = {}",
                (auto - expected).abs(),
            );
            if with_assert {
                assert!((auto - expected).abs() < 1e-10);
            }
        }
    }
}
