pub use super::multi_ad::MultiAD;
pub use super::types::BackwardResultBox;
use crate::error::Result;

/// Type alias for a multi-variable computation graph
pub type GraphType = [(MultiAD, Vec<usize>)];

/// Trait for multi-variable functions with analytical gradients.
///
/// Implement this trait to define custom mathematical functions that can be
/// compared against automatic differentiation results.
///
/// This trait is primarily intended for testing and demonstration purposes.
/// Most users will work directly with the `MultiAD` enum.
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
    fn compute(&self) -> Result<f64> {
        MultiAD::compute(self.graph(), &self.inputs())
    }

    /// Computes both value and gradients using automatic differentiation.
    fn compute_with_gradients(&self) -> Result<BackwardResultBox> {
        MultiAD::compute_grad(self.graph(), &self.inputs())
    }

    fn demonstrate(&self, with_assert: bool) {
        // Forward pass only
        let result = self.compute().unwrap();
        if with_assert {
            assert!((result - self.expected_value()).abs() < 1e-10);
        }

        println!("\nForward pass only:");
        println!("f({:?}) = {}", self.inputs(), result);

        // Forward + backward (automatic differentiation)
        let (value, backprop_fn) = self.compute_with_gradients().unwrap();
        if with_assert {
            assert!((value - self.expected_value()).abs() < 1e-10);
        }
        let grads = backprop_fn(1.0);
        println!("\nForward + backward (automatic differentiation):");
        println!("f({:?}) = {}", self.inputs(), value);
        for (i, grad) in grads.iter().enumerate() {
            println!("∂f/∂x{} = {}", i + 1, grad);
        }

        // Verify against analytical solution
        let expected_grad = self.expected_gradients();
        println!("\nAnalytical gradients:");
        println!("∂f/∂x₁, ∂f/∂x₂, ... = {:?}", expected_grad);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::approx_eq_eps as approx_eq;
    use std::sync::OnceLock;

    /// f(x, y) = x * y, ∇f = [y, x]
    struct MulFn;

    impl MultiFn for MulFn {
        fn inputs(&self) -> Vec<f64> {
            vec![2.0, 3.0]
        }
        fn graph(&self) -> &'static GraphType {
            static G: OnceLock<Vec<(MultiAD, Vec<usize>)>> = OnceLock::new();
            G.get_or_init(|| {
                vec![
                    (MultiAD::Inp, vec![0]),
                    (MultiAD::Inp, vec![1]),
                    (MultiAD::Mul, vec![0, 1]),
                ]
            })
        }
        fn expected_value(&self) -> f64 {
            6.0
        }
        fn expected_gradients(&self) -> Vec<f64> {
            vec![3.0, 2.0]
        }
    }

    #[test]
    fn test_multi_fn_compute_with_gradients() {
        let f = MulFn;
        let (value, backprop) = f.compute_with_gradients().unwrap();
        assert!(
            approx_eq(value, f.expected_value(), 1e-10),
            "value mismatch"
        );
        let grads = backprop(1.0);
        let expected = f.expected_gradients();
        for (a, b) in grads.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "gradient mismatch");
        }
    }

    #[test]
    fn test_multi_fn_demonstrate_with_assert() {
        MulFn.demonstrate(true);
    }

    #[test]
    fn test_multi_fn_demonstrate_without_assert() {
        MulFn.demonstrate(false);
    }

    #[test]
    fn test_multi_fn_compute_forward_only() {
        let f = MulFn;
        let result = f.compute().unwrap();
        assert!((result - f.expected_value()).abs() < 1e-10);
    }
}
