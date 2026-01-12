use super::mono_fn::{GraphType, MonoFn};
use crate::mono_ops;

pub struct MF1(pub f64);

impl MonoFn for MF1 {
    fn to_value(&self) -> f64 {
        self.0
    }

    fn f(&self) -> f64 {
        (self.0.sin().sin()).exp()
    }

    fn graph(&self) -> &'static GraphType {
        &mono_ops![sin, sin, exp]
    }

    fn grad(&self) -> f64 {
        (self.0.sin().sin()).exp() * self.0.sin().cos() * self.0.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mono::MonoAD;

    // Helper function for approximate equality with f64
    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_mf1_compute() {
        let mf1 = MF1(2.0);
        let (value, backprop) = MonoAD::compute_grad(mf1.graph(), mf1.to_value());
        let expected_value = mf1.f();
        assert!(approx_eq(value, expected_value, 1e-10), "value mismatch");

        let compute_value = mf1.compute();
        assert!(
            approx_eq(compute_value, expected_value, 1e-10),
            "compute value mismatch"
        );

        let grad = mf1.grad();
        let expected_grad = backprop(1.0);
        assert!(approx_eq(expected_grad, grad, 1e-10), "gradient mismatch");

        let compute_value = mf1.auto_grad().1(1.0);
        assert!(
            approx_eq(compute_value, expected_grad, 1e-10),
            "compute gradient mismatch"
        );
    }
}
