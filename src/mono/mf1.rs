#[cfg(test)]
use super::mono_fn::{GraphType, MonoFn};
#[cfg(test)]
use crate::mono_ops;

#[cfg(test)]
pub struct MF1(pub f64);

#[cfg(test)]
impl MonoFn for MF1 {
    fn input(&self) -> f64 {
        self.0
    }

    fn graph(&self) -> &'static GraphType {
        &mono_ops![sin, sin, exp]
    }

    fn expected_value(&self) -> f64 {
        (self.0.sin().sin()).exp()
    }

    fn expected_gradient(&self) -> f64 {
        (self.0.sin().sin()).exp() * self.0.sin().cos() * self.0.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mono::MonoAD;
    use crate::test_utils::approx_eq_eps as approx_eq;

    #[test]
    fn test_mf1_compute() {
        let mf1 = MF1(2.0);
        let (value, backprop) = MonoAD::compute_grad(mf1.graph(), mf1.input());
        let expected_value = mf1.expected_value();
        assert!(approx_eq(value, expected_value, 1e-10), "value mismatch");

        let compute_value = mf1.compute();
        assert!(
            approx_eq(compute_value, expected_value, 1e-10),
            "compute value mismatch"
        );

        let grad = mf1.expected_gradient();
        let expected_grad = backprop(1.0);
        assert!(approx_eq(expected_grad, grad, 1e-10), "gradient mismatch");

        let compute_value = mf1.compute_with_gradient().1(1.0);
        assert!(
            approx_eq(compute_value, expected_grad, 1e-10),
            "compute gradient mismatch"
        );
    }
}
