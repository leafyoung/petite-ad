#[cfg(test)]
use super::multi_ad::MultiAD;
#[cfg(test)]
use super::multi_fn::{GraphType, MultiFn};
#[cfg(test)]
use crate::multi_ops;

#[cfg(test)]
pub struct F3(pub f64, pub f64); // Represents f(x₁, x₂) = sin(x₁) * ln(x₂)

#[cfg(test)]
impl MultiFn for F3 {
    fn inputs(&self) -> Vec<f64> {
        vec![self.0, self.1]
    }

    fn graph(&self) -> &'static GraphType {
        use std::sync::LazyLock;
        static GRAPH: LazyLock<Vec<(MultiAD, Vec<usize>)>> = LazyLock::new(|| {
            Vec::from(multi_ops![
                (inp, 0),    // x₁ at index 0
                (inp, 1),    // x₂ at index 1
                (ln, 1),     // ln(x₂) at index 2
                (sin, 0),    // sin(x₁) at index 3
                (mul, 3, 2), // sin(x₁) * ln(x₂) at index 4
            ])
        });
        &GRAPH
    }

    /// Example function: f(x₁, x₂) = sin(x₁) * ln(x₂)
    fn expected_value(&self) -> f64 {
        self.0.sin() * self.1.ln()
    }

    /// Analytical gradient of f: (∂f/∂x₁, ∂f/∂x₂)
    /// Using product rule: d(sin(x) * ln(y))/dx = cos(x) * ln(y), d/dy = sin(x) / y
    fn expected_gradients(&self) -> Vec<f64> {
        let df_dx1 = self.0.cos() * self.1.ln();
        let df_dx2 = self.0.sin() / self.1;
        vec![df_dx1, df_dx2]
    }
}
