//! Test-only example implementations of [`MultiFn`] for library tests.
//!
//! These are production-type examples used internally by the test suite and
//! are gated behind `#[cfg(test)]` to keep them out of release builds.

#[cfg(test)]
use super::first_order::MultiAD;
#[cfg(test)]
use super::func::{GraphType, MultiFn};
#[cfg(test)]
use crate::multi_ops;

/// f(x₁, x₂) = sin(x₁) * (x₁ + x₂)
#[cfg(test)]
pub struct F1(pub f64, pub f64);

#[cfg(test)]
impl MultiFn for F1 {
    fn inputs(&self) -> Vec<f64> {
        vec![self.0, self.1]
    }

    fn graph(&self) -> &'static GraphType {
        use std::sync::LazyLock;
        static GRAPH: LazyLock<Vec<(MultiAD, Vec<usize>)>> = LazyLock::new(|| {
            Vec::from(multi_ops![
                (inp, 0),    // x₁ at index 0
                (inp, 1),    // x₂ at index 1
                (add, 0, 1), // x₁ + x₂ at index 2
                (sin, 0),    // sin(x₁) at index 3
                (mul, 2, 3), // sin(x₁) * (x₁ + x₂) at index 4
            ])
        });
        &GRAPH
    }

    fn expected_value(&self) -> f64 {
        self.0.sin() * (self.0 + self.1)
    }

    fn expected_gradients(&self) -> Vec<f64> {
        let df_dx1 = self.0.cos() * (self.0 + self.1) + self.0.sin();
        let df_dx2 = self.0.sin();
        vec![df_dx1, df_dx2]
    }
}

/// f(x₁, x₂) = sin(x₁) / (x₁ - x₂)
#[cfg(test)]
pub struct F2(pub f64, pub f64);

#[cfg(test)]
impl MultiFn for F2 {
    fn inputs(&self) -> Vec<f64> {
        vec![self.0, self.1]
    }

    fn graph(&self) -> &'static GraphType {
        use std::sync::LazyLock;
        static GRAPH: LazyLock<Vec<(MultiAD, Vec<usize>)>> = LazyLock::new(|| {
            Vec::from(multi_ops![
                (inp, 0),    // x₁ at index 0
                (inp, 1),    // x₂ at index 1
                (sub, 0, 1), // x₁ - x₂ at index 2
                (sin, 0),    // sin(x₁) at index 3
                (div, 3, 2), // sin(x₁) / (x₁ - x₂) at index 4
            ])
        });
        &GRAPH
    }

    fn expected_value(&self) -> f64 {
        self.0.sin() / (self.0 - self.1)
    }

    fn expected_gradients(&self) -> Vec<f64> {
        let df_dx1 = self.0.cos() / (self.0 - self.1) - self.0.sin() / (self.0 - self.1).powi(2);
        let df_dx2 = self.0.sin() / (self.0 - self.1).powi(2);
        vec![df_dx1, df_dx2]
    }
}

/// f(x₁, x₂) = sin(x₁) * ln(x₂)
#[cfg(test)]
pub struct F3(pub f64, pub f64);

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

    fn expected_value(&self) -> f64 {
        self.0.sin() * self.1.ln()
    }

    fn expected_gradients(&self) -> Vec<f64> {
        let df_dx1 = self.0.cos() * self.1.ln();
        let df_dx2 = self.0.sin() / self.1;
        vec![df_dx1, df_dx2]
    }
}
