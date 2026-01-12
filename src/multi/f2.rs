use crate::multi_ops;

pub use super::multi_fn::{GraphType, MultiFn};

#[allow(unused)]
pub struct F2(pub f64, pub f64); // Represents f(x₁, x₂) = sin(x₁) * (x₁ + x₂)

impl MultiFn for F2 {
    fn to_vec(&self) -> Vec<f64> {
        vec![self.0, self.1]
    }

    /// Example function: f(x₁, x₂) = sin(x₁) / (x₁ - x₂)
    fn f(&self) -> f64 {
        self.0.sin() / (self.0 - self.1)
    }

    /// Analytical gradient of f: (∂f/∂x₁, ∂f/∂x₂)
    /// Using product rule: d(sin(x) / (x - y))/dx = cos(x) * (x - y) - sin(x) / x.powi(2)
    fn grad(&self) -> Vec<f64> {
        let df_dx1 = self.0.cos() / (self.0 - self.1) - self.0.sin() / (self.0 - self.1).powi(2);
        let df_dx2 = self.0.sin() / (self.0 - self.1).powi(2);
        vec![df_dx1, df_dx2]
    }

    fn graph(&self) -> &'static GraphType {
        // Using Box::leak to create a static reference
        // Box::leak converts Box<T> -> &'static T by leaking memory (never freed)
        // This is idiomatic for application constants that should live forever
        use std::sync::OnceLock;
        static GRAPH: OnceLock<&'static GraphType> = OnceLock::new();

        GRAPH.get_or_init(|| {
            Box::leak(Box::new(multi_ops![
                (inp, 0),    // x₁ at index 0
                (inp, 1),    // x₂ at index 1
                (sub, 0, 1), // x₁ - x₂ at index 2
                (sin, 0),    // sin(x₁) at index 3
                (div, 3, 2), // sin(x₁) * (x₁ - x₂) at index 4
            ]))
        })
    }
}
