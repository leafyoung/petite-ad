//! Shared test utilities for the autodiff crate.

/// Check if two f64 values are approximately equal within a small epsilon.
///
/// Uses a default epsilon of 1e-10 for high precision comparisons.
#[cfg(test)]
#[allow(dead_code)] // Used conditionally in tests
pub(crate) fn approx_eq(a: f64, b: f64) -> bool {
    const EPSILON: f64 = 1e-10;
    (a - b).abs() < EPSILON
}

/// Check if two f64 values are approximately equal with a custom epsilon.
#[cfg(test)]
pub(crate) fn approx_eq_eps(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}
