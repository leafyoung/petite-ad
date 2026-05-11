//! Type definitions for single-variable automatic differentiation.

use std::sync::Arc;

/// Dynamic trait object for single-variable gradient functions
pub type DynMathFn = dyn Fn(f64) -> f64;

/// Result type containing value and gradient function (Box-wrapped)
pub type BackwardResultBox = (f64, Box<DynMathFn>);

/// Result type containing value and gradient function (Arc-wrapped for sharing)
pub type BackwardResultArc = (f64, Arc<DynMathFn>);

/// Dual number for forward-mode automatic differentiation.
///
/// A dual number represents a value and its derivative (tangent) in a specified direction.
/// Used in Forward-over-Reverse (FR) and Reverse-over-Forward (RF) modes to compute
/// second-order derivatives by differentiating first-order gradients.
///
/// # Structure
///
/// - `val`: The primal (function) value
/// - `tan`: The tangent (derivative in the forward direction)
///
/// # Arithmetic Rules
///
/// For an operation y = op(x) where x is a dual number:
///
/// ```text
/// Sin: (sin(x.val), cos(x.val) * x.tan)
/// Cos: (cos(x.val), -sin(x.val) * x.tan)
/// Exp: (exp(x.val), exp(x.val) * x.tan)
/// Neg: (-x.val, -x.tan)
/// ```
///
/// # Example (unit test)
///
/// See [`tests::test_dual_forward_sin`] for a runnable forward-mode example.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    pub val: f64,
    pub tan: f64,
}

impl Dual {
    /// Create a new dual number with specified value and tangent.
    #[allow(dead_code)]
    pub fn new(val: f64, tan: f64) -> Self {
        Self { val, tan }
    }

    /// Create a constant dual number (zero derivative).
    ///
    /// Used for constants in the computation graph that don't depend on inputs.
    #[allow(dead_code)]
    pub fn constant(val: f64) -> Self {
        Self { val, tan: 0.0 }
    }

    /// Create a variable dual number (unit derivative).
    ///
    /// Used for input variable when computing derivatives with respect to it.
    pub fn variable(val: f64) -> Self {
        Self { val, tan: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_variable() {
        let d = Dual::variable(2.0);
        assert_eq!(d.val, 2.0);
        assert_eq!(d.tan, 1.0);
    }

    #[test]
    fn test_dual_new() {
        let d = Dual::new(3.0, 0.5);
        assert_eq!(d.val, 3.0);
        assert_eq!(d.tan, 0.5);
    }

    #[test]
    fn test_dual_constant() {
        let d = Dual::constant(5.0);
        assert_eq!(d.val, 5.0);
        assert_eq!(d.tan, 0.0);
    }

    #[test]
    fn test_dual_forward_sin() {
        let x = Dual::variable(2.0);
        let y = Dual {
            val: x.val.sin(),
            tan: x.val.cos() * x.tan,
        };
        assert!((y.val - 2.0_f64.sin()).abs() < 1e-15);
        assert!((y.tan - 2.0_f64.cos()).abs() < 1e-15);
    }
}
