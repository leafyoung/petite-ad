//! Type definitions for single-variable automatic differentiation.

use std::sync::Arc;

/// Dynamic trait object for single-variable gradient functions
pub type DynMathFn = dyn Fn(f64) -> f64;

/// Result type containing value and gradient function (Box-wrapped)
pub type BackwardResultBox = (f64, Box<DynMathFn>);

/// Result type containing value and gradient function (Arc-wrapped for sharing)
pub type BackwardResultArc = (f64, Arc<DynMathFn>);
