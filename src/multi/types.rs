//! Type definitions for multi-variable automatic differentiation.

use std::sync::Arc;

/// Dynamic trait object for multi-variable gradient functions
pub type DynGradFn = dyn Fn(f64) -> Vec<f64> + 'static;

/// Result type containing value and gradient function (Box-wrapped)
pub type BackwardResultBox = (f64, Box<DynGradFn>);

/// Result type containing value and gradient function (Arc-wrapped for sharing)
pub type BackwardResultArc = (f64, Arc<DynGradFn>);
