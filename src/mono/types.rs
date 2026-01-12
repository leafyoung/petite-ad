use super::mono_ad::MonoAD;
use std::sync::Arc;

#[allow(unused)]
pub type MathFn = fn(f64) -> f64;
pub type DynMathFn = dyn Fn(f64) -> f64;
pub type BackwardResultBox = (f64, Box<DynMathFn>);
pub type BackwardResultArc = (f64, Arc<DynMathFn>);

pub type GraphType = [MonoAD];
