use std::sync::Arc;

#[allow(unused)]
pub type BackwardResult = (f64, Vec<f64>);

pub type DynGradFn = dyn Fn(f64) -> Vec<f64> + 'static;

pub type BackwardResultBox = (f64, Box<DynGradFn>);

pub type BackwardResultArc = (f64, Arc<DynGradFn>);
