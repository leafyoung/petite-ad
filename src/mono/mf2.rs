#[cfg(test)]
use super::mono_fn::{GraphType, MonoFn};
#[cfg(test)]
use crate::mono_ops;

#[cfg(test)]
pub struct MF2(pub f64);

#[cfg(test)]
impl MonoFn for MF2 {
    fn input(&self) -> f64 {
        self.0
    }

    fn graph(&self) -> &'static GraphType {
        &mono_ops![neg]
    }

    fn expected_value(&self) -> f64 {
        -self.0
    }

    fn expected_gradient(&self) -> f64 {
        -1.0
    }
}
