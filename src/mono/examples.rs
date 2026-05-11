//! Test-only example implementations of [`MonoFn`] for library tests.
//!
//! These are production-type examples used internally by the test suite and
//! are gated behind `#[cfg(test)]` to keep them out of release builds.

#[cfg(test)]
use super::func::{GraphType, MonoFn};
#[cfg(test)]
use crate::mono_ops;

/// f(x) = exp(sin(sin(x)))
#[cfg(test)]
pub struct MF1(pub f64);

#[cfg(test)]
impl MonoFn for MF1 {
    fn input(&self) -> f64 {
        self.0
    }

    fn graph(&self) -> &'static GraphType {
        &mono_ops![sin, sin, exp]
    }

    fn expected_value(&self) -> f64 {
        (self.0.sin().sin()).exp()
    }

    fn expected_gradient(&self) -> f64 {
        (self.0.sin().sin()).exp() * self.0.sin().cos() * self.0.cos()
    }
}

/// f(x) = -x
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

/// f(x) = sin(-x)
#[cfg(test)]
pub struct MF3(pub f64);

#[cfg(test)]
impl MonoFn for MF3 {
    fn input(&self) -> f64 {
        self.0
    }

    fn graph(&self) -> &'static GraphType {
        &mono_ops![neg, sin]
    }

    fn expected_value(&self) -> f64 {
        (-self.0).sin()
    }

    fn expected_gradient(&self) -> f64 {
        -((-self.0).cos())
    }
}

/// f(x) = -sin(x)
#[cfg(test)]
pub struct MF4(pub f64);

#[cfg(test)]
impl MonoFn for MF4 {
    fn input(&self) -> f64 {
        self.0
    }

    fn graph(&self) -> &'static GraphType {
        &mono_ops![sin, neg]
    }

    fn expected_value(&self) -> f64 {
        -(self.0.sin())
    }

    fn expected_gradient(&self) -> f64 {
        -(self.0.cos())
    }
}
