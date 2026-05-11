//! Single-variable automatic differentiation.
//!
//! This module provides functionality for computing derivatives of
//! single-variable functions using reverse-mode differentiation.

pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_ho;

pub mod first_order;
pub use first_order::MonoAD;

pub mod second_order;
pub use second_order::fr::MonoAD2FR;
pub use second_order::rf::MonoAD2RF;
pub use second_order::rr::MonoAD2RR;

pub mod func;
// Re-export trait for library extension - users can implement custom mono functions
#[allow(unused_imports)]
pub use func::MonoFn;

#[cfg(test)]
mod examples;
