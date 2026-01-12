//! Multi-variable automatic differentiation.
//!
//! This module provides functionality for computing gradients of
//! multi-variable functions using computational graphs.

// Example implementations - not part of public API
mod f1;
mod f2;
mod f3;

mod multi_ad;
mod multi_fn;
#[cfg(test)]
mod tests;
pub mod types;

pub use multi_ad::MultiAD;
// Re-export trait for library extension - users can implement custom multi-variable functions
#[allow(unused_imports)] // May not be used internally, but part of public API
pub use multi_fn::MultiFn;
