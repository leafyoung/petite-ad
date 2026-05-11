pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_ho;

pub mod mono_ad;
pub use mono_ad::MonoAD;

pub mod mono_ad_rr;
pub use mono_ad_rr::MonoAD2RR;

pub mod mono_ad_fr;
mod mono_hessian_common;
pub use mono_ad_fr::MonoAD2FR;

pub mod mono_ad_rf;
pub use mono_ad_rf::MonoAD2RF;

mod mono_fn;
// Re-export trait for library extension - users can implement custom mono functions
#[allow(unused_imports)] // May not be used internally, but part of public API
pub use mono_fn::MonoFn;

// Example implementation - not part of public API
mod mf1;
mod mf2;
mod mf3;
mod mf4;
