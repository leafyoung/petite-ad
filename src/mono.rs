mod types;
#[allow(unused)]
use types::*;

#[cfg(test)]
mod tests;

mod mono_ad;
pub use mono_ad::MonoAD;

mod mono_fn;
pub use mono_fn::MonoFn;

// Example implementation - not part of public API
mod mf1;
#[cfg(test)]
pub(crate) use mf1::MF1;
