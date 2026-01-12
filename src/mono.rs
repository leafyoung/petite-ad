pub mod types;

#[cfg(test)]
mod tests;

mod mono_ad;
pub use mono_ad::MonoAD;

mod mono_fn;
// Re-export trait for library extension - users can implement custom mono functions
#[allow(unused_imports)] // May not be used internally, but part of public API
pub use mono_fn::MonoFn;

// Example implementation - not part of public API
mod mf1;
