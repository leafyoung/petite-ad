// Example implementations - not part of public API
mod f1;
mod f2;
mod f3;

mod multi_ad;
mod multi_fn;
#[cfg(test)]
mod tests;
mod types;

#[cfg(test)]
pub(crate) use f1::F1;
#[cfg(test)]
pub(crate) use f2::F2;
#[cfg(test)]
pub(crate) use f3::F3;

pub use multi_ad::MultiAD;
pub use multi_fn::MultiFn;
