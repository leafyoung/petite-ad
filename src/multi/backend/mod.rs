//! Backend abstraction module for compiled graph execution.
//!
//! This module contains the types, traits, and backend implementations
//! for executing compiled graphs on different backends (scalar, SIMD, WGPU).

pub(crate) mod device;
pub(crate) mod dispatch;
pub(crate) mod mock;
pub(crate) mod scalar;
pub(crate) mod simd;
pub(crate) mod types;
#[cfg(feature = "backend-wgpu")]
pub(crate) mod wgpu;

pub use device::*;
pub use dispatch::*;
pub use mock::*;
pub use scalar::*;
pub use simd::*;
pub use types::*;
#[cfg(feature = "backend-wgpu")]
pub use wgpu::*;
