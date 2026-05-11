//! Core type definitions for the backend abstraction.

use crate::multi::multi_ad::MultiAD;
use crate::{AutodiffError, NodeId, Result};

/// One closure-free instruction in a compiled scalar graph.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    /// Literal constant node.
    Constant(f64),
    /// Unary operation with one input node id.
    Unary { op: MultiAD, arg: NodeId },
    /// Binary operation with two input node ids.
    Binary {
        op: MultiAD,
        left: NodeId,
        right: NodeId,
    },
}

/// Sentinel node id used by flat backend instructions when an argument is absent.
pub const UNUSED_NODE_ID: NodeId = usize::MAX;

/// Compact operation code used by flat backend instructions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    Constant,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sin,
    Cos,
    Tan,
    Tanh,
    Relu,
    Log1pExp,
    LogAddExp,
    Neg,
    Exp,
    Ln,
    Sqrt,
    Abs,
}

impl OpCode {
    pub(crate) fn from_multi_ad(op: MultiAD) -> Result<Self> {
        Ok(match op {
            MultiAD::Add => OpCode::Add,
            MultiAD::Sub => OpCode::Sub,
            MultiAD::Mul => OpCode::Mul,
            MultiAD::Div => OpCode::Div,
            MultiAD::Pow => OpCode::Pow,
            MultiAD::Sin => OpCode::Sin,
            MultiAD::Cos => OpCode::Cos,
            MultiAD::Tan => OpCode::Tan,
            MultiAD::Tanh => OpCode::Tanh,
            MultiAD::Relu => OpCode::Relu,
            MultiAD::Log1pExp => OpCode::Log1pExp,
            MultiAD::LogAddExp => OpCode::LogAddExp,
            MultiAD::Neg => OpCode::Neg,
            MultiAD::Exp => OpCode::Exp,
            MultiAD::Ln => OpCode::Ln,
            MultiAD::Sqrt => OpCode::Sqrt,
            MultiAD::Abs => OpCode::Abs,
            MultiAD::Inp => {
                return Err(AutodiffError::InvalidGraph {
                    reason: "input markers are not backend instructions",
                });
            }
        })
    }

    pub(crate) fn to_multi_ad(self) -> Option<MultiAD> {
        Some(match self {
            OpCode::Constant => return None,
            OpCode::Add => MultiAD::Add,
            OpCode::Sub => MultiAD::Sub,
            OpCode::Mul => MultiAD::Mul,
            OpCode::Div => MultiAD::Div,
            OpCode::Pow => MultiAD::Pow,
            OpCode::Sin => MultiAD::Sin,
            OpCode::Cos => MultiAD::Cos,
            OpCode::Tan => MultiAD::Tan,
            OpCode::Tanh => MultiAD::Tanh,
            OpCode::Relu => MultiAD::Relu,
            OpCode::Log1pExp => MultiAD::Log1pExp,
            OpCode::LogAddExp => MultiAD::LogAddExp,
            OpCode::Neg => MultiAD::Neg,
            OpCode::Exp => MultiAD::Exp,
            OpCode::Ln => MultiAD::Ln,
            OpCode::Sqrt => MultiAD::Sqrt,
            OpCode::Abs => MultiAD::Abs,
        })
    }

    /// Return the number of input slots consumed by this opcode.
    #[must_use]
    pub fn arity(&self) -> usize {
        match self {
            OpCode::Constant => 0,
            OpCode::Sin
            | OpCode::Cos
            | OpCode::Tan
            | OpCode::Tanh
            | OpCode::Relu
            | OpCode::Log1pExp
            | OpCode::Neg
            | OpCode::Exp
            | OpCode::Ln
            | OpCode::Sqrt
            | OpCode::Abs => 1,
            OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Pow
            | OpCode::LogAddExp => 2,
        }
    }
}

/// Fully explicit instruction shape for SIMD/GPU backend lowering.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatInstruction {
    pub opcode: OpCode,
    pub output: NodeId,
    pub left: NodeId,
    pub right: NodeId,
    pub value: f64,
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn supports_simd_f64x2_runtime() -> bool {
    true
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn supports_simd_f64x2_runtime() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn supports_simd_f64x4_runtime() -> bool {
    std::is_x86_feature_detected!("avx")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn supports_simd_f64x4_runtime() -> bool {
    false
}

#[inline]
pub(crate) fn supports_wgpu_runtime() -> bool {
    cfg!(feature = "backend-wgpu")
}

/// Backend feature declaration used before dispatching compiled graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub supports_f64: bool,
    pub supports_f32: bool,
    pub supports_constants: bool,
    pub supports_unary: bool,
    pub supports_binary: bool,
    pub supports_multi_output: bool,
    pub supports_reverse_gradient: bool,
    pub supports_batch_compute: bool,
    pub supports_batch_gradient: bool,
    pub supported_opcodes: Vec<OpCode>,
}

impl BackendCapabilities {
    /// Capabilities for the reference scalar f64 backend.
    #[must_use]
    pub fn scalar_f64() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: true,
            supports_batch_compute: true,
            supports_batch_gradient: true,
            supported_opcodes: vec![
                OpCode::Constant,
                OpCode::Add,
                OpCode::Sub,
                OpCode::Mul,
                OpCode::Div,
                OpCode::Pow,
                OpCode::Sin,
                OpCode::Cos,
                OpCode::Tan,
                OpCode::Tanh,
                OpCode::Relu,
                OpCode::Log1pExp,
                OpCode::LogAddExp,
                OpCode::Neg,
                OpCode::Exp,
                OpCode::Ln,
                OpCode::Sqrt,
                OpCode::Abs,
            ],
        }
    }

    /// Capabilities for the prototype f64x2 SIMD batch backend.
    #[must_use]
    pub fn simd_f64x2() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: false,
            supports_batch_compute: supports_simd_f64x2_runtime(),
            supports_batch_gradient: supports_simd_f64x2_runtime(),
            supported_opcodes: vec![
                OpCode::Constant,
                OpCode::Add,
                OpCode::Sub,
                OpCode::Mul,
                OpCode::Div,
                OpCode::Pow,
                OpCode::Sin,
                OpCode::Cos,
                OpCode::Tan,
                OpCode::Tanh,
                OpCode::Log1pExp,
                OpCode::LogAddExp,
                OpCode::Neg,
                OpCode::Exp,
                OpCode::Ln,
                OpCode::Sqrt,
                OpCode::Relu,
                OpCode::Abs,
            ],
        }
    }

    /// Capabilities for the prototype f64x4 SIMD batch backend.
    #[must_use]
    pub fn simd_f64x4() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: false,
            supports_batch_compute: supports_simd_f64x4_runtime(),
            supports_batch_gradient: supports_simd_f64x4_runtime(),
            supported_opcodes: vec![
                OpCode::Constant,
                OpCode::Add,
                OpCode::Sub,
                OpCode::Mul,
                OpCode::Div,
                OpCode::Pow,
                OpCode::Sin,
                OpCode::Cos,
                OpCode::Tan,
                OpCode::Tanh,
                OpCode::Log1pExp,
                OpCode::LogAddExp,
                OpCode::Neg,
                OpCode::Exp,
                OpCode::Ln,
                OpCode::Sqrt,
                OpCode::Relu,
                OpCode::Abs,
            ],
        }
    }

    /// Capabilities for the feature-gated WGPU batch backend skeleton.
    #[must_use]
    pub fn wgpu_f64() -> Self {
        Self {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: true,
            supports_batch_compute: supports_wgpu_runtime(),
            supports_batch_gradient: supports_wgpu_runtime(),
            supported_opcodes: BackendCapabilities::scalar_f64().supported_opcodes,
        }
    }

    /// Return whether the backend declares support for an opcode.
    #[must_use]
    pub fn supports_opcode(&self, opcode: OpCode) -> bool {
        self.supported_opcodes.contains(&opcode)
    }
}
