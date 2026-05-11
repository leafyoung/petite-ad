//! Prototype f64x2/f64x4 SIMD backend for batch compute and batch gradients.

use crate::multi::backend::device::DeviceBackend;
use crate::multi::backend::dispatch::{BackendKind, ExecutionBackend};
use crate::multi::backend::types::{
    supports_simd_f64x4_runtime, BackendCapabilities, FlatInstruction, OpCode,
};
use crate::multi::compiled::{BatchGradientsBuffer, BatchInputs, BatchValuesBuffer, CompiledGraph};
use crate::multi::op_rules;
use crate::{AutodiffError, NodeId, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128d, __m256d};

/// Prototype f64x2 SIMD backend for batch compute and batch gradients.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimdBackend;

impl DeviceBackend for SimdBackend {
    fn backend_kind(&self, graph: &CompiledGraph) -> BackendKind {
        graph
            .simd_support_report()
            .map(|report| report.backend)
            .unwrap_or(BackendKind::SimdF64x2)
    }
}

impl ExecutionBackend for SimdBackend {
    fn name(&self) -> &'static str {
        "simd"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::simd_f64x2()
    }

    fn compute(&self, _graph: &CompiledGraph, _inputs: &[f64]) -> Result<f64> {
        Err(AutodiffError::InvalidGraph {
            reason: "simd backend currently supports batch compute only",
        })
    }

    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        if let Ok(report) = graph.backend_support_report(BackendKind::SimdF64x4) {
            if report.can_compute_batch() {
                return compute_batch_simd_f64x4(graph, batch, buffer);
            }
        }
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_compute {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch compute on this target",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        compute_batch_simd_f64x2(graph, batch, buffer)
    }

    fn gradient(&self, _graph: &CompiledGraph, _inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        Err(AutodiffError::InvalidGraph {
            reason: "simd backend does not support reverse gradients yet",
        })
    }

    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        if let Ok(report) = graph.backend_support_report(BackendKind::SimdF64x4) {
            if report.can_gradient_batch() {
                return gradient_batch_simd_f64x4(graph, batch, buffer);
            }
        }
        let capabilities = self.capabilities();
        if !capabilities.supports_batch_gradient {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support batch gradients on this target",
            });
        }
        graph.validate_backend_capabilities(&capabilities)?;
        gradient_batch_simd_f64x2(graph, batch, buffer)
    }
}

fn simd_unsupported_opcode_error() -> AutodiffError {
    AutodiffError::InvalidGraph {
        reason: "simd backend does not support a required opcode",
    }
}

fn simd_scalar_value(opcode: OpCode, args: &[f64]) -> Result<f64> {
    let op = opcode
        .to_multi_ad()
        .ok_or_else(simd_unsupported_opcode_error)?;
    op_rules::forward_value(op, args)
}

fn simd_scalar_first_derivatives(opcode: OpCode, args: &[f64], value: f64) -> Result<Vec<f64>> {
    let op = opcode
        .to_multi_ad()
        .ok_or_else(simd_unsupported_opcode_error)?;
    op_rules::first_derivatives(op, args, value)
}

fn append_scalar_compute_tail(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    start_row: usize,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    let mut workspace = graph.workspace();
    for row_index in start_row..batch.batch_size {
        let outputs =
            graph.compute_many_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
        buffer.data.extend_from_slice(outputs);
    }
    Ok(())
}

fn append_scalar_gradient_tail(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    start_row: usize,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    let mut workspace = graph.workspace();
    for row_index in start_row..batch.batch_size {
        let (value, gradient) =
            graph.gradient_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
        buffer.values.push(value);
        buffer.gradients.extend_from_slice(gradient);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn checked_m128_lane(values: &[__m128d], index: NodeId) -> Result<__m128d> {
    values
        .get(index)
        .copied()
        .ok_or(AutodiffError::IndexOutOfBounds {
            index,
            max_index: values.len().saturating_sub(1),
        })
}

#[cfg(target_arch = "x86_64")]
unsafe fn add_m128_cotangent(
    cotangents: &mut [__m128d],
    index: NodeId,
    contribution: __m128d,
) -> Result<()> {
    use std::arch::x86_64::_mm_add_pd;

    let max_index = cotangents.len().saturating_sub(1);
    let target = cotangents
        .get_mut(index)
        .ok_or(AutodiffError::IndexOutOfBounds { index, max_index })?;
    *target = _mm_add_pd(*target, contribution);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn active_m128_contribution(current: __m128d, contribution: __m128d) -> __m128d {
    use std::arch::x86_64::{_mm_and_pd, _mm_cmpneq_pd, _mm_setzero_pd};

    let active = _mm_cmpneq_pd(current, _mm_setzero_pd());
    _mm_and_pd(contribution, active)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_unary(input: __m128d, opcode: OpCode) -> Result<__m128d> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut stored = [0.0_f64; 2];
    _mm_storeu_pd(stored.as_mut_ptr(), input);
    let first = simd_scalar_value(opcode, &[stored[0]])?;
    let second = simd_scalar_value(opcode, &[stored[1]])?;
    Ok(_mm_set_pd(second, first))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_unary_derivative(
    input: __m128d,
    output: __m128d,
    opcode: OpCode,
) -> Result<__m128d> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut input_values = [0.0_f64; 2];
    let mut output_values = [0.0_f64; 2];
    _mm_storeu_pd(input_values.as_mut_ptr(), input);
    _mm_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(opcode, &[input_values[0]], output_values[0])?[0];
    let second = simd_scalar_first_derivatives(opcode, &[input_values[1]], output_values[1])?[0];
    Ok(_mm_set_pd(second, first))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_binary(
    left: __m128d,
    right: __m128d,
    opcode: OpCode,
) -> Result<__m128d> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut left_values = [0.0_f64; 2];
    let mut right_values = [0.0_f64; 2];
    _mm_storeu_pd(left_values.as_mut_ptr(), left);
    _mm_storeu_pd(right_values.as_mut_ptr(), right);
    let first = simd_scalar_value(opcode, &[left_values[0], right_values[0]])?;
    let second = simd_scalar_value(opcode, &[left_values[1], right_values[1]])?;
    Ok(_mm_set_pd(second, first))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_scalar_binary_derivatives(
    left: __m128d,
    right: __m128d,
    output: __m128d,
    opcode: OpCode,
) -> Result<(__m128d, __m128d)> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    let mut left_values = [0.0_f64; 2];
    let mut right_values = [0.0_f64; 2];
    let mut output_values = [0.0_f64; 2];
    _mm_storeu_pd(left_values.as_mut_ptr(), left);
    _mm_storeu_pd(right_values.as_mut_ptr(), right);
    _mm_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(
        opcode,
        &[left_values[0], right_values[0]],
        output_values[0],
    )?;
    let second = simd_scalar_first_derivatives(
        opcode,
        &[left_values[1], right_values[1]],
        output_values[1],
    )?;
    Ok((
        _mm_set_pd(second[0], first[0]),
        _mm_set_pd(second[1], first[1]),
    ))
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_f64x2_forward_values(
    flat: &[FlatInstruction],
    values: &mut Vec<__m128d>,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm_add_pd, _mm_and_pd, _mm_andnot_pd, _mm_cmpgt_pd, _mm_div_pd, _mm_mul_pd, _mm_set1_pd,
        _mm_setzero_pd, _mm_sqrt_pd, _mm_sub_pd,
    };

    for instruction in flat {
        let value = match instruction.opcode {
            OpCode::Constant => _mm_set1_pd(instruction.value),
            OpCode::Add => _mm_add_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Sub => _mm_sub_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Mul => _mm_mul_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Div => _mm_div_pd(
                checked_m128_lane(values, instruction.left)?,
                checked_m128_lane(values, instruction.right)?,
            ),
            OpCode::Pow | OpCode::LogAddExp => {
                let left = checked_m128_lane(values, instruction.left)?;
                let right = checked_m128_lane(values, instruction.right)?;
                simd_f64x2_scalar_binary(left, right, instruction.opcode)?
            }
            OpCode::Neg => _mm_sub_pd(
                _mm_setzero_pd(),
                checked_m128_lane(values, instruction.left)?,
            ),
            OpCode::Sqrt => _mm_sqrt_pd(checked_m128_lane(values, instruction.left)?),
            OpCode::Relu => {
                let input = checked_m128_lane(values, instruction.left)?;
                let mask = _mm_cmpgt_pd(input, _mm_setzero_pd());
                _mm_and_pd(input, mask)
            }
            OpCode::Abs => {
                let input = checked_m128_lane(values, instruction.left)?;
                _mm_andnot_pd(_mm_set1_pd(-0.0), input)
            }
            OpCode::Sin
            | OpCode::Cos
            | OpCode::Tan
            | OpCode::Tanh
            | OpCode::Log1pExp
            | OpCode::Exp
            | OpCode::Ln => {
                let input = checked_m128_lane(values, instruction.left)?;
                simd_f64x2_scalar_unary(input, instruction.opcode)?
            }
        };
        values.push(value);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn checked_m256_lane(values: &[__m256d], index: NodeId) -> Result<__m256d> {
    values
        .get(index)
        .copied()
        .ok_or(AutodiffError::IndexOutOfBounds {
            index,
            max_index: values.len().saturating_sub(1),
        })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_m256_cotangent(
    cotangents: &mut [__m256d],
    index: NodeId,
    contribution: __m256d,
) -> Result<()> {
    use std::arch::x86_64::_mm256_add_pd;

    let max_index = cotangents.len().saturating_sub(1);
    let target = cotangents
        .get_mut(index)
        .ok_or(AutodiffError::IndexOutOfBounds { index, max_index })?;
    *target = _mm256_add_pd(*target, contribution);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn active_m256_contribution(current: __m256d, contribution: __m256d) -> __m256d {
    use std::arch::x86_64::{_mm256_and_pd, _mm256_cmp_pd, _mm256_setzero_pd, _CMP_NEQ_UQ};

    let active = _mm256_cmp_pd(current, _mm256_setzero_pd(), _CMP_NEQ_UQ);
    _mm256_and_pd(contribution, active)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_unary(input: __m256d, opcode: OpCode) -> Result<__m256d> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut stored = [0.0_f64; 4];
    _mm256_storeu_pd(stored.as_mut_ptr(), input);
    let first = simd_scalar_value(opcode, &[stored[0]])?;
    let second = simd_scalar_value(opcode, &[stored[1]])?;
    let third = simd_scalar_value(opcode, &[stored[2]])?;
    let fourth = simd_scalar_value(opcode, &[stored[3]])?;
    Ok(_mm256_set_pd(fourth, third, second, first))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_unary_derivative(
    input: __m256d,
    output: __m256d,
    opcode: OpCode,
) -> Result<__m256d> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut input_values = [0.0_f64; 4];
    let mut output_values = [0.0_f64; 4];
    _mm256_storeu_pd(input_values.as_mut_ptr(), input);
    _mm256_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(opcode, &[input_values[0]], output_values[0])?[0];
    let second = simd_scalar_first_derivatives(opcode, &[input_values[1]], output_values[1])?[0];
    let third = simd_scalar_first_derivatives(opcode, &[input_values[2]], output_values[2])?[0];
    let fourth = simd_scalar_first_derivatives(opcode, &[input_values[3]], output_values[3])?[0];
    Ok(_mm256_set_pd(fourth, third, second, first))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_binary(
    left: __m256d,
    right: __m256d,
    opcode: OpCode,
) -> Result<__m256d> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut left_values = [0.0_f64; 4];
    let mut right_values = [0.0_f64; 4];
    _mm256_storeu_pd(left_values.as_mut_ptr(), left);
    _mm256_storeu_pd(right_values.as_mut_ptr(), right);
    let first = simd_scalar_value(opcode, &[left_values[0], right_values[0]])?;
    let second = simd_scalar_value(opcode, &[left_values[1], right_values[1]])?;
    let third = simd_scalar_value(opcode, &[left_values[2], right_values[2]])?;
    let fourth = simd_scalar_value(opcode, &[left_values[3], right_values[3]])?;
    Ok(_mm256_set_pd(fourth, third, second, first))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_scalar_binary_derivatives(
    left: __m256d,
    right: __m256d,
    output: __m256d,
    opcode: OpCode,
) -> Result<(__m256d, __m256d)> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    let mut left_values = [0.0_f64; 4];
    let mut right_values = [0.0_f64; 4];
    let mut output_values = [0.0_f64; 4];
    _mm256_storeu_pd(left_values.as_mut_ptr(), left);
    _mm256_storeu_pd(right_values.as_mut_ptr(), right);
    _mm256_storeu_pd(output_values.as_mut_ptr(), output);
    let first = simd_scalar_first_derivatives(
        opcode,
        &[left_values[0], right_values[0]],
        output_values[0],
    )?;
    let second = simd_scalar_first_derivatives(
        opcode,
        &[left_values[1], right_values[1]],
        output_values[1],
    )?;
    let third = simd_scalar_first_derivatives(
        opcode,
        &[left_values[2], right_values[2]],
        output_values[2],
    )?;
    let fourth = simd_scalar_first_derivatives(
        opcode,
        &[left_values[3], right_values[3]],
        output_values[3],
    )?;
    Ok((
        _mm256_set_pd(fourth[0], third[0], second[0], first[0]),
        _mm256_set_pd(fourth[1], third[1], second[1], first[1]),
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_f64x4_forward_values(
    flat: &[FlatInstruction],
    values: &mut Vec<__m256d>,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_andnot_pd, _mm256_cmp_pd, _mm256_div_pd,
        _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_sqrt_pd, _mm256_sub_pd,
        _CMP_GT_OQ,
    };

    for instruction in flat {
        let value = match instruction.opcode {
            OpCode::Constant => _mm256_set1_pd(instruction.value),
            OpCode::Add => _mm256_add_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Sub => _mm256_sub_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Mul => _mm256_mul_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Div => _mm256_div_pd(
                checked_m256_lane(values, instruction.left)?,
                checked_m256_lane(values, instruction.right)?,
            ),
            OpCode::Pow | OpCode::LogAddExp => {
                let left = checked_m256_lane(values, instruction.left)?;
                let right = checked_m256_lane(values, instruction.right)?;
                simd_f64x4_scalar_binary(left, right, instruction.opcode)?
            }
            OpCode::Neg => _mm256_sub_pd(
                _mm256_setzero_pd(),
                checked_m256_lane(values, instruction.left)?,
            ),
            OpCode::Sqrt => _mm256_sqrt_pd(checked_m256_lane(values, instruction.left)?),
            OpCode::Relu => {
                let input = checked_m256_lane(values, instruction.left)?;
                let mask = _mm256_cmp_pd(input, _mm256_setzero_pd(), _CMP_GT_OQ);
                _mm256_and_pd(input, mask)
            }
            OpCode::Abs => {
                let input = checked_m256_lane(values, instruction.left)?;
                _mm256_andnot_pd(_mm256_set1_pd(-0.0), input)
            }
            OpCode::Sin
            | OpCode::Cos
            | OpCode::Tan
            | OpCode::Tanh
            | OpCode::Log1pExp
            | OpCode::Exp
            | OpCode::Ln => {
                let input = checked_m256_lane(values, instruction.left)?;
                simd_f64x4_scalar_unary(input, instruction.opcode)?
            }
        };
        values.push(value);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn compute_batch_simd_f64x2(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    use std::arch::x86_64::{_mm_set_pd, _mm_storeu_pd};

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    let output_dim = graph.output_nodes.len();
    let value_count = graph.num_inputs + graph.instructions.len();
    buffer.reset(batch.batch_size, output_dim);

    let mut values: Vec<__m128d> = Vec::with_capacity(value_count);
    let mut pair_outputs: Vec<[f64; 2]> = Vec::with_capacity(output_dim);
    let mut row_index = 0;
    while row_index + 1 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            unsafe {
                values.push(_mm_set_pd(second[input_index], first[input_index]));
            }
        }

        unsafe {
            simd_f64x2_forward_values(flat, &mut values)?;
        }

        pair_outputs.clear();
        for &output in &graph.output_nodes {
            let lane = checked_m128_lane(&values, output)?;
            let mut stored = [0.0_f64; 2];
            unsafe {
                _mm_storeu_pd(stored.as_mut_ptr(), lane);
            }
            pair_outputs.push(stored);
        }
        for output in &pair_outputs {
            buffer.data.push(output[0]);
        }
        for output in &pair_outputs {
            buffer.data.push(output[1]);
        }

        row_index += 2;
    }

    append_scalar_compute_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn compute_batch_simd_f64x4_impl(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    use std::arch::x86_64::{_mm256_set_pd, _mm256_storeu_pd};

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    let output_dim = graph.output_nodes.len();
    let value_count = graph.num_inputs + graph.instructions.len();
    buffer.reset(batch.batch_size, output_dim);

    let mut values: Vec<__m256d> = Vec::with_capacity(value_count);
    let mut quad_outputs: Vec<[f64; 4]> = Vec::with_capacity(output_dim);
    let mut row_index = 0;
    while row_index + 3 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        let third = batch.try_row(row_index + 2)?;
        let fourth = batch.try_row(row_index + 3)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            values.push(_mm256_set_pd(
                fourth[input_index],
                third[input_index],
                second[input_index],
                first[input_index],
            ));
        }

        simd_f64x4_forward_values(flat, &mut values)?;

        quad_outputs.clear();
        for &output in &graph.output_nodes {
            let lane = checked_m256_lane(&values, output)?;
            let mut stored = [0.0_f64; 4];
            _mm256_storeu_pd(stored.as_mut_ptr(), lane);
            quad_outputs.push(stored);
        }
        for lane_index in 0..4 {
            for output in &quad_outputs {
                buffer.data.push(output[lane_index]);
            }
        }

        row_index += 4;
    }

    append_scalar_compute_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn compute_batch_simd_f64x4(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    if !supports_simd_f64x4_runtime() {
        return Err(AutodiffError::InvalidGraph {
            reason: "simd f64x4 backend requires x86_64 AVX support",
        });
    }
    unsafe { compute_batch_simd_f64x4_impl(graph, batch, buffer) }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn compute_batch_simd_f64x4(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd f64x4 backend requires x86_64 AVX support",
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn compute_batch_simd_f64x2(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchValuesBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd backend requires x86_64 SSE2 support",
    })
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn gradient_batch_simd_f64x2(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm_add_pd, _mm_and_pd, _mm_cmpgt_pd, _mm_cmplt_pd, _mm_div_pd, _mm_mul_pd, _mm_set1_pd,
        _mm_set_pd, _mm_setzero_pd, _mm_storeu_pd, _mm_sub_pd,
    };

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    buffer.reset(batch.batch_size, graph.num_inputs);

    let value_count = graph.num_inputs + graph.instructions.len();
    let mut values: Vec<__m128d> = Vec::with_capacity(value_count);
    let mut cotangents: Vec<__m128d> = Vec::with_capacity(value_count);
    let mut gradient_pairs = Vec::with_capacity(graph.num_inputs);
    let mut row_index = 0;
    while row_index + 1 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            unsafe {
                values.push(_mm_set_pd(second[input_index], first[input_index]));
            }
        }

        unsafe {
            simd_f64x2_forward_values(flat, &mut values)?;
        }

        let Some(&output) = graph.output_nodes.first() else {
            buffer.values.extend_from_slice(&[0.0, 0.0]);
            buffer
                .gradients
                .resize(buffer.gradients.len() + graph.num_inputs * 2, 0.0);
            row_index += 2;
            continue;
        };

        cotangents.clear();
        cotangents.resize_with(value_count, || unsafe { _mm_setzero_pd() });
        let max_index = cotangents.len().saturating_sub(1);
        *cotangents
            .get_mut(output)
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index,
            })? = unsafe { _mm_set1_pd(1.0) };

        for instruction in flat.iter().rev() {
            let current = checked_m128_lane(&cotangents, instruction.output)?;
            unsafe {
                match instruction.opcode {
                    OpCode::Constant => {}
                    OpCode::Add => {
                        let contribution = active_m128_contribution(current, current);
                        add_m128_cotangent(&mut cotangents, instruction.left, contribution)?;
                        add_m128_cotangent(&mut cotangents, instruction.right, contribution)?;
                    }
                    OpCode::Sub => {
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, current),
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(
                                current,
                                _mm_sub_pd(_mm_setzero_pd(), current),
                            ),
                        )?;
                    }
                    OpCode::Mul => {
                        let left = checked_m128_lane(&values, instruction.left)?;
                        let right = checked_m128_lane(&values, instruction.right)?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, right)),
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(current, _mm_mul_pd(current, left)),
                        )?;
                    }
                    OpCode::Div => {
                        let left = checked_m128_lane(&values, instruction.left)?;
                        let right = checked_m128_lane(&values, instruction.right)?;
                        let right_squared = _mm_mul_pd(right, right);
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_div_pd(current, right)),
                        )?;
                        let right_contribution = _mm_sub_pd(
                            _mm_setzero_pd(),
                            _mm_div_pd(_mm_mul_pd(current, left), right_squared),
                        );
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(current, right_contribution),
                        )?;
                    }
                    OpCode::Pow | OpCode::LogAddExp => {
                        let left = checked_m128_lane(&values, instruction.left)?;
                        let right = checked_m128_lane(&values, instruction.right)?;
                        let output_value = checked_m128_lane(&values, instruction.output)?;
                        let (left_derivative, right_derivative) =
                            simd_f64x2_scalar_binary_derivatives(
                                left,
                                right,
                                output_value,
                                instruction.opcode,
                            )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, left_derivative)),
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.right,
                            active_m128_contribution(
                                current,
                                _mm_mul_pd(current, right_derivative),
                            ),
                        )?;
                    }
                    OpCode::Neg => {
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(
                                current,
                                _mm_sub_pd(_mm_setzero_pd(), current),
                            ),
                        )?;
                    }
                    OpCode::Sqrt => {
                        let output_value = checked_m128_lane(&values, instruction.output)?;
                        let contribution =
                            _mm_div_pd(_mm_mul_pd(current, _mm_set1_pd(0.5)), output_value);
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, contribution),
                        )?;
                    }
                    OpCode::Relu => {
                        let input_value = checked_m128_lane(&values, instruction.left)?;
                        let mask = _mm_cmpgt_pd(input_value, _mm_setzero_pd());
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_and_pd(current, mask)),
                        )?;
                    }
                    OpCode::Sin
                    | OpCode::Cos
                    | OpCode::Tan
                    | OpCode::Tanh
                    | OpCode::Log1pExp
                    | OpCode::Exp
                    | OpCode::Ln => {
                        let input_value = checked_m128_lane(&values, instruction.left)?;
                        let output_value = checked_m128_lane(&values, instruction.output)?;
                        let derivative = simd_f64x2_scalar_unary_derivative(
                            input_value,
                            output_value,
                            instruction.opcode,
                        )?;
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, derivative)),
                        )?;
                    }
                    OpCode::Abs => {
                        let input_value = checked_m128_lane(&values, instruction.left)?;
                        let positive = _mm_cmpgt_pd(input_value, _mm_setzero_pd());
                        let negative = _mm_cmplt_pd(input_value, _mm_setzero_pd());
                        let sign = _mm_add_pd(
                            _mm_and_pd(_mm_set1_pd(1.0), positive),
                            _mm_and_pd(_mm_set1_pd(-1.0), negative),
                        );
                        add_m128_cotangent(
                            &mut cotangents,
                            instruction.left,
                            active_m128_contribution(current, _mm_mul_pd(current, sign)),
                        )?;
                    }
                }
            }
        }

        let output_value = checked_m128_lane(&values, output)?;
        let mut value_pair = [0.0_f64; 2];
        unsafe {
            _mm_storeu_pd(value_pair.as_mut_ptr(), output_value);
        }
        buffer.values.extend_from_slice(&value_pair);

        gradient_pairs.clear();
        for input_index in 0..graph.num_inputs {
            let gradient_lane = checked_m128_lane(&cotangents, input_index)?;
            let mut stored = [0.0_f64; 2];
            unsafe {
                _mm_storeu_pd(stored.as_mut_ptr(), gradient_lane);
            }
            gradient_pairs.push(stored);
        }
        for pair in &gradient_pairs {
            buffer.gradients.push(pair[0]);
        }
        for pair in &gradient_pairs {
            buffer.gradients.push(pair[1]);
        }

        row_index += 2;
    }

    append_scalar_gradient_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gradient_batch_simd_f64x4_impl(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_cmp_pd, _mm256_div_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_set_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_GT_OQ, _CMP_LT_OQ,
    };

    graph.check_batch(batch)?;
    let flat = graph.flat_instructions_slice();
    buffer.reset(batch.batch_size, graph.num_inputs);

    let value_count = graph.num_inputs + graph.instructions.len();
    let mut values: Vec<__m256d> = Vec::with_capacity(value_count);
    let mut cotangents: Vec<__m256d> = Vec::with_capacity(value_count);
    let mut gradient_quads = Vec::with_capacity(graph.num_inputs);
    let mut row_index = 0;
    while row_index + 3 < batch.batch_size {
        let first = batch.try_row(row_index)?;
        let second = batch.try_row(row_index + 1)?;
        let third = batch.try_row(row_index + 2)?;
        let fourth = batch.try_row(row_index + 3)?;
        values.clear();
        for input_index in 0..graph.num_inputs {
            values.push(_mm256_set_pd(
                fourth[input_index],
                third[input_index],
                second[input_index],
                first[input_index],
            ));
        }

        simd_f64x4_forward_values(flat, &mut values)?;

        let Some(&output) = graph.output_nodes.first() else {
            buffer.values.extend_from_slice(&[0.0; 4]);
            buffer
                .gradients
                .resize(buffer.gradients.len() + graph.num_inputs * 4, 0.0);
            row_index += 4;
            continue;
        };

        cotangents.clear();
        cotangents.resize_with(value_count, || _mm256_setzero_pd());
        let max_index = cotangents.len().saturating_sub(1);
        *cotangents
            .get_mut(output)
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index,
            })? = _mm256_set1_pd(1.0);

        for instruction in flat.iter().rev() {
            let current = checked_m256_lane(&cotangents, instruction.output)?;
            match instruction.opcode {
                OpCode::Constant => {}
                OpCode::Add => {
                    let contribution = active_m256_contribution(current, current);
                    add_m256_cotangent(&mut cotangents, instruction.left, contribution)?;
                    add_m256_cotangent(&mut cotangents, instruction.right, contribution)?;
                }
                OpCode::Sub => {
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, current),
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(
                            current,
                            _mm256_sub_pd(_mm256_setzero_pd(), current),
                        ),
                    )?;
                }
                OpCode::Mul => {
                    let left = checked_m256_lane(&values, instruction.left)?;
                    let right = checked_m256_lane(&values, instruction.right)?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, right)),
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(current, _mm256_mul_pd(current, left)),
                    )?;
                }
                OpCode::Div => {
                    let left = checked_m256_lane(&values, instruction.left)?;
                    let right = checked_m256_lane(&values, instruction.right)?;
                    let right_squared = _mm256_mul_pd(right, right);
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_div_pd(current, right)),
                    )?;
                    let right_contribution = _mm256_sub_pd(
                        _mm256_setzero_pd(),
                        _mm256_div_pd(_mm256_mul_pd(current, left), right_squared),
                    );
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(current, right_contribution),
                    )?;
                }
                OpCode::Pow | OpCode::LogAddExp => {
                    let left = checked_m256_lane(&values, instruction.left)?;
                    let right = checked_m256_lane(&values, instruction.right)?;
                    let output_value = checked_m256_lane(&values, instruction.output)?;
                    let (left_derivative, right_derivative) = simd_f64x4_scalar_binary_derivatives(
                        left,
                        right,
                        output_value,
                        instruction.opcode,
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, left_derivative)),
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.right,
                        active_m256_contribution(current, _mm256_mul_pd(current, right_derivative)),
                    )?;
                }
                OpCode::Neg => {
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(
                            current,
                            _mm256_sub_pd(_mm256_setzero_pd(), current),
                        ),
                    )?;
                }
                OpCode::Sqrt => {
                    let output_value = checked_m256_lane(&values, instruction.output)?;
                    let contribution =
                        _mm256_div_pd(_mm256_mul_pd(current, _mm256_set1_pd(0.5)), output_value);
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, contribution),
                    )?;
                }
                OpCode::Relu => {
                    let input_value = checked_m256_lane(&values, instruction.left)?;
                    let mask = _mm256_cmp_pd(input_value, _mm256_setzero_pd(), _CMP_GT_OQ);
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_and_pd(current, mask)),
                    )?;
                }
                OpCode::Sin
                | OpCode::Cos
                | OpCode::Tan
                | OpCode::Tanh
                | OpCode::Log1pExp
                | OpCode::Exp
                | OpCode::Ln => {
                    let input_value = checked_m256_lane(&values, instruction.left)?;
                    let output_value = checked_m256_lane(&values, instruction.output)?;
                    let derivative = simd_f64x4_scalar_unary_derivative(
                        input_value,
                        output_value,
                        instruction.opcode,
                    )?;
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, derivative)),
                    )?;
                }
                OpCode::Abs => {
                    let input_value = checked_m256_lane(&values, instruction.left)?;
                    let positive = _mm256_cmp_pd(input_value, _mm256_setzero_pd(), _CMP_GT_OQ);
                    let negative = _mm256_cmp_pd(input_value, _mm256_setzero_pd(), _CMP_LT_OQ);
                    let sign = _mm256_add_pd(
                        _mm256_and_pd(_mm256_set1_pd(1.0), positive),
                        _mm256_and_pd(_mm256_set1_pd(-1.0), negative),
                    );
                    add_m256_cotangent(
                        &mut cotangents,
                        instruction.left,
                        active_m256_contribution(current, _mm256_mul_pd(current, sign)),
                    )?;
                }
            }
        }

        let output_value = checked_m256_lane(&values, output)?;
        let mut value_quad = [0.0_f64; 4];
        _mm256_storeu_pd(value_quad.as_mut_ptr(), output_value);
        buffer.values.extend_from_slice(&value_quad);

        gradient_quads.clear();
        for input_index in 0..graph.num_inputs {
            let gradient_lane = checked_m256_lane(&cotangents, input_index)?;
            let mut stored = [0.0_f64; 4];
            _mm256_storeu_pd(stored.as_mut_ptr(), gradient_lane);
            gradient_quads.push(stored);
        }
        for lane_index in 0..4 {
            for quad in &gradient_quads {
                buffer.gradients.push(quad[lane_index]);
            }
        }

        row_index += 4;
    }

    append_scalar_gradient_tail(graph, batch, row_index, buffer)?;

    Ok(())
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn gradient_batch_simd_f64x4(
    graph: &CompiledGraph,
    batch: BatchInputs<'_>,
    buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    if !supports_simd_f64x4_runtime() {
        return Err(AutodiffError::InvalidGraph {
            reason: "simd f64x4 backend requires x86_64 AVX support",
        });
    }
    unsafe { gradient_batch_simd_f64x4_impl(graph, batch, buffer) }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn gradient_batch_simd_f64x4(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd f64x4 backend requires x86_64 AVX support",
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn gradient_batch_simd_f64x2(
    _graph: &CompiledGraph,
    _batch: BatchInputs<'_>,
    _buffer: &mut BatchGradientsBuffer,
) -> Result<()> {
    Err(AutodiffError::InvalidGraph {
        reason: "simd backend requires x86_64 SSE2 support",
    })
}
