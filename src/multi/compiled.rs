//! Closure-free compiled instruction IR for acceleration-ready graph execution.

use super::multi_ad::MultiAD;
use super::op_rules;
use crate::{AutodiffError, NodeId, Result};

use super::backend::{
    BackendCapabilities, BackendKind, BackendRejectionReason, BackendSupportReport, DeviceBackend,
    DeviceBatchPlan, DeviceBufferKind, DeviceBufferSet, DeviceExecutionTrace, DeviceMemoryLocation,
    ExecutionBackend, FlatInstruction, Instruction, MockDeviceBackend, OpCode, ScalarBackend,
    SimdBackend, UNUSED_NODE_ID,
};
#[cfg(feature = "backend-wgpu")]
use super::backend::{WgpuBackend, WgpuBufferSet};

/// Flat row-major batch input view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchInputs<'a> {
    /// Flat row-major input data.
    pub data: &'a [f64],
    /// Number of rows in the batch.
    pub batch_size: usize,
    /// Number of inputs per row.
    pub input_dim: usize,
}

impl<'a> BatchInputs<'a> {
    /// Create a validated row-major input view.
    pub fn new(data: &'a [f64], batch_size: usize, input_dim: usize) -> Result<Self> {
        if data.len() != batch_size.saturating_mul(input_dim) {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch data length must equal batch_size * input_dim",
            });
        }
        Ok(Self {
            data,
            batch_size,
            input_dim,
        })
    }

    /// Return one input row.
    ///
    /// Prefer [`BatchInputs::try_row`] when invalid row indices should return an error.
    #[must_use]
    pub fn row(&self, index: usize) -> &[f64] {
        self.try_row(index).expect("batch row index out of bounds")
    }

    /// Return one input row, or an error when `index` is out of range.
    pub fn try_row(&self, index: usize) -> Result<&[f64]> {
        if index >= self.batch_size {
            return Err(AutodiffError::IndexOutOfBounds {
                index,
                max_index: self.batch_size.saturating_sub(1),
            });
        }
        let start = index * self.input_dim;
        Ok(&self.data[start..start + self.input_dim])
    }
}

/// Flat row-major batch values.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchValues {
    pub data: Vec<f64>,
    pub batch_size: usize,
    pub output_dim: usize,
}

/// Reusable flat row-major batch value buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchValuesBuffer {
    pub data: Vec<f64>,
    pub batch_size: usize,
    pub output_dim: usize,
}

impl BatchValuesBuffer {
    /// Create an empty reusable output buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self, batch_size: usize, output_dim: usize) {
        self.data.clear();
        self.data.reserve(batch_size.saturating_mul(output_dim));
        self.batch_size = batch_size;
        self.output_dim = output_dim;
    }

    /// Clone the current buffer contents into an owned result value.
    #[must_use]
    pub fn to_values(&self) -> BatchValues {
        BatchValues {
            data: self.data.clone(),
            batch_size: self.batch_size,
            output_dim: self.output_dim,
        }
    }
}

/// Flat row-major batch scalar-output gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchGradients {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
    pub batch_size: usize,
    pub input_dim: usize,
}

/// Reusable flat row-major batch scalar-output gradient buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchGradientsBuffer {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
    pub batch_size: usize,
    pub input_dim: usize,
}

impl BatchGradientsBuffer {
    /// Create an empty reusable gradient buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self, batch_size: usize, input_dim: usize) {
        self.values.clear();
        self.gradients.clear();
        self.values.reserve(batch_size);
        self.gradients.reserve(batch_size.saturating_mul(input_dim));
        self.batch_size = batch_size;
        self.input_dim = input_dim;
    }

    /// Clone the current buffer contents into an owned result value.
    #[must_use]
    pub fn to_gradients(&self) -> BatchGradients {
        BatchGradients {
            values: self.values.clone(),
            gradients: self.gradients.clone(),
            batch_size: self.batch_size,
            input_dim: self.input_dim,
        }
    }
}

/// Static metadata for backend selection and workspace planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledGraphMetadata {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_instructions: usize,
    pub num_constants: usize,
    pub num_unary: usize,
    pub num_binary: usize,
    pub value_count: usize,
    pub is_scalar_output: bool,
}

/// Reusable buffers for [`CompiledGraph`] execution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompiledWorkspace {
    values: Vec<f64>,
    cotangents: Vec<f64>,
    gradients: Vec<f64>,
    outputs: Vec<f64>,
}

/// Closure-free compiled graph representation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGraph {
    pub(crate) num_inputs: usize,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) output_nodes: Vec<NodeId>,
    flat_instructions: Vec<FlatInstruction>,
}

impl CompiledGraph {
    pub(crate) fn new(
        num_inputs: usize,
        instructions: Vec<Instruction>,
        output_nodes: Vec<NodeId>,
    ) -> Result<Self> {
        let flat_instructions = Self::lower_flat_instructions(num_inputs, &instructions)?;
        Ok(Self {
            num_inputs,
            instructions,
            output_nodes,
            flat_instructions,
        })
    }

    /// Return number of graph inputs.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Return compiled instructions.
    #[must_use]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Return selected output nodes.
    #[must_use]
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }

    fn lower_flat_instructions(
        num_inputs: usize,
        instructions: &[Instruction],
    ) -> Result<Vec<FlatInstruction>> {
        let mut flat = Vec::with_capacity(instructions.len());
        for (offset, instruction) in instructions.iter().enumerate() {
            let output = num_inputs + offset;
            let item = match *instruction {
                Instruction::Constant(value) => FlatInstruction {
                    opcode: OpCode::Constant,
                    output,
                    left: UNUSED_NODE_ID,
                    right: UNUSED_NODE_ID,
                    value,
                },
                Instruction::Unary { op, arg } => FlatInstruction {
                    opcode: OpCode::from_multi_ad(op)?,
                    output,
                    left: arg,
                    right: UNUSED_NODE_ID,
                    value: 0.0,
                },
                Instruction::Binary { op, left, right } => FlatInstruction {
                    opcode: OpCode::from_multi_ad(op)?,
                    output,
                    left,
                    right,
                    value: 0.0,
                },
            };
            flat.push(item);
        }
        Ok(flat)
    }

    /// Return a zero-copy view of flat instructions with explicit output and argument slots.
    #[must_use]
    pub fn flat_instructions_slice(&self) -> &[FlatInstruction] {
        &self.flat_instructions
    }

    /// Return owned flat instructions with explicit output and argument slots.
    pub fn flat_instructions(&self) -> Result<Vec<FlatInstruction>> {
        Ok(self.flat_instructions_slice().to_vec())
    }

    /// Validate whether static graph requirements fit backend capabilities.
    pub fn validate_backend_capabilities(&self, capabilities: &BackendCapabilities) -> Result<()> {
        if !capabilities.supports_f64 {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend must support f64 execution",
            });
        }
        if self.output_nodes.len() > 1 && !capabilities.supports_multi_output {
            return Err(AutodiffError::InvalidGraph {
                reason: "backend does not support multi-output graphs",
            });
        }
        for instruction in self.flat_instructions_slice() {
            if !capabilities.supports_opcode(instruction.opcode) {
                return Err(AutodiffError::InvalidGraph {
                    reason: "backend does not support a required opcode",
                });
            }
            match instruction.opcode.arity() {
                0 if !capabilities.supports_constants => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "backend does not support constants",
                    });
                }
                1 if !capabilities.supports_unary => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "backend does not support unary operations",
                    });
                }
                2 if !capabilities.supports_binary => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "backend does not support binary operations",
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Return static execution metadata for backend planning.
    #[must_use]
    pub fn metadata(&self) -> CompiledGraphMetadata {
        let mut num_constants = 0;
        let mut num_unary = 0;
        let mut num_binary = 0;
        for instruction in &self.instructions {
            match instruction {
                Instruction::Constant(_) => num_constants += 1,
                Instruction::Unary { .. } => num_unary += 1,
                Instruction::Binary { .. } => num_binary += 1,
            }
        }
        CompiledGraphMetadata {
            num_inputs: self.num_inputs,
            num_outputs: self.output_nodes.len(),
            num_instructions: self.instructions.len(),
            num_constants,
            num_unary,
            num_binary,
            value_count: self.num_inputs + self.instructions.len(),
            is_scalar_output: self.output_nodes.len() == 1,
        }
    }

    /// Create a reusable workspace sized for this compiled graph.
    #[must_use]
    pub fn workspace(&self) -> CompiledWorkspace {
        let value_count = self.num_inputs + self.instructions.len();
        CompiledWorkspace {
            values: Vec::with_capacity(value_count),
            cotangents: Vec::with_capacity(value_count),
            gradients: Vec::with_capacity(self.num_inputs),
            outputs: Vec::with_capacity(self.output_nodes.len()),
        }
    }

    pub(crate) fn check_input_len(&self, inputs: &[f64]) -> Result<()> {
        if inputs.len() == self.num_inputs {
            Ok(())
        } else {
            Err(AutodiffError::InvalidGraph {
                reason: "input length must match compiled graph input count",
            })
        }
    }

    pub(crate) fn check_batch(&self, batch: BatchInputs<'_>) -> Result<()> {
        if batch.input_dim == self.num_inputs {
            Ok(())
        } else {
            Err(AutodiffError::InvalidGraph {
                reason: "batch input_dim must match compiled graph input count",
            })
        }
    }

    fn gather_unary(values: &[f64], arg: NodeId) -> Result<[f64; 1]> {
        let value = values
            .get(arg)
            .copied()
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: arg,
                max_index: values.len().saturating_sub(1),
            })?;
        Ok([value])
    }

    fn gather_binary(values: &[f64], left: NodeId, right: NodeId) -> Result<[f64; 2]> {
        let left_value = values
            .get(left)
            .copied()
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: left,
                max_index: values.len().saturating_sub(1),
            })?;
        let right_value = values
            .get(right)
            .copied()
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: right,
                max_index: values.len().saturating_sub(1),
            })?;
        Ok([left_value, right_value])
    }

    pub(crate) fn fill_values(
        &self,
        inputs: &[f64],
        workspace: &mut CompiledWorkspace,
    ) -> Result<()> {
        self.check_input_len(inputs)?;
        workspace.values.clear();
        workspace
            .values
            .reserve(self.num_inputs + self.instructions.len());
        workspace.values.extend_from_slice(inputs);

        for instruction in &self.instructions {
            let value = match *instruction {
                Instruction::Constant(value) => value,
                Instruction::Unary { op, arg } => {
                    let args = Self::gather_unary(&workspace.values, arg)?;
                    op_rules::forward_value(op, &args)?
                }
                Instruction::Binary { op, left, right } => {
                    let args = Self::gather_binary(&workspace.values, left, right)?;
                    op_rules::forward_value(op, &args)?
                }
            };
            workspace.values.push(value);
        }
        Ok(())
    }

    /// Compute primary output.
    pub fn compute(&self, inputs: &[f64]) -> Result<f64> {
        let mut workspace = self.workspace();
        self.compute_with_workspace(inputs, &mut workspace)
    }

    /// Compute all selected outputs.
    pub fn compute_many(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        let mut workspace = self.workspace();
        Ok(self
            .compute_many_with_workspace(inputs, &mut workspace)?
            .to_vec())
    }

    /// Compute primary output with a reusable workspace.
    pub fn compute_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut CompiledWorkspace,
    ) -> Result<f64> {
        let outputs = self.compute_many_with_workspace(inputs, workspace)?;
        Ok(outputs.first().copied().unwrap_or(0.0))
    }

    /// Compute all selected outputs with a reusable workspace.
    pub fn compute_many_with_workspace<'a>(
        &self,
        inputs: &[f64],
        workspace: &'a mut CompiledWorkspace,
    ) -> Result<&'a [f64]> {
        self.fill_values(inputs, workspace)?;
        workspace.outputs.clear();
        for &output in &self.output_nodes {
            let value =
                workspace
                    .values
                    .get(output)
                    .copied()
                    .ok_or(AutodiffError::IndexOutOfBounds {
                        index: output,
                        max_index: workspace.values.len().saturating_sub(1),
                    })?;
            workspace.outputs.push(value);
        }
        Ok(&workspace.outputs)
    }

    fn reverse_for_output(&self, output: NodeId, workspace: &mut CompiledWorkspace) -> Result<()> {
        workspace.cotangents.clear();
        workspace.cotangents.resize(workspace.values.len(), 0.0);
        if output >= workspace.cotangents.len() {
            return Err(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index: workspace.cotangents.len().saturating_sub(1),
            });
        }
        workspace.cotangents[output] = 1.0;

        for (offset, instruction) in self.instructions.iter().enumerate().rev() {
            let node_id = self.num_inputs + offset;
            let current = workspace.cotangents[node_id];
            if current == 0.0 {
                continue;
            }
            match *instruction {
                Instruction::Constant(_) => {}
                Instruction::Unary { op, arg } => {
                    let args = Self::gather_unary(&workspace.values, arg)?;
                    let value = workspace.values[node_id];
                    if let op_rules::LocalRule::Unary { dy, .. } =
                        op_rules::local_rule(op, &args, value)?
                    {
                        workspace.cotangents[arg] += current * dy;
                    }
                }
                Instruction::Binary { op, left, right } => {
                    let args = Self::gather_binary(&workspace.values, left, right)?;
                    let value = workspace.values[node_id];
                    if let op_rules::LocalRule::Binary {
                        dy_left, dy_right, ..
                    } = op_rules::local_rule(op, &args, value)?
                    {
                        workspace.cotangents[left] += current * dy_left;
                        workspace.cotangents[right] += current * dy_right;
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute primary output and gradient.
    pub fn gradient(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let mut workspace = self.workspace();
        let (value, gradient) = self.gradient_with_workspace(inputs, &mut workspace)?;
        Ok((value, gradient.to_vec()))
    }

    /// Compute primary output and gradient with a reusable workspace.
    pub fn gradient_with_workspace<'a>(
        &self,
        inputs: &[f64],
        workspace: &'a mut CompiledWorkspace,
    ) -> Result<(f64, &'a [f64])> {
        self.fill_values(inputs, workspace)?;
        workspace.gradients.clear();
        workspace.gradients.resize(self.num_inputs, 0.0);
        let Some(&output) = self.output_nodes.first() else {
            return Ok((0.0, &workspace.gradients));
        };
        self.reverse_for_output(output, workspace)?;
        workspace
            .gradients
            .copy_from_slice(&workspace.cotangents[..self.num_inputs]);
        Ok((workspace.values[output], &workspace.gradients))
    }

    /// Compute Jacobian for all selected outputs.
    pub fn jacobian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut workspace = self.workspace();
        self.jacobian_with_workspace(inputs, &mut workspace)
    }

    /// Compute Jacobian for all selected outputs with a reusable workspace.
    pub fn jacobian_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut CompiledWorkspace,
    ) -> Result<Vec<Vec<f64>>> {
        self.fill_values(inputs, workspace)?;
        let mut jacobian = Vec::with_capacity(self.output_nodes.len());
        for &output in &self.output_nodes {
            self.reverse_for_output(output, workspace)?;
            jacobian.push(workspace.cotangents[..self.num_inputs].to_vec());
        }
        Ok(jacobian)
    }

    /// Return static compatibility details for a backend.
    pub fn backend_support_report(&self, backend: BackendKind) -> Result<BackendSupportReport> {
        let capabilities = backend.capabilities();
        let supports_required_outputs =
            self.output_nodes.len() <= 1 || capabilities.supports_multi_output;
        let mut missing_opcodes = Vec::new();
        let mut supports_required_arities = true;
        for instruction in self.flat_instructions_slice() {
            if !capabilities.supports_opcode(instruction.opcode)
                && !missing_opcodes.contains(&instruction.opcode)
            {
                missing_opcodes.push(instruction.opcode);
            }
            match instruction.opcode.arity() {
                0 if !capabilities.supports_constants => supports_required_arities = false,
                1 if !capabilities.supports_unary => supports_required_arities = false,
                2 if !capabilities.supports_binary => supports_required_arities = false,
                _ => {}
            }
        }
        let runtime_available = backend.runtime_available();
        let supports_required_opcodes = missing_opcodes.is_empty();
        let mut common_reasons = Vec::new();
        if !capabilities.supports_f64 {
            common_reasons.push(BackendRejectionReason::MissingF64);
        }
        if !supports_required_outputs {
            common_reasons.push(BackendRejectionReason::UnsupportedOutputs);
        }
        if !supports_required_opcodes {
            common_reasons.push(BackendRejectionReason::UnsupportedOpcodes);
        }
        if !supports_required_arities {
            common_reasons.push(BackendRejectionReason::UnsupportedArities);
        }
        if !runtime_available {
            common_reasons.push(BackendRejectionReason::UnavailableRuntime);
        }

        let mut batch_compute_rejection_reasons = common_reasons.clone();
        if !capabilities.supports_batch_compute {
            batch_compute_rejection_reasons.push(BackendRejectionReason::NoBatchCompute);
        }
        let mut batch_gradient_rejection_reasons = common_reasons;
        if !capabilities.supports_batch_gradient {
            batch_gradient_rejection_reasons.push(BackendRejectionReason::NoBatchGradient);
        }

        Ok(BackendSupportReport {
            backend,
            supports_f64: capabilities.supports_f64,
            supports_required_outputs,
            supports_required_opcodes,
            supports_required_arities,
            supports_batch_compute: capabilities.supports_batch_compute,
            supports_batch_gradient: capabilities.supports_batch_gradient,
            lane_width: backend.lane_width(),
            runtime_available,
            required_runtime_features: backend.required_runtime_features().to_vec(),
            unavailable_runtime_features: backend.unavailable_runtime_features(),
            missing_opcodes,
            batch_compute_rejection_reasons,
            batch_gradient_rejection_reasons,
        })
    }

    /// Return static compatibility details for all built-in backends.
    pub fn backend_support_reports(&self) -> Result<Vec<BackendSupportReport>> {
        Ok(vec![
            self.backend_support_report(BackendKind::Scalar)?,
            self.backend_support_report(BackendKind::MockDeviceCpu)?,
            self.backend_support_report(BackendKind::Wgpu)?,
            self.backend_support_report(BackendKind::SimdF64x4)?,
            self.backend_support_report(BackendKind::SimdF64x2)?,
        ])
    }

    /// Return a device-oriented batch buffer plan for a backend.
    #[must_use]
    pub fn device_batch_plan(&self, backend: BackendKind, batch_size: usize) -> DeviceBatchPlan {
        backend.batch_plan(self, batch_size)
    }

    /// Allocate mock-device buffers for this compiled graph and batch size.
    #[must_use]
    pub fn allocate_mock_device_buffers(&self, batch_size: usize) -> DeviceBufferSet {
        MockDeviceBackend.allocate_batch_buffers(self, batch_size)
    }

    /// Execute batch value computation through mock-device buffers.
    pub fn compute_batch_mock_device_into(
        &self,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        MockDeviceBackend.compute_batch_with_buffers(self, batch, buffers, output)
    }

    /// Execute batch gradient computation through mock-device buffers.
    pub fn gradient_batch_mock_device_into(
        &self,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        MockDeviceBackend.gradient_batch_with_buffers(self, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Allocate real WGPU buffers for this compiled graph and batch size.
    pub fn allocate_wgpu_buffers(
        &self,
        backend: &WgpuBackend,
        batch_size: usize,
    ) -> Result<WgpuBufferSet> {
        backend.allocate_batch_buffers(self, batch_size)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Execute batch value computation through real WGPU buffers.
    pub fn compute_batch_wgpu_into(
        &self,
        backend: &WgpuBackend,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        backend.compute_batch_with_buffers(self, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Execute batch gradient computation through real WGPU buffers.
    pub fn gradient_batch_wgpu_into(
        &self,
        backend: &WgpuBackend,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        backend.gradient_batch_with_buffers(self, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Return whether this graph is statically eligible for the exact-safe native WGPU compute path.
    #[must_use]
    pub fn supports_native_wgpu_batch_compute(&self, backend: &WgpuBackend) -> bool {
        backend.supports_native_batch_compute(self)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Return whether this graph and concrete batch can use the exact-safe native WGPU compute path.
    #[must_use]
    pub fn supports_native_wgpu_batch_compute_for_batch(
        &self,
        backend: &WgpuBackend,
        batch: BatchInputs<'_>,
    ) -> bool {
        backend.supports_native_batch_compute_for_batch(self, batch)
    }

    /// Return static compatibility details for the preferred SIMD backend.
    pub fn simd_support_report(&self) -> Result<BackendSupportReport> {
        let f64x4 = self.backend_support_report(BackendKind::SimdF64x4)?;
        if f64x4.supports_batch_compute || f64x4.supports_batch_gradient {
            return Ok(f64x4);
        }
        self.backend_support_report(BackendKind::SimdF64x2)
    }

    /// Return the preferred backend for batch value computation.
    #[must_use]
    pub fn recommended_batch_compute_backend(&self) -> BackendKind {
        match self.backend_support_report(BackendKind::SimdF64x4) {
            Ok(report) if report.can_compute_batch() => BackendKind::SimdF64x4,
            _ => match self.backend_support_report(BackendKind::SimdF64x2) {
                Ok(report) if report.can_compute_batch() => BackendKind::SimdF64x2,
                _ => BackendKind::Scalar,
            },
        }
    }

    /// Return the preferred backend for batch gradient computation.
    #[must_use]
    pub fn recommended_batch_gradient_backend(&self) -> BackendKind {
        match self.backend_support_report(BackendKind::SimdF64x4) {
            Ok(report) if report.can_gradient_batch() => BackendKind::SimdF64x4,
            _ => match self.backend_support_report(BackendKind::SimdF64x2) {
                Ok(report) if report.can_gradient_batch() => BackendKind::SimdF64x2,
                _ => BackendKind::Scalar,
            },
        }
    }

    /// Compute all selected outputs for a batch of rows.
    pub fn compute_batch(&self, batch: BatchInputs<'_>) -> Result<BatchValues> {
        let mut buffer = BatchValuesBuffer::new();
        self.compute_batch_into(batch, &mut buffer)?;
        Ok(BatchValues {
            data: buffer.data,
            batch_size: buffer.batch_size,
            output_dim: buffer.output_dim,
        })
    }

    /// Compute all selected outputs for a batch with automatic backend dispatch.
    pub fn compute_batch_auto(&self, batch: BatchInputs<'_>) -> Result<(BackendKind, BatchValues)> {
        let mut buffer = BatchValuesBuffer::new();
        let backend = self.compute_batch_auto_into(batch, &mut buffer)?;
        Ok((
            backend,
            BatchValues {
                data: buffer.data,
                batch_size: buffer.batch_size,
                output_dim: buffer.output_dim,
            },
        ))
    }

    /// Compute all selected outputs into a reusable output buffer.
    pub fn compute_batch_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        self.check_batch(batch)?;
        let mut workspace = self.workspace();
        let output_dim = self.output_nodes.len();
        buffer.reset(batch.batch_size, output_dim);
        for row_index in 0..batch.batch_size {
            let outputs =
                self.compute_many_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
            buffer.data.extend_from_slice(outputs);
        }
        Ok(())
    }

    /// Compute all selected outputs into a reusable buffer with automatic backend dispatch.
    pub fn compute_batch_auto_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<BackendKind> {
        let backend = self.recommended_batch_compute_backend();
        backend.compute_batch(self, batch, buffer)?;
        Ok(backend)
    }

    /// Compute primary-output values and gradients for a batch of rows.
    pub fn gradient_batch(&self, batch: BatchInputs<'_>) -> Result<BatchGradients> {
        let mut buffer = BatchGradientsBuffer::new();
        self.gradient_batch_into(batch, &mut buffer)?;
        Ok(BatchGradients {
            values: buffer.values,
            gradients: buffer.gradients,
            batch_size: buffer.batch_size,
            input_dim: buffer.input_dim,
        })
    }

    /// Compute primary-output values and gradients for a batch with automatic backend dispatch.
    pub fn gradient_batch_auto(
        &self,
        batch: BatchInputs<'_>,
    ) -> Result<(BackendKind, BatchGradients)> {
        let mut buffer = BatchGradientsBuffer::new();
        let backend = self.gradient_batch_auto_into(batch, &mut buffer)?;
        Ok((
            backend,
            BatchGradients {
                values: buffer.values,
                gradients: buffer.gradients,
                batch_size: buffer.batch_size,
                input_dim: buffer.input_dim,
            },
        ))
    }

    /// Compute primary-output values and gradients into a reusable gradient buffer.
    pub fn gradient_batch_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        self.check_batch(batch)?;
        let mut workspace = self.workspace();
        buffer.reset(batch.batch_size, self.num_inputs);
        for row_index in 0..batch.batch_size {
            let (value, gradient) =
                self.gradient_with_workspace(batch.try_row(row_index)?, &mut workspace)?;
            buffer.values.push(value);
            buffer.gradients.extend_from_slice(gradient);
        }
        Ok(())
    }

    /// Compute primary-output values and gradients into a reusable buffer with automatic backend dispatch.
    pub fn gradient_batch_auto_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<BackendKind> {
        let backend = self.recommended_batch_gradient_backend();
        backend.gradient_batch(self, batch, buffer)?;
        Ok(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::approx_eq_eps as approx_eq;

    // --- helpers ---

    fn make_simple_graph() -> CompiledGraph {
        let instructions = vec![Instruction::Binary {
            op: MultiAD::Add,
            left: 0,
            right: 1,
        }];
        CompiledGraph::new(2, instructions, vec![2]).unwrap()
    }

    fn make_multi_output_graph() -> CompiledGraph {
        // f(x) = x, outputs: [x, sin(x)]
        let instructions = vec![Instruction::Unary {
            op: MultiAD::Sin,
            arg: 0,
        }];
        CompiledGraph::new(1, instructions, vec![0, 1]).unwrap()
    }

    // ---- 1. compute / gradient input-length checks ----

    #[test]
    fn test_compiled_compute_wrong_input_length_error() {
        let graph = make_simple_graph();
        assert!(graph.compute(&[1.0]).is_err());
        assert!(graph.compute(&[1.0, 2.0, 3.0]).is_err());
        assert!(graph.compute(&[1.0, 2.0]).is_ok());
    }

    #[test]
    fn test_compiled_gradient_wrong_input_length_error() {
        let graph = make_simple_graph();
        assert!(graph.gradient(&[1.0]).is_err());
        assert!(graph.gradient(&[1.0, 2.0, 3.0]).is_err());
        assert!(graph.gradient(&[1.0, 2.0]).is_ok());
    }

    // ---- 2. batch with odd batch sizes ----

    #[test]
    fn test_compiled_compute_batch_odd_size() {
        let graph = make_simple_graph();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let batch = BatchInputs::new(&data, 3, 2).unwrap();
        let result = graph.compute_batch(batch).unwrap();
        assert_eq!(result.batch_size, 3);
        assert_eq!(result.output_dim, 1);
        assert_eq!(result.data.len(), 3);
        assert!(approx_eq(result.data[0], 3.0, 1e-10));
        assert!(approx_eq(result.data[1], 7.0, 1e-10));
        assert!(approx_eq(result.data[2], 11.0, 1e-10));
    }

    #[test]
    fn test_compiled_gradient_batch_odd_size() {
        let graph = make_simple_graph();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let batch = BatchInputs::new(&data, 3, 2).unwrap();
        let result = graph.gradient_batch(batch).unwrap();
        assert_eq!(result.batch_size, 3);
        assert_eq!(result.input_dim, 2);
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.gradients.len(), 6);
        // f(x,y) = x+y, gradients are always [1, 1]
        for i in 0..3 {
            assert!(approx_eq(result.gradients[i * 2], 1.0, 1e-10));
            assert!(approx_eq(result.gradients[i * 2 + 1], 1.0, 1e-10));
        }
    }

    // ---- 3. large batch ----

    #[test]
    fn test_compiled_compute_batch_large() {
        let graph = make_simple_graph();
        let batch_size = 100;
        let mut data = Vec::with_capacity(batch_size * 2);
        for i in 0..batch_size {
            let x = i as f64;
            data.push(x);
            data.push(x * 2.0);
        }
        let batch = BatchInputs::new(&data, batch_size, 2).unwrap();
        let result = graph.compute_batch(batch).unwrap();
        assert_eq!(result.batch_size, 100);
        assert_eq!(result.data.len(), 100);
        for i in 0..batch_size {
            assert!(approx_eq(result.data[i], (i as f64) * 3.0, 1e-10));
        }
    }

    // ---- 4. multi-output batch ----

    #[test]
    fn test_compiled_compute_batch_many() {
        let graph = make_multi_output_graph();
        let data = vec![0.0, 1.0, 2.0];
        let batch = BatchInputs::new(&data, 3, 1).unwrap();
        let result = graph.compute_batch(batch).unwrap();
        assert_eq!(result.batch_size, 3);
        assert_eq!(result.output_dim, 2);
        assert_eq!(result.data.len(), 6);
        // [x, sin(x)] for each row
        assert!(approx_eq(result.data[0], 0.0, 1e-10));
        assert!(approx_eq(result.data[1], 0.0_f64.sin(), 1e-10));
        assert!(approx_eq(result.data[2], 1.0, 1e-10));
        assert!(approx_eq(result.data[3], 1.0_f64.sin(), 1e-10));
        assert!(approx_eq(result.data[4], 2.0, 1e-10));
        assert!(approx_eq(result.data[5], 2.0_f64.sin(), 1e-10));
    }

    // ---- 5. CompiledWorkspace reuse ----

    #[test]
    fn test_workspace_reuse_compute() {
        let graph = make_simple_graph();
        let mut ws = graph.workspace();
        let v1 = graph.compute_with_workspace(&[1.0, 2.0], &mut ws).unwrap();
        assert!(approx_eq(v1, 3.0, 1e-10));
        let v2 = graph.compute_with_workspace(&[3.0, 4.0], &mut ws).unwrap();
        assert!(approx_eq(v2, 7.0, 1e-10));
    }

    #[test]
    fn test_workspace_reuse_gradient() {
        let graph = make_simple_graph();
        let mut ws = graph.workspace();
        let (v1, g1) = graph.gradient_with_workspace(&[1.0, 2.0], &mut ws).unwrap();
        assert!(approx_eq(v1, 3.0, 1e-10));
        assert!(approx_eq(g1[0], 1.0, 1e-10));
        assert!(approx_eq(g1[1], 1.0, 1e-10));
        let (v2, g2) = graph.gradient_with_workspace(&[3.0, 4.0], &mut ws).unwrap();
        assert!(approx_eq(v2, 7.0, 1e-10));
        assert!(approx_eq(g2[0], 1.0, 1e-10));
        assert!(approx_eq(g2[1], 1.0, 1e-10));
    }

    // ---- 6. BackendKind ----

    #[test]
    fn test_backend_kind_name_non_empty() {
        for kind in &[
            BackendKind::Scalar,
            BackendKind::MockDeviceCpu,
            BackendKind::Wgpu,
            BackendKind::SimdF64x4,
            BackendKind::SimdF64x2,
        ] {
            assert!(!kind.name().is_empty());
        }
    }

    #[test]
    fn test_backend_kind_capabilities() {
        for kind in &[
            BackendKind::Scalar,
            BackendKind::MockDeviceCpu,
            BackendKind::Wgpu,
            BackendKind::SimdF64x4,
            BackendKind::SimdF64x2,
        ] {
            let caps = kind.capabilities();
            // Every built-in backend should support f64.
            assert!(caps.supports_f64);
            // Every built-in backend should support at least one opcode.
            assert!(!caps.supported_opcodes.is_empty());
        }
    }

    // ---- 7. BatchInputs ----

    #[test]
    fn test_batch_inputs_try_row_valid() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let batch = BatchInputs::new(&data, 2, 2).unwrap();
        let row0 = batch.try_row(0).unwrap();
        assert_eq!(row0, &[1.0, 2.0]);
        let row1 = batch.try_row(1).unwrap();
        assert_eq!(row1, &[3.0, 4.0]);
    }

    #[test]
    fn test_batch_inputs_try_row_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let batch = BatchInputs::new(&data, 2, 2).unwrap();
        assert!(batch.try_row(2).is_err());
        assert!(batch.try_row(100).is_err());
    }

    #[test]
    fn test_batch_inputs_invalid_shape() {
        assert!(BatchInputs::new(&[1.0, 2.0], 2, 2).is_err());
        assert!(BatchInputs::new(&[1.0, 2.0, 3.0], 1, 2).is_err());
    }

    // ---- 8. BatchValues / BatchValuesBuffer ----

    #[test]
    fn test_batch_values_buffer_reuse() {
        let graph = make_simple_graph();
        let mut buffer = BatchValuesBuffer::new();

        let batch1 = BatchInputs::new(&[1.0, 2.0], 1, 2).unwrap();
        graph.compute_batch_into(batch1, &mut buffer).unwrap();
        let values1 = buffer.to_values();
        assert_eq!(values1.batch_size, 1);
        assert_eq!(values1.data, &[3.0]);

        let batch2 = BatchInputs::new(&[4.0, 5.0, 6.0, 7.0], 2, 2).unwrap();
        graph.compute_batch_into(batch2, &mut buffer).unwrap();
        let values2 = buffer.to_values();
        assert_eq!(values2.batch_size, 2);
        assert_eq!(values2.data, &[9.0, 13.0]);
    }

    // ---- 9. BatchGradients / BatchGradientsBuffer ----

    #[test]
    fn test_batch_gradients_buffer_reuse() {
        let graph = make_simple_graph();
        let mut buffer = BatchGradientsBuffer::new();

        let batch1 = BatchInputs::new(&[1.0, 2.0], 1, 2).unwrap();
        graph.gradient_batch_into(batch1, &mut buffer).unwrap();
        let grad1 = buffer.to_gradients();
        assert_eq!(grad1.batch_size, 1);
        assert!(approx_eq(grad1.values[0], 3.0, 1e-10));
        assert_eq!(grad1.gradients, &[1.0, 1.0]);

        let batch2 = BatchInputs::new(&[4.0, 5.0, 6.0, 7.0], 2, 2).unwrap();
        graph.gradient_batch_into(batch2, &mut buffer).unwrap();
        let grad2 = buffer.to_gradients();
        assert_eq!(grad2.batch_size, 2);
        assert!(approx_eq(grad2.values[0], 9.0, 1e-10));
        assert!(approx_eq(grad2.values[1], 13.0, 1e-10));
        assert_eq!(grad2.gradients, &[1.0, 1.0, 1.0, 1.0]);
    }

    // ---- 10. FlatInstruction and OpCode ----

    #[test]
    fn test_flat_instruction_debug() {
        let fi = FlatInstruction {
            opcode: OpCode::Add,
            output: 2,
            left: 0,
            right: 1,
            value: 0.0,
        };
        let s = format!("{:?}", fi);
        assert!(s.contains("Add"));
        assert!(s.contains("2"));
    }

    #[test]
    fn test_opcode_debug() {
        // Every variant should have a non-empty Debug representation.
        for op in &[
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
        ] {
            assert!(!format!("{:?}", op).is_empty());
        }
    }

    #[test]
    fn test_opcode_arity() {
        assert_eq!(OpCode::Constant.arity(), 0);
        assert_eq!(OpCode::Sin.arity(), 1);
        assert_eq!(OpCode::Neg.arity(), 1);
        assert_eq!(OpCode::Add.arity(), 2);
        assert_eq!(OpCode::Mul.arity(), 2);
    }

    #[test]
    fn test_opcode_from_multi_ad() {
        assert_eq!(OpCode::from_multi_ad(MultiAD::Add).unwrap(), OpCode::Add);
        assert_eq!(OpCode::from_multi_ad(MultiAD::Sin).unwrap(), OpCode::Sin);
        assert!(OpCode::from_multi_ad(MultiAD::Inp).is_err());
    }

    // ---- 11. backend support reports ----

    #[test]
    fn test_backend_support_report_scalar() {
        let graph = make_simple_graph();
        let report = graph.backend_support_report(BackendKind::Scalar).unwrap();
        assert!(report.supports_f64);
        assert!(report.supports_required_opcodes);
        assert!(report.can_compute_batch());
        assert!(report.can_gradient_batch());
        assert_eq!(report.backend, BackendKind::Scalar);
    }

    #[test]
    fn test_simd_support_report() {
        let graph = make_simple_graph();
        let report = graph.simd_support_report().unwrap();
        // Report should belong to a SIMD backend.
        assert!(matches!(
            report.backend,
            BackendKind::SimdF64x4 | BackendKind::SimdF64x2
        ));
    }

    // ---- 12. Device types ----

    #[test]
    fn test_device_memory_location_variants() {
        let host = DeviceMemoryLocation::Host;
        let device = DeviceMemoryLocation::Device;
        assert_ne!(host, device);
        // Equality on same variant.
        assert_eq!(host, DeviceMemoryLocation::Host);
        assert_eq!(device, DeviceMemoryLocation::Device);
    }

    #[test]
    fn test_device_batch_plan_creation() {
        let graph = make_simple_graph();
        let plan = graph.device_batch_plan(BackendKind::Scalar, 10);
        assert_eq!(plan.backend, BackendKind::Scalar);
        assert_eq!(plan.batch_size, 10);
        assert_eq!(plan.input_dim, 2);
        assert_eq!(plan.output_dim, 1);
        assert!(!plan.buffers.is_empty());
        assert!(!plan.buffer_handles.is_empty());
    }

    #[test]
    fn test_mock_device_allocate_batch_buffers() {
        let graph = make_simple_graph();
        let buffers = graph.allocate_mock_device_buffers(5);
        assert_eq!(buffers.plan().backend, BackendKind::MockDeviceCpu);
        assert_eq!(buffers.plan().batch_size, 5);
        assert!(!buffers.buffers().is_empty());
    }

    #[test]
    fn test_device_buffer_set_upload_download() {
        let graph = make_simple_graph();
        let mut buffers = graph.allocate_mock_device_buffers(3);
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buffers
            .upload(DeviceBufferKind::Inputs, &input_data)
            .unwrap();
        let downloaded = buffers.download(DeviceBufferKind::Inputs).unwrap();
        assert_eq!(downloaded, input_data);
    }

    // ---- 13. graph with all ops ----

    #[test]
    fn test_compiled_graph_all_ops() {
        // Build a graph that uses many different ops.
        // Graph: 2 inputs (x, y)
        //  0: x (input)
        //  1: y (input)
        //  2: sin(x)            [Unary Sin]
        //  3: cos(y)            [Unary Cos]
        //  4: x + y            [Binary Add]
        //  5: x * y            [Binary Mul]
        //  6: -x               [Unary Neg]
        //  7: exp(x)           [Unary Exp]
        //  8: ln(y)            [Unary Ln]
        //  9: (x+y) / (x*y)    [Binary Div]
        // 10: x - y            [Binary Sub]
        // 11: tanh(x)          [Unary Tanh]
        // 12: sqrt(abs(x))     [Unary Sqrt] on abs(x)...
        //   Let's simplify: just use relu, abs on input
        let instructions = vec![
            Instruction::Unary {
                op: MultiAD::Sin,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Cos,
                arg: 1,
            },
            Instruction::Binary {
                op: MultiAD::Add,
                left: 0,
                right: 1,
            },
            Instruction::Binary {
                op: MultiAD::Mul,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Neg,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Exp,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Ln,
                arg: 1,
            },
            Instruction::Binary {
                op: MultiAD::Div,
                left: 3,
                right: 4,
            },
            Instruction::Binary {
                op: MultiAD::Sub,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Tanh,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Relu,
                arg: 0,
            },
            Instruction::Unary {
                op: MultiAD::Abs,
                arg: 0,
            },
            Instruction::Constant(42.0),
            Instruction::Binary {
                op: MultiAD::Pow,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Log1pExp,
                arg: 0,
            },
            Instruction::Binary {
                op: MultiAD::LogAddExp,
                left: 0,
                right: 1,
            },
            Instruction::Unary {
                op: MultiAD::Sqrt,
                arg: 4,
            },
            Instruction::Unary {
                op: MultiAD::Tan,
                arg: 0,
            },
        ];
        // Final output: use the Add node (index 4) as primary output.
        let graph = CompiledGraph::new(2, instructions, vec![4]).unwrap();

        // Compute with valid inputs.
        let result = graph.compute(&[0.5, 2.0]).unwrap();
        assert!(approx_eq(result, 2.5, 1e-10));

        // Gradient should also work.
        let (val, grad) = graph.gradient(&[0.5, 2.0]).unwrap();
        assert!(approx_eq(val, 2.5, 1e-10));
        assert_eq!(grad.len(), 2);
        // d/dx (x+y) = 1, d/dy (x+y) = 1
        assert!(approx_eq(grad[0], 1.0, 1e-10));
        assert!(approx_eq(grad[1], 1.0, 1e-10));
    }
}
