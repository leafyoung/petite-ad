//! WGPU backend skeleton with real device allocation and transfer plumbing.

#[cfg(feature = "backend-wgpu")]
use std::sync::mpsc;

#[cfg(feature = "backend-wgpu")]
use pollster::block_on;
#[cfg(feature = "backend-wgpu")]
use wgpu::{self, BufferUsages};

#[cfg(feature = "backend-wgpu")]
use crate::multi::backend::device::{
    AcceleratorDeviceContext, AcceleratorDeviceKind, DeviceBackend, DeviceBatchPlan,
    DeviceBufferHandle, DeviceBufferKind, DeviceExecutionMode, DeviceExecutionTrace,
    DeviceTransferKind, DeviceTransferPolicy, GpuBackendBoundary,
};
#[cfg(feature = "backend-wgpu")]
use crate::multi::backend::dispatch::{BackendKind, ExecutionBackend};
#[cfg(feature = "backend-wgpu")]
use crate::multi::backend::scalar::ScalarBackend;
#[cfg(feature = "backend-wgpu")]
use crate::multi::backend::types::{BackendCapabilities, FlatInstruction, OpCode, UNUSED_NODE_ID};
#[cfg(feature = "backend-wgpu")]
use crate::multi::compiled::{BatchGradientsBuffer, BatchInputs, BatchValuesBuffer, CompiledGraph};
#[cfg(feature = "backend-wgpu")]
use crate::{AutodiffError, Result};

#[cfg(feature = "backend-wgpu")]
/// Initialized WGPU backend skeleton with real device allocation and transfer plumbing.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    pub(crate) boundary: GpuBackendBoundary,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
}

#[cfg(feature = "backend-wgpu")]
/// One real WGPU buffer allocated for a logical batch-plan role.
#[derive(Debug, Clone)]
pub struct WgpuBuffer {
    pub(crate) handle: DeviceBufferHandle,
    pub(crate) buffer: wgpu::Buffer,
}

#[cfg(feature = "backend-wgpu")]
impl WgpuBuffer {
    /// Return the immutable planned buffer handle.
    #[must_use]
    pub fn handle(&self) -> DeviceBufferHandle {
        self.handle
    }
}

#[cfg(feature = "backend-wgpu")]
/// Owned WGPU buffers allocated from a [`DeviceBatchPlan`].
#[derive(Debug, Clone)]
pub struct WgpuBufferSet {
    pub(crate) plan: DeviceBatchPlan,
    pub(crate) buffers: Vec<WgpuBuffer>,
}

#[cfg(feature = "backend-wgpu")]
impl WgpuBufferSet {
    /// Return the plan used to allocate this buffer set.
    #[must_use]
    pub fn plan(&self) -> &DeviceBatchPlan {
        &self.plan
    }

    /// Return all allocated GPU buffers in plan order.
    #[must_use]
    pub fn buffers(&self) -> &[WgpuBuffer] {
        &self.buffers
    }

    /// Return an immutable GPU buffer by logical kind.
    pub fn buffer(&self, kind: DeviceBufferKind) -> Result<&WgpuBuffer> {
        self.buffers
            .iter()
            .find(|buffer| buffer.handle.kind == kind)
            .ok_or(AutodiffError::InvalidGraph {
                reason: "wgpu buffer kind is not in the plan",
            })
    }

    /// Upload host data into a planned GPU buffer.
    pub fn upload(
        &self,
        backend: &WgpuBackend,
        kind: DeviceBufferKind,
        data: &[f64],
    ) -> Result<()> {
        backend.upload_buffer(self, kind, data)
    }

    /// Download a planned GPU buffer into an owned vector.
    pub fn download(&self, backend: &WgpuBackend, kind: DeviceBufferKind) -> Result<Vec<f64>> {
        backend.download_buffer(self, kind)
    }
}

#[cfg(feature = "backend-wgpu")]
#[inline]
fn wgpu_buffer_size_bytes(len: usize) -> u64 {
    let logical = len.saturating_mul(std::mem::size_of::<f64>()) as u64;
    logical.max(8)
}

#[cfg(feature = "backend-wgpu")]
fn encode_f64_slice(data: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().saturating_mul(std::mem::size_of::<f64>()));
    for value in data {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(feature = "backend-wgpu")]
fn decode_f64_bytes(bytes: &[u8], len: usize) -> Result<Vec<f64>> {
    let expected_len = len.saturating_mul(std::mem::size_of::<f64>());
    if bytes.len() < expected_len {
        return Err(AutodiffError::InvalidGraph {
            reason: "wgpu readback length does not match planned buffer length",
        });
    }
    let mut values = Vec::with_capacity(len);
    for chunk in bytes[..expected_len].chunks_exact(std::mem::size_of::<f64>()) {
        let mut array = [0_u8; std::mem::size_of::<f64>()];
        array.copy_from_slice(chunk);
        values.push(f64::from_ne_bytes(array));
    }
    Ok(values)
}

#[cfg(feature = "backend-wgpu")]
#[inline]
fn wgpu_buffer_size_bytes_f32(len: usize) -> u64 {
    let logical = len.saturating_mul(std::mem::size_of::<f32>()) as u64;
    logical.max(4)
}

#[cfg(feature = "backend-wgpu")]
fn encode_f32_slice(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().saturating_mul(std::mem::size_of::<f32>()));
    for value in data {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(feature = "backend-wgpu")]
fn decode_f32_bytes(bytes: &[u8], len: usize) -> Result<Vec<f32>> {
    let expected_len = len.saturating_mul(std::mem::size_of::<f32>());
    if bytes.len() < expected_len {
        return Err(AutodiffError::InvalidGraph {
            reason: "wgpu readback length does not match requested f32 buffer length",
        });
    }
    let mut values = Vec::with_capacity(len);
    for chunk in bytes[..expected_len].chunks_exact(std::mem::size_of::<f32>()) {
        let mut array = [0_u8; std::mem::size_of::<f32>()];
        array.copy_from_slice(chunk);
        values.push(f32::from_ne_bytes(array));
    }
    Ok(values)
}

#[cfg(feature = "backend-wgpu")]
fn encode_u32_slice(data: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len().saturating_mul(std::mem::size_of::<u32>()));
    for value in data {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(feature = "backend-wgpu")]
/// Conservative opcode subset currently supported by the exact-safe native WGPU batch-compute path.
pub const WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES: [OpCode; 4] =
    [OpCode::Constant, OpCode::Neg, OpCode::Relu, OpCode::Abs];

#[cfg(feature = "backend-wgpu")]
fn checked_u32_from_usize(value: usize, reason: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| AutodiffError::InvalidGraph { reason })
}

#[cfg(feature = "backend-wgpu")]
#[inline]
fn is_exact_f32_roundtrip(value: f64) -> bool {
    if value.is_nan() {
        return false;
    }
    let narrowed = value as f32;
    let widened = f64::from(narrowed);
    widened == value && (value != 0.0 || widened.is_sign_negative() == value.is_sign_negative())
}

#[cfg(feature = "backend-wgpu")]
fn wgpu_native_exact_safe_supports_opcode(opcode: OpCode) -> bool {
    WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES.contains(&opcode)
}

#[cfg(feature = "backend-wgpu")]
fn wgpu_native_exact_safe_graph(graph: &CompiledGraph) -> bool {
    !graph.output_nodes().is_empty()
        && graph.flat_instructions_slice().iter().all(|instruction| {
            wgpu_native_exact_safe_supports_opcode(instruction.opcode)
                && (instruction.opcode != OpCode::Constant
                    || is_exact_f32_roundtrip(instruction.value))
        })
}

#[cfg(feature = "backend-wgpu")]
fn wgpu_native_exact_safe_batch(batch: BatchInputs<'_>) -> bool {
    batch.data.iter().copied().all(is_exact_f32_roundtrip)
}

#[cfg(feature = "backend-wgpu")]
const WGPU_NATIVE_WORDS_PER_INSTRUCTION: usize = 8;

#[cfg(feature = "backend-wgpu")]
const WGPU_NATIVE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> instruction_words: array<u32>;
@group(0) @binding(2) var<storage, read> output_nodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> value_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(5) var<storage, read> kernel_meta: array<u32>;

fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 {
        return x;
    }
    return 0.0;
}

fn log1p_exp_scalar(x: f32) -> f32 {
    if x > 0.0 {
        return x + log(1.0 + exp(-x));
    }
    return log(1.0 + exp(x));
}

fn log_add_exp_scalar(a: f32, b: f32) -> f32 {
    var max_value = a;
    var min_value = b;
    if a < b {
        max_value = b;
        min_value = a;
    }
    return max_value + log(1.0 + exp(min_value - max_value));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let num_inputs = kernel_meta[0u];
    let num_instructions = kernel_meta[1u];
    let num_outputs = kernel_meta[2u];
    let value_count = kernel_meta[3u];
    let batch_size = kernel_meta[4u];
    if row >= batch_size {
        return;
    }

    let input_base = row * num_inputs;
    let value_base = row * value_count;
    let output_base = row * num_outputs;

    var input_index = 0u;
    loop {
        if input_index >= num_inputs {
            break;
        }
        value_data[value_base + input_index] = input_data[input_base + input_index];
        input_index = input_index + 1u;
    }

    var instruction_index = 0u;
    loop {
        if instruction_index >= num_instructions {
            break;
        }
        let word_base = instruction_index * 8u;
        let opcode = instruction_words[word_base + 0u];
        let output_index = instruction_words[word_base + 1u];
        let left_index = instruction_words[word_base + 2u];
        let right_index = instruction_words[word_base + 3u];
        let value_bits = instruction_words[word_base + 4u];
        var left_value: f32 = 0.0;
        if left_index != 4294967295u {
            left_value = value_data[value_base + left_index];
        }
        var right_value: f32 = 0.0;
        if right_index != 4294967295u {
            right_value = value_data[value_base + right_index];
        }
        var result_value: f32 = 0.0;
        switch opcode {
            case 0u: {
                result_value = bitcast<f32>(value_bits);
            }
            case 1u: {
                result_value = left_value + right_value;
            }
            case 2u: {
                result_value = left_value - right_value;
            }
            case 3u: {
                result_value = left_value * right_value;
            }
            case 4u: {
                result_value = left_value / right_value;
            }
            case 5u: {
                result_value = pow(left_value, right_value);
            }
            case 6u: {
                result_value = sin(left_value);
            }
            case 7u: {
                result_value = cos(left_value);
            }
            case 8u: {
                result_value = tan(left_value);
            }
            case 9u: {
                result_value = tanh(left_value);
            }
            case 10u: {
                result_value = relu_scalar(left_value);
            }
            case 11u: {
                result_value = log1p_exp_scalar(left_value);
            }
            case 12u: {
                result_value = log_add_exp_scalar(left_value, right_value);
            }
            case 13u: {
                result_value = -left_value;
            }
            case 14u: {
                result_value = exp(left_value);
            }
            case 15u: {
                result_value = log(left_value);
            }
            case 16u: {
                result_value = sqrt(left_value);
            }
            case 17u: {
                result_value = abs(left_value);
            }
            default: {
                result_value = 0.0;
            }
        }
        value_data[value_base + output_index] = result_value;
        instruction_index = instruction_index + 1u;
    }

    var output_index = 0u;
    loop {
        if output_index >= num_outputs {
            break;
        }
        let node = output_nodes[output_index];
        output_data[output_base + output_index] = value_data[value_base + node];
        output_index = output_index + 1u;
    }
}
"#;

#[cfg(feature = "backend-wgpu")]
impl WgpuBackend {
    /// Create the default WGPU backend using device id 0 and automatic transfers.
    pub fn new_default() -> Result<Self> {
        Self::new(
            AcceleratorDeviceContext::wgpu(0),
            DeviceTransferPolicy::Automatic,
        )
    }

    /// Create the WGPU backend skeleton from a context and transfer policy.
    pub fn new(
        context: AcceleratorDeviceContext,
        transfer_policy: DeviceTransferPolicy,
    ) -> Result<Self> {
        Self::from_boundary(GpuBackendBoundary::new(context, transfer_policy))
    }

    /// Create the WGPU backend skeleton from a boundary descriptor.
    pub fn from_boundary(boundary: GpuBackendBoundary) -> Result<Self> {
        if boundary.context.kind != AcceleratorDeviceKind::Wgpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu backend requires a WGPU accelerator context",
            });
        }
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|_| AutodiffError::InvalidGraph {
            reason: "wgpu adapter request failed",
        })?;
        let adapter_info = adapter.get_info();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("petite-ad-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|_| AutodiffError::InvalidGraph {
            reason: "wgpu device request failed",
        })?;
        Ok(Self {
            boundary,
            device,
            queue,
            adapter_info,
        })
    }

    /// Return the boundary descriptor used for this backend.
    #[must_use]
    pub fn boundary(&self) -> &GpuBackendBoundary {
        &self.boundary
    }

    /// Return the accelerator context used for this backend.
    #[must_use]
    pub fn context(&self) -> &AcceleratorDeviceContext {
        &self.boundary.context
    }

    /// Return the configured transfer policy.
    #[must_use]
    pub fn transfer_policy(&self) -> DeviceTransferPolicy {
        self.boundary.transfer_policy
    }

    /// Return the resolved adapter name.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Return the conservative exact-safe opcode subset used by the native WGPU batch-compute path.
    #[must_use]
    pub fn native_batch_compute_supported_opcodes() -> &'static [OpCode] {
        &WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES
    }

    /// Return whether this graph is eligible for the restricted exact-safe native WGPU path.
    #[must_use]
    pub fn supports_native_batch_compute(&self, graph: &CompiledGraph) -> bool {
        wgpu_native_exact_safe_graph(graph)
    }

    /// Return whether a concrete batch is eligible for the exact-safe native WGPU path.
    #[must_use]
    pub fn supports_native_batch_compute_for_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
    ) -> bool {
        self.supports_native_batch_compute(graph) && wgpu_native_exact_safe_batch(batch)
    }

    fn create_initialized_storage_buffer(
        &self,
        label: &'static str,
        bytes: &[u8],
        min_size: u64,
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: min_size.max(bytes.len() as u64).max(4),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    }

    fn download_raw_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("petite-ad-wgpu-readback"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("petite-ad-wgpu-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        let submission = self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_for(submission));
        match rx.recv() {
            Ok(Ok(())) => {}
            _ => {
                return Err(AutodiffError::InvalidGraph {
                    reason: "wgpu readback mapping failed",
                });
            }
        }
        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    fn compute_batch_native(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
    ) -> Result<Vec<f64>> {
        let metadata = graph.metadata();
        let input_f32: Vec<f32> = batch.data.iter().map(|value| *value as f32).collect();
        let output_count = batch.batch_size.saturating_mul(metadata.num_outputs);
        let input_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-inputs",
            &encode_f32_slice(&input_f32),
            wgpu_buffer_size_bytes_f32(input_f32.len()),
        );
        let instruction_words = encode_u32_slice(&self.native_instruction_words(graph)?);
        let instruction_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-instructions",
            &instruction_words,
            instruction_words.len() as u64,
        );
        let output_nodes: Result<Vec<u32>> = graph
            .output_nodes()
            .iter()
            .map(|node| {
                checked_u32_from_usize(*node, "wgpu native output node index exceeds u32 range")
            })
            .collect();
        let output_nodes = output_nodes?;
        let output_node_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-output-nodes",
            &encode_u32_slice(&output_nodes),
            (output_nodes.len().max(1) * std::mem::size_of::<u32>()) as u64,
        );
        let value_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("petite-ad-wgpu-native-values"),
            size: wgpu_buffer_size_bytes_f32(batch.batch_size.saturating_mul(metadata.value_count)),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("petite-ad-wgpu-native-outputs"),
            size: wgpu_buffer_size_bytes_f32(output_count),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let meta = [
            checked_u32_from_usize(
                metadata.num_inputs,
                "wgpu native input dimension exceeds u32 range",
            )?,
            checked_u32_from_usize(
                metadata.num_instructions,
                "wgpu native instruction count exceeds u32 range",
            )?,
            checked_u32_from_usize(
                metadata.num_outputs,
                "wgpu native output dimension exceeds u32 range",
            )?,
            checked_u32_from_usize(
                metadata.value_count,
                "wgpu native value count exceeds u32 range",
            )?,
            checked_u32_from_usize(batch.batch_size, "wgpu native batch size exceeds u32 range")?,
        ];
        let meta_buffer = self.create_initialized_storage_buffer(
            "petite-ad-wgpu-native-meta",
            &encode_u32_slice(&meta),
            (meta.len() * std::mem::size_of::<u32>()) as u64,
        );
        self.queue.submit([]);

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("petite-ad-wgpu-native-compute-shader"),
                source: wgpu::ShaderSource::Wgsl(WGPU_NATIVE_SHADER.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("petite-ad-wgpu-native-compute-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("petite-ad-wgpu-native-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instruction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: meta_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("petite-ad-wgpu-native-compute-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("petite-ad-wgpu-native-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = checked_u32_from_usize(
                batch.batch_size.div_ceil(64),
                "wgpu native workgroup count exceeds u32 range",
            )?;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        let raw_output =
            self.download_raw_buffer(&output_buffer, wgpu_buffer_size_bytes_f32(output_count))?;
        let output_f32 = decode_f32_bytes(&raw_output, output_count)?;
        Ok(output_f32.into_iter().map(f64::from).collect())
    }

    fn native_instruction_words(&self, graph: &CompiledGraph) -> Result<Vec<u32>> {
        let mut words = Vec::with_capacity(
            graph.flat_instructions_slice().len() * WGPU_NATIVE_WORDS_PER_INSTRUCTION,
        );
        for instruction in graph.flat_instructions_slice() {
            words.push(match instruction.opcode {
                OpCode::Constant => 0,
                OpCode::Add => 1,
                OpCode::Sub => 2,
                OpCode::Mul => 3,
                OpCode::Div => 4,
                OpCode::Pow => 5,
                OpCode::Sin => 6,
                OpCode::Cos => 7,
                OpCode::Tan => 8,
                OpCode::Tanh => 9,
                OpCode::Relu => 10,
                OpCode::Log1pExp => 11,
                OpCode::LogAddExp => 12,
                OpCode::Neg => 13,
                OpCode::Exp => 14,
                OpCode::Ln => 15,
                OpCode::Sqrt => 16,
                OpCode::Abs => 17,
            });
            words.push(checked_u32_from_usize(
                instruction.output,
                "wgpu native instruction output index exceeds u32 range",
            )?);
            words.push(if instruction.left == UNUSED_NODE_ID {
                u32::MAX
            } else {
                checked_u32_from_usize(
                    instruction.left,
                    "wgpu native instruction left index exceeds u32 range",
                )?
            });
            words.push(if instruction.right == UNUSED_NODE_ID {
                u32::MAX
            } else {
                checked_u32_from_usize(
                    instruction.right,
                    "wgpu native instruction right index exceeds u32 range",
                )?
            });
            words.push((instruction.value as f32).to_bits());
            words.extend_from_slice(&[0, 0, 0]);
        }
        Ok(words)
    }

    /// Allocate a real WGPU buffer set for a compiled graph and batch size.
    pub fn allocate_batch_buffers(
        &self,
        graph: &CompiledGraph,
        batch_size: usize,
    ) -> Result<WgpuBufferSet> {
        let plan = self.batch_plan(graph, batch_size);
        let mut buffers = Vec::with_capacity(plan.buffer_handles.len());
        for handle in &plan.buffer_handles {
            let label = format!("petite-ad-wgpu-{:?}", handle.kind);
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label.as_str()),
                size: wgpu_buffer_size_bytes(handle.len),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            buffers.push(WgpuBuffer {
                handle: *handle,
                buffer,
            });
        }
        Ok(WgpuBufferSet { plan, buffers })
    }

    fn upload_buffer(
        &self,
        buffers: &WgpuBufferSet,
        kind: DeviceBufferKind,
        data: &[f64],
    ) -> Result<()> {
        let buffer = buffers.buffer(kind)?;
        if buffer.handle.len != data.len() {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu upload length must match planned buffer length",
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let bytes = encode_f64_slice(data);
        self.queue.write_buffer(&buffer.buffer, 0, &bytes);
        self.queue.submit([]);
        Ok(())
    }

    fn download_buffer(&self, buffers: &WgpuBufferSet, kind: DeviceBufferKind) -> Result<Vec<f64>> {
        let buffer = buffers.buffer(kind)?;
        if buffer.handle.len == 0 {
            return Ok(Vec::new());
        }
        let raw =
            self.download_raw_buffer(&buffer.buffer, wgpu_buffer_size_bytes(buffer.handle.len))?;
        decode_f64_bytes(&raw, buffer.handle.len)
    }

    /// Execute batch value computation through explicit WGPU transfers.
    pub fn compute_batch_with_buffers(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        graph.check_batch(batch)?;
        if buffers.plan.backend != BackendKind::Wgpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu execution requires a WGPU buffer plan",
            });
        }
        if buffers.plan.batch_size != batch.batch_size || buffers.plan.input_dim != batch.input_dim
        {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch shape must match WGPU buffer plan",
            });
        }

        let transfers = buffers.plan.compute_transfer_plan.clone();
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::HostToDevice {
                if transfer.buffer != DeviceBufferKind::Inputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "wgpu compute supports host-to-device input transfers only",
                    });
                }
                buffers.upload(self, DeviceBufferKind::Inputs, batch.data)?;
            }
        }

        let used_native_kernel = self.supports_native_batch_compute_for_batch(graph, batch);
        let scratch_data = if used_native_kernel {
            self.compute_batch_native(graph, batch)?
        } else {
            let device_inputs = buffers.download(self, DeviceBufferKind::Inputs)?;
            let device_batch = BatchInputs::new(&device_inputs, batch.batch_size, batch.input_dim)?;
            let mut scratch = BatchValuesBuffer::new();
            ScalarBackend.compute_batch(graph, device_batch, &mut scratch)?;
            scratch.data
        };
        buffers.upload(self, DeviceBufferKind::Outputs, &scratch_data)?;
        output.reset(batch.batch_size, graph.output_nodes.len());
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::DeviceToHost {
                if transfer.buffer != DeviceBufferKind::Outputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "wgpu compute supports device-to-host output transfers only",
                    });
                }
                output
                    .data
                    .extend_from_slice(&buffers.download(self, DeviceBufferKind::Outputs)?);
            }
        }

        Ok(DeviceExecutionTrace {
            backend: BackendKind::Wgpu,
            mode: DeviceExecutionMode::ComputeBatch,
            transfers,
            used_native_kernel,
        })
    }

    /// Execute batch gradient computation through explicit WGPU transfers.
    pub fn gradient_batch_with_buffers(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffers: &mut WgpuBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        graph.check_batch(batch)?;
        if buffers.plan.backend != BackendKind::Wgpu {
            return Err(AutodiffError::InvalidGraph {
                reason: "wgpu execution requires a WGPU buffer plan",
            });
        }
        if buffers.plan.batch_size != batch.batch_size || buffers.plan.input_dim != batch.input_dim
        {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch shape must match WGPU buffer plan",
            });
        }

        let transfers = buffers.plan.gradient_transfer_plan.clone();
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::HostToDevice {
                if transfer.buffer != DeviceBufferKind::Inputs {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "wgpu gradient supports host-to-device input transfers only",
                    });
                }
                buffers.upload(self, DeviceBufferKind::Inputs, batch.data)?;
            }
        }

        let device_inputs = buffers.download(self, DeviceBufferKind::Inputs)?;
        let device_batch = BatchInputs::new(&device_inputs, batch.batch_size, batch.input_dim)?;
        let mut scratch = BatchGradientsBuffer::new();
        ScalarBackend.gradient_batch(graph, device_batch, &mut scratch)?;
        buffers.upload(self, DeviceBufferKind::PrimaryValues, &scratch.values)?;
        buffers.upload(self, DeviceBufferKind::Gradients, &scratch.gradients)?;
        output.reset(batch.batch_size, graph.num_inputs);
        for transfer in &transfers {
            if transfer.kind == DeviceTransferKind::DeviceToHost {
                match transfer.buffer {
                    DeviceBufferKind::PrimaryValues => output.values.extend_from_slice(
                        &buffers.download(self, DeviceBufferKind::PrimaryValues)?,
                    ),
                    DeviceBufferKind::Gradients => output
                        .gradients
                        .extend_from_slice(&buffers.download(self, DeviceBufferKind::Gradients)?),
                    _ => {
                        return Err(AutodiffError::InvalidGraph {
                            reason:
                                "wgpu gradient supports primary-value and gradient downloads only",
                        });
                    }
                }
            }
        }

        Ok(DeviceExecutionTrace {
            backend: BackendKind::Wgpu,
            mode: DeviceExecutionMode::GradientBatch,
            transfers,
            used_native_kernel: false,
        })
    }
}

impl DeviceBackend for WgpuBackend {
    fn backend_kind(&self, _graph: &CompiledGraph) -> BackendKind {
        BackendKind::Wgpu
    }
}

impl ExecutionBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::wgpu_f64()
    }

    fn compute(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<f64> {
        let batch = BatchInputs::new(inputs, 1, inputs.len())?;
        let mut values = BatchValuesBuffer::new();
        self.compute_batch(graph, batch, &mut values)?;
        values
            .data
            .first()
            .copied()
            .ok_or(AutodiffError::InvalidGraph {
                reason: "wgpu compute did not produce an output value",
            })
    }

    fn compute_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        graph.validate_backend_capabilities(&self.capabilities())?;
        let mut gpu_buffers = self.allocate_batch_buffers(graph, batch.batch_size)?;
        self.compute_batch_with_buffers(graph, batch, &mut gpu_buffers, buffer)?;
        Ok(())
    }

    fn gradient(&self, graph: &CompiledGraph, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let batch = BatchInputs::new(inputs, 1, inputs.len())?;
        let mut gradients = BatchGradientsBuffer::new();
        self.gradient_batch(graph, batch, &mut gradients)?;
        let value = gradients
            .values
            .first()
            .copied()
            .ok_or(AutodiffError::InvalidGraph {
                reason: "wgpu gradient did not produce a primary output value",
            })?;
        Ok((value, gradients.gradients))
    }

    fn gradient_batch(
        &self,
        graph: &CompiledGraph,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        graph.validate_backend_capabilities(&self.capabilities())?;
        let mut gpu_buffers = self.allocate_batch_buffers(graph, batch.batch_size)?;
        self.gradient_batch_with_buffers(graph, batch, &mut gpu_buffers, buffer)?;
        Ok(())
    }
}
