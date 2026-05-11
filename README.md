# petite-ad

A pure Rust automatic differentiation library supporting both single-variable and multi-variable functions with reverse-mode differentiation (backpropagation).

## Features

- **Single-variable autodiff** ([`MonoAD`](src/mono/mono_ad.rs)) - Chain operations like `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, and `abs` with automatic gradient computation
- **Multi-variable autodiff** ([`MultiAD`](src/multi/multi_ad.rs)) - Build computational graphs for functions with multiple inputs
- **Box-wrapped by default** - Results use `Box<dyn Fn>` for flexibility; convert to `Arc` when needed for thread-safety
- **Zero-copy backward pass** - Gradients computed efficiently through closure chains
- **Convenient macros** - Use `mono_ops![]` for concise operation lists
- **Builder API** - Fluent interface for constructing computation graphs
- **Reusable graph/tape API** - Build graphs with node handles, select explicit single or multiple outputs, and evaluate repeatedly
- **Compiled IR, batching, and backend hooks** - Evaluate closure-free compiled graphs over scalar inputs, flat row-major batches, or prototype SIMD batch backends
- **Graph validation/export** - Validate reusable graphs and export Mermaid/DOT diagrams
- **Second-order derivatives** - Hessian computation for both single and multi-variable functions
- **Public forward-mode AD** - Single-variable derivatives and multivariate directional derivatives / JVPs
- **Opt-in checked mode** - Real-domain validation for scalar `Ln`/`Sqrt` and graph `Div`/`Ln`/`Sqrt`/`Pow`
- **Comprehensive tests** - Unit tests covering operations, edge cases, graph migration, and Hessian methods

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
petite-ad = "0.1.2"
```

## Quick Start

### Single-Variable Functions

```rust
use petite_ad::{mono_ops, MonoAD};

let exprs = mono_ops![sin, cos, exp];
let (value, backprop) = MonoAD::compute_grad(&exprs, 2.0);
let gradient = backprop(1.0);

println!("f(2.0) = {}", value);      // exp(cos(sin(2.0)))
println!("f'(2.0) = {}", gradient);  // derivative
```

### Multi-Variable Functions

#### Using the reusable Graph API (Recommended)

```rust
use petite_ad::Graph;

// Build: f(x, y) = sin(x) * (x + y)
let mut graph = Graph::new(2);
let x = graph.input(0);
let y = graph.input(1);
let sum = graph.add(x, y);
let sin_x = graph.sin(x);
graph.mul(sum, sin_x);

let inputs = &[0.6, 1.4];
let (value, backprop_fn) = graph.compute_grad(inputs).unwrap();
let gradients = backprop_fn(1.0);

println!("f(0.6, 1.4) = {}", value);
println!("∇f = {:?}", gradients);  // [∂f/∂x, ∂f/∂y]

// Reuse the same graph but select a different output node.
graph.set_output(sum).unwrap();
assert!((graph.compute(inputs).unwrap() - 2.0).abs() < 1e-10);

// Or expose multiple outputs directly.
graph.set_outputs(&[sum, sin_x]).unwrap();
let values = graph.compute_many(inputs).unwrap();
let jacobian = graph.jacobian(inputs).unwrap();
assert_eq!(values.len(), 2);
assert_eq!(jacobian.len(), 2);
```

You can also compile a reusable tape for repeated evaluation:

```rust
# use petite_ad::Graph;
# let mut graph = Graph::new(2);
# let x = graph.input(0);
# let y = graph.input(1);
# let sum = graph.add(x, y);
# let sin_x = graph.sin(x);
# graph.mul(sum, sin_x);
let tape = graph.compile();
let value = tape.compute(&[0.6, 1.4]).unwrap();
assert!(value.is_finite());
```

For repeated hot-loop evaluation, reuse a `TapeWorkspace` to avoid reallocating buffers:

```rust
# use petite_ad::Graph;
# let mut graph = Graph::new(2);
# let x = graph.input(0);
# let y = graph.input(1);
# let sum = graph.add(x, y);
# graph.mul(sum, y);
let tape = graph.compile();
let mut workspace = tape.workspace();
let (value, grad) = tape.gradient_with_workspace(&[2.0, 3.0], &mut workspace).unwrap();
assert_eq!(grad.len(), 2);
assert!(value.is_finite());
```

For closure-free execution and batch evaluation, compile to the instruction IR:

```rust
# use petite_ad::{BatchGradientsBuffer, BatchInputs, BatchValuesBuffer, ExecutionBackend, Graph, SimdBackend};
# let mut graph = Graph::new(2);
# let x = graph.input(0);
# let y = graph.input(1);
# graph.mul(x, y);
let compiled = graph.compile_ir().unwrap();
let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
let values = compiled.compute_batch(batch).unwrap();
assert_eq!(values.data, vec![6.0, 20.0]);
let (backend, auto_values) = compiled.compute_batch_auto(batch).unwrap();
assert_eq!(auto_values.data, values.data);
let simd_report = compiled.simd_support_report().unwrap();
assert_eq!(simd_report.lane_width, simd_report.backend.lane_width());
assert_eq!(backend.name(), if simd_report.can_compute_batch() { simd_report.backend.name() } else { "scalar" });
let mut auto_buffer = BatchValuesBuffer::new();
backend.compute_batch(&compiled, batch, &mut auto_buffer).unwrap();
assert_eq!(auto_buffer.data, values.data);
let plan = compiled.device_batch_plan(backend, batch.batch_size);
assert_eq!(plan.layout, petite_ad::BatchLayout::RowMajor);

// Mock device execution uses explicit allocated buffers and transfer plans.
let mock = petite_ad::MockDeviceBackend;
let mut device_buffers = mock.allocate_batch_buffers(&compiled, batch.batch_size);
let mut device_values = BatchValuesBuffer::new();
let trace = mock
    .compute_batch_with_buffers(&compiled, batch, &mut device_buffers, &mut device_values)
    .unwrap();
assert_eq!(trace.mode, petite_ad::DeviceExecutionMode::ComputeBatch);
assert_eq!(device_values.data, values.data);

// With the optional `backend-wgpu` feature, the first real GPU backend can
// initialize a WGPU device, allocate real GPU buffers, and run a restricted
// exact-safe native compute kernel for batch value execution. Today the native
// path is intentionally conservative: it only accepts graphs composed from
// exact f32-roundtrippable constants plus neg/relu/abs, and it also requires
// the concrete batch inputs to roundtrip through f32 exactly. Other graphs or
// batches still use the host fallback path. Gradients also still use host
// fallback for now.
# #[cfg(feature = "backend-wgpu")]
# {
let boundary = petite_ad::GpuBackendBoundary::new(
    petite_ad::AcceleratorDeviceContext::wgpu(0),
    petite_ad::DeviceTransferPolicy::Explicit,
);
let wgpu = boundary.initialize_wgpu().unwrap();
let mut gpu_buffers = compiled.allocate_wgpu_buffers(&wgpu, batch.batch_size).unwrap();
let mut gpu_values = BatchValuesBuffer::new();
let native_ok = compiled.supports_native_wgpu_batch_compute_for_batch(&wgpu, batch);
let trace = compiled
    .compute_batch_wgpu_into(&wgpu, batch, &mut gpu_buffers, &mut gpu_values)
    .unwrap();
assert_eq!(trace.used_native_kernel, native_ok);
if native_ok {
    assert_eq!(gpu_values.data, values.data);
}
assert_eq!(
    petite_ad::WgpuBackend::native_batch_compute_supported_opcodes(),
    petite_ad::WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES,
);
# }

// Hot loops can reuse output buffers and inspect static metadata.
let metadata = compiled.metadata();
assert_eq!(metadata.num_inputs, 2);
let flat = compiled.flat_instructions().unwrap();
assert_eq!(flat.len(), metadata.num_instructions);
let mut buffer = BatchValuesBuffer::new();
compiled.compute_batch_into(batch, &mut buffer).unwrap();
assert_eq!(buffer.data, values.data);

// SIMD backends use the same flat batch ABI. The prototypes support f64x2
// and f64x4 batch compute/gradients for constants, +, -, *, /, pow,
// log1p_exp, logaddexp, negation, sin, cos, tan, exp, ln, sqrt, ReLU,
// abs, and tanh. Non-native math uses exact scalar-lane fallback semantics.
// The optional WGPU backend adds real device allocation/upload/download plus
// a first native batch-compute kernel. To preserve strict parity, the native
// path is currently restricted to an exact-safe subset instead of running all
// graphs through WGSL f32 math. Unsupported graphs and inexact batches fall
// back to the host path; use `supports_native_wgpu_batch_compute_for_batch`
// and `DeviceExecutionTrace::used_native_kernel` to see whether a concrete
// batch stayed on the native path. The currently exposed subset is available
// as `WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES`.
let simd = SimdBackend;
let simd_capabilities = simd.capabilities();
if simd_capabilities.supports_batch_compute {
    simd.compute_batch(&compiled, batch, &mut buffer).unwrap();
    assert_eq!(buffer.data, values.data);
}
if simd_capabilities.supports_batch_gradient {
    let mut gradients = BatchGradientsBuffer::new();
    simd.gradient_batch(&compiled, batch, &mut gradients).unwrap();
    assert_eq!(gradients.batch_size, 2);
}
```

Optimizers operate directly on parameter and gradient slices:

```rust
# use petite_ad::Adam;
let mut optimizer = Adam::new(1, 0.1);
let mut params = vec![0.0];
optimizer.step(&mut params, &[-1.0]).unwrap();
assert!(params[0] > 0.0);
```

You can convert reusable graphs back to the legacy tuple form when they do not contain constants:

```rust
# use petite_ad::Graph;
# let mut graph = Graph::new(2);
# let x = graph.input(0);
# let y = graph.input(1);
# let sum = graph.add(x, y);
# graph.sin(sum);
let legacy_ops = graph.to_operations().unwrap();
assert!(!legacy_ops.is_empty());
```

#### Using the GraphBuilder API

```rust
use petite_ad::GraphBuilder;

let mut builder = GraphBuilder::new(2);
let x = builder.input_node(0);
let y = builder.input_node(1);
let sum = builder.add_node(x, y);
let sin_x = builder.sin_node(x);
builder.mul_node(sum, sin_x);

let graph = builder.build_graph_with_output(sum).unwrap();
let inputs = &[0.6, 1.4];
let (value, backprop_fn) = graph.compute_grad(inputs).unwrap();
let gradients = backprop_fn(1.0);

println!("f(0.6, 1.4) = {}", value);
println!("∇f = {:?}", gradients);

let graph_multi = builder.build_graph_with_outputs(&[sum, sin_x]).unwrap();
let values = graph_multi.compute_many(inputs).unwrap();
assert_eq!(values.len(), 2);
```

#### Using Manual Tuple Graph Construction

```rust
use petite_ad::MultiAD;

// Build computational graph: f(x, y) = sin(x) * (x + y)
let exprs = &[
    (MultiAD::Inp, vec![0]),    // x at index 0
    (MultiAD::Inp, vec![1]),    // y at index 1
    (MultiAD::Add, vec![0, 1]), // x + y at index 2
    (MultiAD::Sin, vec![0]),    // sin(x) at index 3
    (MultiAD::Mul, vec![2, 3]), // sin(x) * (x + y) at index 4
];

let inputs = &[0.6, 1.4];
let (value, backprop_fn) = MultiAD::compute_grad(exprs, inputs).unwrap();
let gradients = backprop_fn(1.0);

println!("f(0.6, 1.4) = {}", value);
println!("∇f = {:?}", gradients);  // [∂f/∂x, ∂f/∂y]
```

## Available Operations

### MonoAD (Single-Variable)

| Operation | Description | Derivative |
| --------- | ----------- | ---------- |
| `Sin`     | Sine        | `x.cos()`            |
| `Cos`     | Cosine      | `-x.sin()`           |
| `Tan`     | Tangent     | `1 / cos²(x)`        |
| `Exp`     | Exponential | `exp(x)`             |
| `Neg`     | Negation    | `-1`                 |
| `Ln`      | Natural log | `1 / x`              |
| `Sqrt`    | Square root | `1 / (2 * sqrt(x))`  |
| `Abs`     | Absolute    | `sign(x)` (0 at `x=0`) |

### MultiAD (Multi-Variable)

| Operation | Arity | Description              |
| --------- | ----- | ------------------------ |
| `Inp`     | 1     | Input placeholder        |
| `Add`     | 2     | Addition: `a + b`        |
| `Sub`     | 2     | Subtraction: `a - b`     |
| `Mul`     | 2     | Multiplication: `a * b`  |
| `Div`     | 2     | Division: `a / b`        |
| `Pow`     | 2     | Power: `a^b`             |
| `Sin`     | 1     | Sine: `sin(x)`           |
| `Cos`     | 1     | Cosine: `cos(x)`         |
| `Tan`     | 1     | Tangent: `tan(x)`        |
| `Tanh`    | 1     | Hyperbolic tangent       |
| `Neg`     | 1     | Negation: `-x`           |
| `Exp`     | 1     | Exponential: `exp(x)`    |
| `Ln`      | 1     | Natural log: `ln(x)`     |
| `Sqrt`    | 1     | Square root: `sqrt(x)`   |
| `Abs`     | 1     | Absolute value: `abs(x)` |

## Higher-Order Derivatives (Second-Order / Hessian)

### Documentation

For comprehensive mathematical background and algorithm details, see:
- **[Univariate Hessian Theory (mono_ad_hessian.md)](docs/mono_ad_hessian.md)** - Complete derivations, algorithms, and examples for single-variable functions
- **[Multivariate Hessian Theory (multi_ad_hessian.md)](docs/multi_ad_hessian.md)** - Complete theory for MultiAD Hessian methods (RR, FR, RF) with examples

### MonoAD Second Derivatives

The library provides **four methods** to compute second derivatives for single-variable functions:

#### 1. Finite Differences (MonoAD::compute_hessian)

Approximate second derivative using numerical differentiation:

```rust
use petite_ad::{MonoAD, mono_ops};

// f(x) = sin(x), f''(x) = -sin(x)
let ops = mono_ops![sin];
let x = 0.5;

let second_deriv = MonoAD::compute_hessian(&ops, x);
println!("f''({}) = {}", x, second_deriv); // ≈ -0.4794
```

- **Accuracy**: ~1e-5 to 1e-6 (using ε = 1e-5)
- **Use case**: Quick prototyping, most practical applications

#### 2. Exact Methods (MonoAD2RR/FR/RF)

Compute exact second derivatives using automatic differentiation:

```rust
use petite_ad::{MonoAD2RR, mono_ops_rr};

// f(x) = exp(sin(x))
let ops = mono_ops_rr![sin, exp];
let x = 0.5;

let hessian = MonoAD2RR::compute_hessian(&ops, x);
// Result is exact up to machine precision (< 1e-12)
```

**Three exact methods available**:
- **MonoAD2RR** (Reverse-over-Reverse): Most direct, propagates second derivatives backward
- **MonoAD2FR** (Forward-over-Reverse): Uses dual numbers to differentiate gradient
- **MonoAD2RF** (Reverse-over-Forward): Equivalent to FR for univariate functions

The exact mono Hessian types support `Sin`, `Cos`, `Tan`, `Exp`, `Neg`, `Ln`, `Sqrt`, and `Abs`. `Abs` follows the library's raw convention at zero: first derivative `0`, second derivative `0`.

**Comparison** (MonoAD - single variable functions):

| Method | Time | Space | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| Finite Diff | O(n) | O(1) | ~1e-5 | Prototyping |
| RR (Exact) | O(n) | O(n) | < 1e-12 | Production, optimization |
| FR (Exact) | O(n) | O(n) | < 1e-12 | Education, research |
| RF (Exact) | O(n) | O(n) | < 1e-12 | Research |

### MultiAD Hessian Computation

For multi-variable functions, compute the Hessian matrix (second-order partial derivatives):

```rust
use petite_ad::{MultiAD, multi_ops};

// f(x, y) = x² + y²
// Hessian: [[2, 0], [0, 2]]
let exprs = multi_ops![
    (inp, 0), (inp, 1),
    (mul, 0, 0), (mul, 1, 1),
    (add, 2, 3)
];

let hessian = MultiAD::compute_hessian(&exprs, &[2.0, 3.0]).unwrap();
println!("H[0][0] = {}", hessian[0][0]); // ∂²f/∂x² = 2
println!("H[0][1] = {}", hessian[0][1]); // ∂²f/∂x∂y = 0
println!("H[1][0] = {}", hessian[1][0]); // ∂²f/∂y∂x = 0
println!("H[1][1] = {}", hessian[1][1]); // ∂²f/∂y² = 2
```

Currently uses central finite differences on gradients (ε = 1e-5; approximate, unlike the exact methods below).

The exact multivariate Hessian types currently cover a larger smooth subset of operations than before, including `Neg`, `Pow`, `Sub`, `Div`, `Tan`, `Ln`, and `Sqrt` in addition to `Inp`, `Sin`, `Cos`, `Exp`, `Add`, and `Mul`.

### MultiAD Second-Order Methods (Exact Hessian)

The library provides **three exact methods** for computing Hessians of multivariate functions:

```rust
use petite_ad::MultiAD2RR;

// f(x, y) = sin(x) + exp(y)
// Hessian: [[-sin(x), 0], [0, exp(y)]]
let ops = &[
    MultiAD2RR::Inp(0), MultiAD2RR::Sin,
    MultiAD2RR::Inp(1), MultiAD2RR::Exp,
    MultiAD2RR::Add,
];

let x = vec![1.0, 2.0];
let hessian = MultiAD2RR::compute_hessian(ops, &x).unwrap();
// Result is exact up to machine precision (< 1e-12)
```

**Three exact methods available** (all machine-precision, no finite differences):
- **MultiAD2RR** (Reverse-over-Reverse): Per-node gradient vectors with outer-product Hessian accumulation in reverse pass
- **MultiAD2FR** (Forward-over-Reverse): Dual-number forward pass + dual-adjoint reverse pass for each seed direction
- **MultiAD2RF** (Reverse-over-Forward): Same dual-number algorithm as FR (equivalent for scalar f: ℝⁿ → ℝ)

**Comparison** (MultiAD - multi-variable functions, n inputs, G = graph size):

| Method | Time | Space | Accuracy | Approach |
|--------|------|-------|----------|----------|
| RR | O(G·n²) | O(G·n) | < 1e-12 | Scalar reverse pass with per-node grad vectors |
| FR | O(n·G) | O(G) | < 1e-12 | Dual forward + dual reverse per seed |
| RF | O(n·G) | O(G) | < 1e-12 | Same algorithm as FR for scalar functions |

For detailed theory and algorithms, see [docs/multi_ad_hessian.md](docs/multi_ad_hessian.md).

## Forward-Mode AD

You can also compute forward tangents directly:

```rust
use petite_ad::{ForwardAD, MonoAD, mono_ops};

let exprs = mono_ops![sin, exp];
let result = ForwardAD::differentiate(&exprs, 0.5);
assert!(result.value.is_finite());
assert!(result.tangent.is_finite());
```

For multivariate functions, forward mode returns a directional derivative / JVP:

```rust
use petite_ad::{ForwardAD, MultiAD};

let exprs = vec![
    (MultiAD::Inp, vec![0]),
    (MultiAD::Inp, vec![1]),
    (MultiAD::Mul, vec![0, 1]),
];
let result = ForwardAD::directional_derivative(&exprs, &[2.0, 3.0], &[1.0, -1.0]).unwrap();
assert_eq!(result.value, 6.0);
assert_eq!(result.tangent, 1.0);
```

You can also assemble Jacobians for vector outputs represented as multiple scalar graphs:

```rust
use petite_ad::{ForwardAD, MultiAD};

let outputs = vec![
    vec![(MultiAD::Inp, vec![0]), (MultiAD::Inp, vec![1]), (MultiAD::Add, vec![0, 1])],
    vec![(MultiAD::Inp, vec![0]), (MultiAD::Inp, vec![1]), (MultiAD::Mul, vec![0, 1])],
];
let jacobian = ForwardAD::jacobian(&outputs, &[2.0, 3.0]).unwrap();
assert_eq!(jacobian.len(), 2);
```

Reusable graphs can now be validated and exported for visualization:

```rust
use petite_ad::Graph;

let mut graph = Graph::new(1);
let x = graph.input(0);
let neg_x = graph.neg(x);
graph.exp(neg_x);

graph.validate().unwrap();
let mermaid = graph.to_mermaid();
let dot = graph.to_dot();
assert!(mermaid.contains("flowchart LR"));
assert!(dot.contains("digraph Graph"));
```

If you want stricter real-domain behavior, use checked evaluation APIs:

```rust
use petite_ad::{Graph, MonoAD, MultiAD, multi_ops};

let mono_exprs = [MonoAD::Sqrt];
assert!(MonoAD::compute_checked(&mono_exprs, 4.0).is_ok());
assert!(MonoAD::compute_checked(&mono_exprs, -1.0).is_err());

let exprs = multi_ops![(ln, 0)];
assert!(MultiAD::compute_checked(&exprs, &[2.0]).is_ok());
assert!(MultiAD::compute_checked(&exprs, &[0.0]).is_err());

let mut graph = Graph::new(1);
let x = graph.input(0);
graph.ln(x);
assert!(graph.compute_checked(&[2.0]).is_ok());
assert!(graph.compute_checked(&[0.0]).is_err());

let mut multi_graph = Graph::new(2);
let x = multi_graph.input(0);
let y = multi_graph.input(1);
let ratio = multi_graph.div(x, y);
let log_y = multi_graph.ln(y);
multi_graph.set_outputs(&[ratio, log_y]).unwrap();
assert!(multi_graph.compute_many_checked(&[4.0, 2.0]).is_ok());
assert!(multi_graph.jacobian_checked(&[4.0, 2.0]).is_ok());
assert!(multi_graph.compute_many_checked(&[4.0, 0.0]).is_err());
```

### Applications

Second derivatives enable:
- **Newton's method** for optimization (requires accurate Hessians)
- **Convexity analysis** (f''(x) > 0 means convex)
- **Taylor series** approximations
- **Curvature** analysis of functions

See [examples/hessian_demo.rs](examples/hessian_demo.rs) for accuracy comparisons.

## License

MIT

## Contributing

Contributions are welcome! Areas for improvement:

- Vector/matrix operations
- Optimization algorithms (SGD, Adam, etc.)
- Additional mathematical operations
- Higher-order derivatives beyond second-order (third derivatives, etc.)
