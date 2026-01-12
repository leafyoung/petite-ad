# autodiff

A pure Rust automatic differentiation library supporting both single-variable and multi-variable functions with reverse-mode differentiation (backpropagation).

## Features

- **Single-variable autodiff** ([`MonoAD`](src/mono/mono_ad.rs)) - Chain operations like `sin`, `cos`, `exp` with automatic gradient computation
- **Multi-variable autodiff** ([`MultiAD`](src/multi/multi_ad.rs)) - Build computational graphs for functions with multiple inputs
- **Dual wrapper support** - Both `Box<T>` and `Arc<T>` for flexible memory management
- **Zero-copy backward pass** - Gradients computed efficiently through closure chains
- **Convenient macros** - Use `mono_ops![]` for concise operation lists
- **Comprehensive tests** - 22 unit tests covering all operations and edge cases

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
autodiff = "0.1.0"
```

## Quick Start

### Single-Variable Functions

```rust
use autodiff::{mono_ops, MonoAD};

let exprs = mono_ops![sin, cos, exp];
let (value, backprop) = MonoAD::compute(&exprs, 2.0);
let gradient = backprop(1.0);

println!("f(2.0) = {}", value);      // exp(cos(sin(2.0)))
println!("f'(2.0) = {}", gradient);  // derivative
```

### Multi-Variable Functions

```rust
use autodiff::MultiAD;

// Build computational graph: f(x, y) = sin(x) * (x + y)
let exprs = &[
    (MultiAD::Inp, vec![0]),    // x at index 0
    (MultiAD::Inp, vec![1]),    // y at index 1
    (MultiAD::Add, vec![0, 1]), // x + y at index 2
    (MultiAD::Sin, vec![0]),    // sin(x) at index 3
    (MultiAD::Mul, vec![2, 3]), // sin(x) * (x + y) at index 4
];

let inputs = &[0.6, 1.4];
let (value, backprop_fn) = MultiAD::compute_grad(exprs, inputs);
let gradients = backprop_fn(1.0);

println!("f(0.6, 1.4) = {}", value);
println!("∇f = {:?}", gradients);  // [∂f/∂x, ∂f/∂y]
```

## Available Operations

### MonoAD (Single-Variable)
| Operation | Description | Derivative |
|-----------|-------------|------------|
| `Sin` | Sine | `x.cos()` |
| `Cos` | Cosine | `-x.sin()` |
| `Exp` | Exponential | `exp(x)` |

### MultiAD (Multi-Variable)
| Operation | Arity | Description |
|-----------|-------|-------------|
| `Inp` | 0 | Input placeholder |
| `Add` | 2 | Addition: `a + b` |
| `Sub` | 2 | Subtraction: `a - b` |
| `Mul` | 2 | Multiplication: `a * b` |
| `Div` | 2 | Division: `a / b` |
| `Sin` | 1 | Sine: `sin(x)` |
| `Cos` | 1 | Cosine: `cos(x)` |
| `Tan` | 1 | Tangent: `tan(x)` |
| `Exp` | 1 | Exponential: `exp(x)` |
| `Ln` | 1 | Natural log: `ln(x)` |

## API Reference

### MonoAD

#### `compute(exprs: &[MonoAD], x: f64) -> BackwardResult`
Computes forward pass and returns `(value, gradient_fn)` using `Box<dyn Fn>`.

#### `compute_arc(exprs: &[MonoAD], x: f64) -> BackwardResultArc`
Same as `compute` but uses `Arc<dyn Fn>` for shared ownership.

**Return types:**
```rust
type BackwardResult = (f64, Box<dyn Fn(f64) -> f64>);
type BackwardResultArc = (f64, Arc<dyn Fn(f64) -> f64>);
```

### MultiAD

#### `compute(exprs: &[(MultiAD, Vec<usize>)], args: &[f64]) -> f64`
Forward pass only - returns the computed value.

#### `compute_grad(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> BackwardResultBox`
Full forward + backward pass using `Box<dyn Fn>`.

#### `compute_grad_arc(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> BackwardResultArc`
Full forward + backward pass using `Arc<dyn Fn>`.

**Return types:**
```rust
type BackwardResultBox = (f64, Box<dyn Fn(f64) -> Vec<f64>>);
type BackwardResultArc = (f64, Arc<dyn Fn(f64) -> Vec<f64>>);
```

## Box vs Arc

| Wrapper | Pros | Cons | Use Case |
|---------|------|------|----------|
| `Box<T>` | Faster, no atomic overhead | Single ownership | Single-threaded, performance-critical |
| `Arc<T>` | Shared ownership, thread-safe | Slower due to atomics | Multi-threaded, shared gradients |

**Performance:** `Arc` is approximately **1.6-1.8x slower** than `Box` for closure calls due to atomic reference counting overhead.

## Cotangent Values

The backward functions accept a `cotangent` parameter (initial gradient value). Use:
- `1.0` for standard gradients (d/dx of the output)
- Other values for chain rule composition or weighted gradients

```rust
let (_value, backprop) = MonoAD::compute(&exprs, x);
let grad = backprop(1.0);  // ∂f/∂x

// For functions with weights
let weighted_grad = backprop(weight);  // weight * ∂f/∂x
```

## Examples

See [`src/main.rs`](src/main.rs) for complete examples demonstrating:
- Single-variable differentiation
- Multi-variable gradient computation
- Box vs Arc usage
- Verification against analytical gradients

Run examples:
```bash
cargo run
```

## Testing

Run the test suite:
```bash
cargo test
```

All 22 tests verify:
- Correctness of forward and backward passes
- Consistency between Box and Arc implementations
- Gradient accuracy against analytical derivatives
- Edge cases (empty operations, chain rule, etc.)

## Benchmarking

Performance benchmarks powered by Criterion:

```bash
cargo bench --bench compute_benchmark
```

Benchmark groups:
- `single_operation` - Individual operation performance
- `chained_operations` - Chain length scaling (2-20 ops)
- `backprop_only` - Backward pass execution time
- `multi_forward` - Multi-variable forward pass
- `multi_forward_backward` - Full autodiff with gradients
- `multi_backward_only` - Closure call overhead (Box vs Arc)
- `multi_graph_complexity` - Graph size scaling (3-15 ops)

Results are saved to `target/criterion/`.

## Project Structure

```
autodiff/
├── src/
│   ├── lib.rs          # Library root, public exports
│   ├── mono_op.rs      # Single-variable autodiff
│   ├── multi_op.rs     # Multi-variable autodiff
│   ├── macros.rs       # mono_ops! macro definition
│   └── main.rs         # Examples and demonstrations
├── benches/
│   └── compute_benchmark.rs  # Criterion benchmarks
├── Cargo.toml
└── README.md
```

## Implementation Details

### Reverse-Mode Autodiff

This library uses **reverse-mode automatic differentiation**, which:
1. Executes a forward pass, computing values and building a computational graph
2. Executes a backward pass, propagating derivatives from output to inputs using the chain rule
3. Is efficient for functions with many inputs and few outputs (gradients)

### Computational Graph

Multi-variable functions are represented as a list of `(operation, argument_indices)` tuples:
- Each operation produces a value stored at a conceptual index
- `argument_indices` specifies which previous values to use as inputs
- The last operation's value is the final output

Example graph indices:
```rust
// f(x, y) = sin(x) * (x + y)
[
    (Inp, vec![0]),    // index 0 → x
    (Inp, vec![1]),    // index 1 → y
    (Add, vec![0, 1]), // index 2 → x + y
    (Sin, vec![0]),    // index 3 → sin(x)
    (Mul, vec![2, 3]), // index 4 → sin(x) * (x + y)
]
```

## License

MIT

## Contributing

Contributions are welcome! Areas for improvement:
- Additional mathematical operations (pow, sqrt, etc.)
- Higher-order derivatives (Hessian computation)
- Vector/matrix operations
- Optimization algorithms (SGD, Adam, etc.)
