
## API Reference

### MonoAD

#### `compute_grad(exprs: &[MonoAD], x: f64) -> BackwardResultBox`
Computes forward pass and returns `(value, gradient_fn)` using `Box<dyn Fn>`.

**Convert to Arc if needed:**
```rust
let (value, grad_fn) = MonoAD::compute_grad(&exprs, x);
let arc_grad_fn: Arc<dyn Fn(f64) -> f64> = Arc::from(grad_fn);
```

**Return type:**
```rust
type BackwardResultBox = (f64, Box<dyn Fn(f64) -> f64>);
```

### MultiAD

#### `compute(exprs: &[(MultiAD, Vec<usize>)], args: &[f64]) -> f64`
Forward pass only - returns the computed value.

#### `compute_grad(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> BackwardResultBox`
Full forward + backward pass using `Box<dyn Fn>`.

**Convert to Arc if needed:**
```rust
let (value, grad_fn) = MultiAD::compute_grad(&exprs, inputs);
let arc_grad_fn: Arc<dyn Fn(f64) -> Vec<f64>> = Arc::from(grad_fn);
```

**Return type:**
```rust
type BackwardResultBox = (f64, Box<dyn Fn(f64) -> Vec<f64>>);
```

### GraphBuilder

Fluent API for building computation graphs without manually managing indices.

#### `new(num_inputs: usize) -> GraphBuilder`
Creates a new builder with the specified number of input variables.

#### Builder Methods
All methods return `&mut Self` for chaining:
- `.sin(arg_index)` - Add sine operation
- `.cos(arg_index)` - Add cosine operation
- `.tan(arg_index)` - Add tangent operation
- `.exp(arg_index)` - Add exponential operation
- `.ln(arg_index)` - Add natural logarithm operation
- `.sqrt(arg_index)` - Add square root operation
- `.abs(arg_index)` - Add absolute value operation
- `.add(left, right)` - Add addition operation
- `.sub(left, right)` - Add subtraction operation
- `.mul(left, right)` - Add multiplication operation
- `.div(left, right)` - Add division operation
- `.pow(base, exp)` - Add power operation

#### `build() -> Vec<(MultiAD, Vec<usize>)>`
Builds the final computation graph for use with `MultiAD::compute()` or `MultiAD::compute_grad()`.

**Example:**
```rust
use petite_ad::GraphBuilder;

let graph = GraphBuilder::new(3)
    .pow(0, 1)     // x^y at index 3
    .add(3, 2)     // x^y + z at index 4
    .build();
```

## Box vs Arc

The library defaults to `Box<dyn Fn>` for better performance. Convert to `Arc` when you need:
- **Thread-safe sharing** - Clone and send gradients across threads
- **Multiple ownership** - Store gradient functions in multiple locations

```rust
// Default Box for single-threaded use
let (value, grad_fn) = MonoAD::compute_grad(&exprs, x);

// Convert to Arc for thread-safe sharing
let arc_grad_fn: Arc<dyn Fn(f64) -> f64> = Arc::from(grad_fn);
let clone = arc_grad_fn.clone(); // Can clone Arc
```

**Performance Note:** `Arc` has slight overhead (~1.1-1.2x) due to atomic reference counting, but the difference is negligible for most autodiff workloads where the computation dominates.

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

See [`src/main.rs`](src/main.rs) for complete demonstrations of:
- Single-variable differentiation with `MonoAD`
- Multi-variable differentiation with `MultiAD`
- Using the convenient `multi_ops![]` macro syntax

Run the demo:
```bash
cargo run --release
```

## Testing

Run the test suite:
```bash
cargo test
```

All 39 tests verify:
- Correctness of forward and backward passes
- Consistency between Box and Arc implementations
- Gradient accuracy against analytical derivatives
- Edge cases (empty operations, chain rule, etc.)
- Builder API correctness

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
petite_ad/
├── src/
│   ├── lib.rs          # Library root, public exports
│   ├── mono.rs         # Single-variable autodiff
│   ├── multi.rs        # Multi-variable autodiff
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
