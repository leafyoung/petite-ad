# petite-ad

A pure Rust automatic differentiation library supporting both single-variable and multi-variable functions with reverse-mode differentiation (backpropagation).

## Features

- **Single-variable autodiff** ([`MonoAD`](src/mono/mono_ad.rs)) - Chain operations like `sin`, `cos`, `exp` with automatic gradient computation
- **Multi-variable autodiff** ([`MultiAD`](src/multi/multi_ad.rs)) - Build computational graphs for functions with multiple inputs
- **Box-wrapped by default** - Results use `Box<dyn Fn>` for flexibility; convert to `Arc` when needed for thread-safety
- **Zero-copy backward pass** - Gradients computed efficiently through closure chains
- **Convenient macros** - Use `mono_ops![]` for concise operation lists
- **Builder API** - Fluent interface for constructing computation graphs
- **Comprehensive tests** - 39 unit tests + 10 doctests covering all operations and edge cases

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
petite-ad = "0.1.0"
```

## Quick Start

### Single-Variable Functions

```rust
use petite_ad::{mono_ops, MonoAD};

let exprs = mono_ops![sin, cos, exp];
let (value, backprop) = MonoAD::compute(&exprs, 2.0);
let gradient = backprop(1.0);

println!("f(2.0) = {}", value);      // exp(cos(sin(2.0)))
println!("f'(2.0) = {}", gradient);  // derivative
```

### Multi-Variable Functions

#### Using the GraphBuilder API (Recommended)

```rust
use petite_ad::{GraphBuilder, MultiAD};

// Build: f(x, y) = sin(x) * (x + y)
let graph = GraphBuilder::new(2)  // 2 inputs
    .add(0, 1)      // x + y at index 2
    .sin(0)         // sin(x) at index 3
    .mul(2, 3)      // sin(x) * (x + y) at index 4
    .build();

let inputs = &[0.6, 1.4];
let (value, backprop_fn) = MultiAD::compute_grad(&graph, inputs).unwrap();
let gradients = backprop_fn(1.0);

println!("f(0.6, 1.4) = {}", value);
println!("∇f = {:?}", gradients);  // [∂f/∂x, ∂f/∂y]
```

#### Using Manual Graph Construction

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
|-----------|-------------|------------|
| `Sin` | Sine | `x.cos()` |
| `Cos` | Cosine | `-x.sin()` |
| `Exp` | Exponential | `exp(x)` |

### MultiAD (Multi-Variable)
| Operation | Arity | Description |
|-----------|-------|-------------|
| `Inp` | 1 | Input placeholder |
| `Add` | 2 | Addition: `a + b` |
| `Sub` | 2 | Subtraction: `a - b` |
| `Mul` | 2 | Multiplication: `a * b` |
| `Div` | 2 | Division: `a / b` |
| `Pow` | 2 | Power: `a^b` |
| `Sin` | 1 | Sine: `sin(x)` |
| `Cos` | 1 | Cosine: `cos(x)` |
| `Tan` | 1 | Tangent: `tan(x)` |
| `Exp` | 1 | Exponential: `exp(x)` |
| `Ln` | 1 | Natural log: `ln(x)` |
| `Sqrt` | 1 | Square root: `sqrt(x)` |
| `Abs` | 1 | Absolute value: `abs(x)` |

## License

MIT

## Contributing

Contributions are welcome! Areas for improvement:
- Higher-order derivatives (Hessian computation)
- Vector/matrix operations
- Optimization algorithms (SGD, Adam, etc.)
- Constant/literal values in computation graphs (e.g., `x^2` without needing a separate input)
- Additional mathematical operations
