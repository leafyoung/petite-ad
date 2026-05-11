# Multivariate Second-Order Automatic Differentiation

This document provides comprehensive mathematical theory and implementation details for computing exact Hessians of multivariate functions using automatic differentiation, with focus on the Reverse-over-Reverse (RR) method.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Hessian Matrix Theory](#hessian-matrix-theory)
3. [Multivariate Chain Rule](#multivariate-chain-rule)
4. [Second-Order AD Methods](#second-order-ad-methods)
5. [Reverse-over-Reverse (RR) for Multivariate Functions](#reverse-over-reverse-rr-for-multivariate-functions)
6. [Implementation Details](#implementation-details)
7. [Operation Derivatives](#operation-derivatives)
8. [Examples](#examples)
9. [Edge Cases and Numerical Considerations](#edge-cases-and-numerical-considerations)
10. [Performance and Complexity](#performance-and-complexity)
11. [Applications](#applications)
12. [References](#references)

---

## Mathematical Foundations

### Multivariate Functions

A multivariate function maps from ℝⁿ to ℝ:

```
f: ℝⁿ → ℝ
f(x) = f(x₁, x₂, ..., xₙ)
```

where **x** = (x₁, x₂, ..., xₙ) is the input vector.

### First Derivatives: The Gradient

The gradient of f is the vector of first partial derivatives:

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ ∈ ℝⁿ
```

The gradient points in the direction of steepest ascent of f at point x.

### Second Derivatives: The Hessian Matrix

The Hessian matrix H(f) contains all second partial derivatives:

```
H(f) = [∂²f/∂xᵢ∂xⱼ] for i, j = 1, ..., n
```

In matrix form:

```
        ∂²f/∂x₁²      ∂²f/∂x₁∂x₂   ...   ∂²f/∂x₁∂xₙ
H(f) =  ∂²f/∂x₂∂x₁    ∂²f/∂x₂²     ...   ∂²f/∂x₂∂xₙ
         ...          ...         ...    ...
        ∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂   ...   ∂²f/∂xₙ²
```

### Hessian Properties

1. **Symmetry**: For twice-differentiable functions, mixed partials are equal:
   ```
   ∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ
   ```
   Thus, H(f) is a symmetric matrix: H = Hᵀ

2. **Positive Definiteness**: A positive definite Hessian indicates local convexity (strict local minimum)

3. **Negative Definiteness**: A negative definite Hessian indicates local concavity (strict local maximum)

4. **Saddle Point**: Indefinite Hessian indicates a saddle point

---

## Hessian Matrix Theory

### Definition

For a function f: ℝⁿ → ℝ, the Hessian at point x is the n×n symmetric matrix:

```
H(f)(x) = ∇²f(x) = J(∇f(x))
```

where J is the Jacobian operator.

### Geometric Interpretation

The Hessian describes the local curvature of the function:

- **Diagonal elements** (∂²f/∂xᵢ²): Curvature along coordinate axes
- **Off-diagonal elements** (∂²f/∂xᵢ∂xⱼ): Coupling between variables (how xᵢ affects curvature in xⱼ direction)

### Quadratic Approximation (Taylor Series)

The second-order Taylor expansion around point a:

```
f(a + Δx) ≈ f(a) + ∇f(a)ᵀ·Δx + (1/2)·Δxᵀ·H(f)(a)·Δx
```

This is the foundation for Newton's method and second-order optimization.

### Example: Quadratic Form

For a quadratic function `f(x) = xᵀ·A·x` where A is symmetric:

```
∇f(x) = 2Ax
H(f)(x) = 2A
```

The Hessian is constant (independent of x).

### Example: Rosenbrock Function

```
f(x₁, x₂) = (a - x₁)² + b(x₂ - x₁²)²

Hessian entries:
∂²f/∂x₁² = 12b x₁² - 4b x₂ + 2
∂²f/∂x₂² = 2b
∂²f/∂x₁∂x₂ = -4b x₁
```

---

## Multivariate Chain Rule

### Chain Rule for First Derivatives

For a composition h(x) = f(g(x)) where g: ℝⁿ → ℝᵐ and f: ℝᵐ → ℝ:

```
∇h(x) = J_g(x)ᵀ · ∇f(g(x))
```

where J_g(x) is the m×n Jacobian matrix of g at x.

### Chain Rule for Second Derivatives

The multivariate chain rule for Hessians is:

```
∇²h(x) = Σₖ=1ᵐ ∂f/∂uₖ · ∇²uₖ(x) + J_g(x)ᵀ · ∇²f(g(x)) · J_g(x)
```

where u = g(x).

This can be rewritten in a more AD-friendly form:

```
∂²L/∂xᵢ∂xⱼ = Σₖ Σₗ [∂²L/∂uₖ∂uₗ · ∂uₖ/∂xᵢ · ∂uₗ/∂xⱼ] + Σₖ [∂L/∂uₖ · ∂²uₖ/∂xᵢ∂xⱼ]
```

### Derivation of the Multivariate Chain Rule

For h(x) = f(g₁(x), g₂(x), ..., gₘ(x)):

**First derivative:**
```
∂h/∂xᵢ = Σₖ [∂f/∂uₖ · ∂uₖ/∂xᵢ]
```

**Second derivative (differentiate with respect to xⱼ):**
```
∂²h/∂xᵢ∂xⱼ = ∂/∂xⱼ[Σₖ (∂f/∂uₖ · ∂uₖ/∂xᵢ)]

= Σₖ [∂/∂xⱼ(∂f/∂uₖ) · ∂uₖ/∂xᵢ + ∂f/∂uₖ · ∂/∂xⱼ(∂uₖ/∂xᵢ)]

= Σₖ [ Σₗ (∂²f/∂uₖ∂uₗ · ∂uₗ/∂xⱼ) · ∂uₖ/∂xᵢ ] + Σₖ [∂f/∂uₖ · ∂²uₖ/∂xᵢ∂xⱼ]

= Σₖ Σₗ [∂²L/∂uₖ∂uₗ · ∂uₖ/∂xᵢ · ∂uₗ/∂xⱼ] + Σₖ [∂L/∂uₖ · ∂²uₖ/∂xᵢ∂xⱼ]
```

### Special Case: Two Variables

For f(u, v) where u and v depend on x and y:

```
∂²f/∂x² = ∂²f/∂u²·(∂u/∂x)² + 2·∂²f/∂u∂v·(∂u/∂x)(∂v/∂x) + ∂²f/∂v²·(∂v/∂x)² + ∂f/∂u·∂²u/∂x² + ∂f/∂v·∂²v/∂x²

∂²f/∂y² = ∂²f/∂u²·(∂u/∂y)² + 2·∂²f/∂u∂v·(∂u/∂y)(∂v/∂y) + ∂²f/∂v²·(∂v/∂y)² + ∂f/∂u·∂²u/∂y² + ∂f/∂v·∂²v/∂y²

∂²f/∂x∂y = ∂²f/∂u²·(∂u/∂x)(∂u/∂y) + ∂²f/∂u∂v·[(∂u/∂x)(∂v/∂y) + (∂v/∂x)(∂u/∂y)] + ∂²f/∂v²·(∂v/∂x)(∂v/∂y)
          + ∂f/∂u·∂²u/∂x∂y + ∂f/∂v·∂²v/∂x∂y
```

This is the formula we use in the reverse pass.

---

## Second-Order AD Methods

### Overview of Methods

For computing the Hessian matrix, three exact methods exist:

1. **RR (Reverse-over-Reverse)**: Reverse-mode for both gradient and Hessian
2. **FR (Forward-over-Reverse)**: Forward-mode on gradient computation
3. **RF (Reverse-over-Forward)**: Reverse-mode on forward-sensitivities

### Comparison

| Method | Gradient Pass | Hessian Pass | Complexity | Space |
|--------|---------------|--------------|------------|-------|
| **RR** | Reverse | Reverse | O(n²·p) | O(n·p) |
| **FR** | Reverse | Forward | O(n²·p) | O(n·p) |
| **RF** | Forward | Reverse | O(n²·p) | O(n·p) |

where n = number of inputs and p = number of operations.

For scalar outputs (most common case):
- All three methods have similar complexity
- RR is often conceptually clearest
- Choice depends on implementation convenience

### Finite Differences

While exact methods are preferred, finite differences are useful for verification:

```
∂²f/∂xᵢ∂xⱼ ≈ [f(x + h·eᵢ + h·eⱼ) - f(x + h·eᵢ) - f(x + h·eⱼ) + f(x)] / h²
```

where eᵢ is the i-th standard basis vector and h is a small step (e.g., 1e-5).

**Accuracy**: O(h) error, typically 1e-10 relative error with h = 1e-5
**Cost**: O(n²) evaluations (each Hessian entry requires a function evaluation)

---

## Reverse-over-Reverse (RR) for Multivariate Functions

### Core Idea

The RR method applies reverse-mode AD twice:
1. First reverse pass: Compute gradient ∇f(x)
2. Second reverse pass: Compute Hessian by differentiating the gradient computation

### Algorithm Overview

**Forward Pass:**
1. Build computation graph
2. At each node, store:
   - Values vᵢ
   - Jacobians (for multivariate operations) or gradients (for scalar operations)
   - Hessian vectors or Hessians (for operations with multiple inputs)

**Reverse Pass (First - Gradient):**
1. Initialize adjoints: v̅ₙ = 1.0 (for output)
2. Propagate backward: v̅ᵢ = v̅ₙ · ∂vₙ/∂vᵢ
3. Result: ∇f(x) = v̅ (for input variables)

**Reverse Pass (Second - Hessian):**
1. Initialize Hessian seeds:
   - H̅ₙ = 0 (no Hessian at output)
   - v̅ already computed from first pass
2. Propagate backward using multivariate chain rule:
   ```
   H̅ᵢ = Σₖ Σₗ H̅ₖ·∂vₖ/∂vᵢ·∂vₗ/∂vᵢ + Σₖ v̅ₖ·∂²vₖ/∂vᵢ²
   ```
3. Result: H(f)(x) = H̅ (for input variables)

### Two-Variable Case (Binary Operations)

For an operation z = op(u, v):

**Stored during forward pass:**
- u, v: input values
- ∂z/∂u, ∂z/∂v: first partial derivatives
- ∂²z/∂u², ∂²z/∂v², ∂²z/∂u∂v: second partial derivatives

**Reverse pass (Hessian accumulation):**

Let:
- v̅ᵤ, v̅ᵥ: adjoints (gradients) for u and v from first pass
- H̅ᵤᵤ, H̅ᵥᵥ, H̅ᵤᵥ: accumulated Hessians for u and v

The chain rule gives:

```
H̅ᵤᵤ (new) = H̅₂₂·(∂z/∂u)² + v̅₂·∂²z/∂u² + H̅ᵤᵤ (old)
H̅ᵥᵥ (new) = H̅₂₂·(∂z/∂v)² + v̅₂·∂²z/∂v² + H̅ᵥᵥ (old)
H̅ᵤᵥ (new) = H̅₂₂·(∂z/∂u)·(∂z/∂v) + v̅₂·∂²z/∂u∂v + H̅ᵤᵥ (old)
```

where H̅₂₂ is the Hessian coming into node 2 (the operation output).

**Initial conditions:**
- At output: H̅₂₂ = 0, v̅₂ = 1
- At inputs: H̅ᵤᵤ = H̅ᵥᵥ = H̅ᵤᵥ = 0 initially

### Unary Operations

For z = op(u):

```
H̅ᵤᵤ (new) = H̅₂·(∂z/∂u)² + v̅₂·∂²z/∂u² + H̅ᵤᵤ (old)
```

This is the same as the single-variable case.

### Complete Algorithm

```python
# Forward pass
for each operation in computation graph:
    compute (value, grad_u, grad_v, hessian_uu, hessian_vv, hessian_uv)
    store all values and derivatives

# First reverse pass (gradient)
for each variable v:
    adjoint[v] = 0

adjoint[output] = 1.0

for each operation in reverse order:
    for each input u of operation:
        adjoint[u] += adjoint[output] * grad_u

# Second reverse pass (Hessian)
for each pair (i, j) of variables:
    hessian[i][j] = 0

for each operation in reverse order:
    for each pair (u, v) of inputs:
        hessian[u][u] += hessian[output][output] * grad_u * grad_u
                         + adjoint[u] * hessian_uu
        hessian[v][v] += hessian[output][output] * grad_v * grad_v
                         + adjoint[v] * hessian_vv
        hessian[u][v] += hessian[output][output] * grad_u * grad_v
                         + adjoint[output] * hessian_uv

# Result is symmetric: hessian[i][j] = hessian[j][i]
```

---

## Forward-over-Reverse (FR) for Multivariate Functions

### Core Idea

The FR implementation computes exact Hessians with dual numbers and a reverse pass.
For scalar functions `f: ℝⁿ → ℝ`, each seed direction `e_j` is handled independently:

1. **Dual forward pass**: evaluate every RPN node as `(val, tan)`, where `tan`
   is the directional derivative in direction `e_j`.
2. **Dual reverse pass**: propagate dual adjoints backward. Each adjoint stores:
   - `val = ∂f/∂node`
   - `tan = ∂²f/(∂node ∂x_j)`
3. At input node `k`, the adjoint tangent gives Hessian entry `H[j][k]`.

This is exact up to floating-point rounding. It does not use finite differences.

### Implementation Details

```rust
pub fn compute_hessian(ops: &[MultiAD2FR], x: &[f64]) -> Result<Vec<Vec<f64>>> {
    let ops_repr: Vec<OpKind> = ops.iter().map(|&op| OpKind::from(op)).collect();
    compute_hessian_dual(&ops_repr, x)
}
```

The shared `compute_hessian_dual` algorithm first validates the RPN graph shape and
input indices. It then repeats the dual forward + dual reverse computation for each
input dimension.

### Error Handling

The method returns `Err(AutodiffError)` when:

- an input index is out of bounds,
- a unary or binary operation does not have enough operands on the RPN stack,
- the RPN expression leaves anything other than exactly one output on the stack.

---

## Reverse-over-Forward (RF) for Multivariate Functions

### Core Idea

For scalar functions `f: ℝⁿ → ℝ`, the RF implementation is computationally identical
to FR in this crate: both use the same dual forward + dual-adjoint reverse algorithm.
The distinction is conceptual:

- **FR**: forward-mode differentiation of a reverse-mode gradient computation.
- **RF**: reverse-mode differentiation of a forward-mode directional derivative.

Both produce the same exact Hessian for the supported operation set.

### Implementation Details

```rust
pub fn compute_hessian(ops: &[MultiAD2RF], x: &[f64]) -> Result<Vec<Vec<f64>>> {
    let ops_repr: Vec<OpKind> = ops.iter().map(|&op| OpKind::from(op)).collect();
    compute_hessian_dual(&ops_repr, x)
}
```

### Error Handling

RF uses the same shared validator as FR and returns the same `AutodiffError` variants
for malformed graphs or invalid input indices.

### Comparison Summary

| Aspect | RR (Exact) | FR (Exact) | RF (Exact) |
|--------|------------|------------|------------|
| **Precision** | Machine (≈1e-15) | Machine (≈1e-15) | Machine (≈1e-15) |
| **Main Cost** | O(G·n²) | O(n·G) | O(n·G) |
| **Implementation** | Per-node gradient vectors + Hessian accumulation | Shared dual forward + dual reverse | Shared dual forward + dual reverse |
| **Best For** | Small/medium n; direct second-order reverse derivation | Larger n for scalar output | Same as FR for scalar output |

---

## Implementation Details

### Data Structures

For the RR method, we need to store:

**Forward pass storage per node:**
```rust
struct NodeData {
    value: f64,              // Node value
    grad_u: f64,             // ∂output/∂input_u (or None for unary)
    grad_v: f64,             // ∂output/∂input_v (for binary)
    hessian_uu: f64,         // ∂²output/∂input_u²
    hessian_vv: f64,         // ∂²output/∂input_v² (for binary)
    hessian_uv: f64,         // ∂²output/∂input_u∂input_v (for binary)
}
```

**Reverse pass state:**
```rust
struct ReverseState {
    adjoints: Vec<f64>,           // v̅: one per variable
    hessian: Vec<Vec<f64>>,       // H̅: n×n matrix for n variables
}
```

### Memory Management

For efficiency, we can:
1. **Pool allocations**: Reuse memory for adjoints and Hessians
2. **Symmetry**: Store only upper triangular Hessian (H̅ᵢⱼ = H̅ⱼᵢ)
3. **Sparse representation**: For functions with sparse Hessians

### Edge Case Handling

**Division by zero:**
- Guard checks before operations
- Return `NaN` or handle gracefully

**Domain restrictions:**
- `ln(x)` requires x > 0
- `sqrt(x)` requires x ≥ 0
- `tan(x)` has singularities at π/2 + kπ

**Overflow/underflow:**
- Use `f64::is_finite()` checks
- Consider scaling for numerical stability

---

## Operation Derivatives

### Unary Operations

#### Input Variable (Inp)
For input variable x:
```
f(x) = x
∂f/∂x = 1
∂²f/∂x² = 0
```

#### Sine (Sin)
```
f(x) = sin(x)
∂f/∂x = cos(x)
∂²f/∂x² = -sin(x)
```

#### Cosine (Cos)
```
f(x) = cos(x)
∂f/∂x = -sin(x)
∂²f/∂x² = -cos(x)
```

#### Exponential (Exp)
```
f(x) = exp(x)
∂f/∂x = exp(x)
∂²f/∂x² = exp(x)
```

### Binary Operations

#### Addition (Add)
For z = u + v:
```
∂z/∂u = 1, ∂z/∂v = 1
∂²z/∂u² = 0, ∂²z/∂v² = 0, ∂²z/∂u∂v = 0
```

**Hessian contribution:**
```
H̅ᵤᵤ += H̅₂₂·1·1 + v̅₂·0 = H̅₂₂
H̅ᵥᵥ += H̅₂₂·1·1 + v̅₂·0 = H̅₂₂
H̅ᵤᵥ += H̅₂₂·1·1 + v̅₂·0 = H̅₂₂
```

#### Multiplication (Mul)
For z = u · v:
```
∂z/∂u = v, ∂z/∂v = u
∂²z/∂u² = 0, ∂²z/∂v² = 0, ∂²z/∂u∂v = 1
```

**Derivations:**
```
∂z/∂u = ∂(uv)/∂u = v
∂z/∂v = ∂(uv)/∂v = u
∂²z/∂u² = ∂v/∂u = 0
∂²z/∂v² = ∂u/∂v = 0
∂²z/∂u∂v = ∂v/∂v = 1
```

**Hessian contribution:**
```
H̅ᵤᵤ += H̅₂₂·v² + v̅₂·0 = H̅₂₂·v²
H̅ᵥᵥ += H̅₂₂·u² + v̅₂·0 = H̅₂₂·u²
H̅ᵤᵥ += H̅₂₂·uv + v̅₂·1 = H̅₂₂·uv + v̅₂
```

#### Subtraction (Sub)
For z = u - v:
```
∂z/∂u = 1, ∂z/∂v = -1
∂²z/∂u² = 0, ∂²z/∂v² = 0, ∂²z/∂u∂v = 0
```

**Hessian contribution:**
```
H̅ᵤᵤ += H̅₂₂·1·1 + v̅₂·0 = H̅₂₂
H̅ᵥᵥ += H̅₂₂·(-1)·(-1) + v̅₂·0 = H̅₂₂
H̅ᵤᵥ += H̅₂₂·1·(-1) + v̅₂·0 = -H̅₂₂
```

#### Division (Div)
For z = u / v:
```
∂z/∂u = 1/v
∂z/∂v = -u/v²
∂²z/∂u² = 0
∂²z/∂v² = 2u/v³
∂²z/∂u∂v = -1/v²
```

**Derivations:**
```
∂z/∂u = 1/v
∂z/∂v = -u/v²
∂²z/∂u² = 0
∂²z/∂v² = ∂(-u/v²)/∂v = 2u/v³
∂²z/∂u∂v = ∂(1/v)/∂v = -1/v²
```

**Hessian contribution:**
```
H̅ᵤᵤ += H̅₂₂·(1/v)² + v̅₂·0 = H̅₂₂/v²
H̅ᵥᵥ += H̅₂₂·(-u/v²)² + v̅₂·(2u/v³) = H̅₂₂·u²/v⁴ + v̅₂·2u/v³
H̅ᵤᵥ += H̅₂₂·(1/v)·(-u/v²) + v̅₂·(-1/v²) = -H̅₂₂·u/v³ - v̅₂/v²
```

#### Power (Pow)
For z = u^v:
```
∂z/∂u = v·u^(v-1)
∂z/∂v = u^v·ln(u)
∂²z/∂u² = v·(v-1)·u^(v-2)
∂²z/∂v² = u^v·(ln(u))²
∂²z/∂u∂v = u^(v-1)·(1 + v·ln(u))
```

**Derivations:**
```
Let z = u^v = exp(v·ln(u))

∂z/∂u = exp(v·ln(u))·(v/u) = u^v·v/u = v·u^(v-1)
∂z/∂v = exp(v·ln(u))·ln(u) = u^v·ln(u)

∂²z/∂u² = ∂(v·u^(v-1))/∂u = v·(v-1)·u^(v-2)
∂²z/∂v² = ∂(u^v·ln(u))/∂v = u^v·(ln(u))²
∂²z/∂u∂v = ∂(u^v·ln(u))/∂u = u^(v-1)·(1 + v·ln(u))
```

**Note**: Requires u > 0 for ln(u) to be defined.

#### Natural Logarithm (Ln)
For z = ln(u):
```
∂z/∂u = 1/u
∂²z/∂u² = -1/u²
```

**Derivation:**
```
∂z/∂u = 1/u
∂²z/∂u² = ∂(1/u)/∂u = -1/u²
```

**Note**: Requires u > 0.

#### Square Root (Sqrt)
For z = sqrt(u):
```
∂z/∂u = 1/(2·sqrt(u))
∂²z/∂u² = -1/(4·u^(3/2))
```

**Derivations:**
```
∂z/∂u = 1/(2√u)
∂²z/∂u² = ∂(1/(2√u))/∂u = -1/(4·u^(3/2))
```

**Note**: Requires u ≥ 0.

#### Tangent (Tan)
For z = tan(u):
```
∂z/∂u = sec²(u) = 1/cos²(u)
∂²z/∂u² = 2·sin(u)/cos³(u)
```

**Derivations:**
```
∂z/∂u = sec²(u)
∂²z/∂u² = 2·sec(u)·sec(u)·tan(u) = 2·tan(u)·sec²(u) = 2·sin(u)/cos³(u)
```

**Note**: Has singularities where cos(u) = 0.

---

## Examples

### Example 1: Simple Quadratic

**Function:** f(x, y) = x² + y²

**Expected Hessian:**
```
H = [[2, 0], [0, 2]]
```

**Manual computation:**
```
∂f/∂x = 2x, ∂²f/∂x² = 2
∂f/∂y = 2y, ∂²f/∂y² = 2
∂²f/∂x∂y = 0
```

**Using RR:**
```
Computation graph:
x → x² → (value₁)
y → y² → (value₂)
(value₁) + (value₂) → f

Forward pass:
- Node 1: z₁ = x², dz₁/dx = 2x, d²z₁/dx² = 2
- Node 2: z₂ = y², dz₂/dy = 2y, d²z₂/dy² = 2
- Node 3: z₃ = z₁ + z₂
  dz₃/dz₁ = 1, dz₃/dz₂ = 1
  d²z₃/dz₁² = 0, d²z₃/dz₂² = 0, d²z₃/dz₁dz₂ = 0

First reverse pass (gradient):
- v̅₃ = 1.0 (output)
- v̅₂ = v̅₃·dz₃/dz₂ = 1.0
- v̅₁ = v̅₃·dz₃/dz₁ = 1.0

Second reverse pass (Hessian):
Initialize: H̅₃₃ = 0

Node 3 (Add):
H̅₂₂ = H̅₃₃·1² + v̅₃·0 = 0
H̅₁₁ = H̅₃₃·1² + v̅₃·0 = 0
H̅₁₂ = H̅₃₃·1·1 + v̅₃·0 = 0

Node 2 (Mul: y²):
H̅ᵧᵧ = H̅₂₂·(2y)² + v̅₂·2 = 0 + 1.0·2 = 2

Node 1 (Mul: x²):
H̅ₓₓ = H̅₁₁·(2x)² + v̅₁·2 = 0 + 1.0·2 = 2

Result: H = [[2, 0], [0, 2]] ✓
```

### Example 2: Product of Variables

**Function:** f(x, y) = x·y

**Expected Hessian:**
```
H = [[0, 1], [1, 0]]
```

**Manual computation:**
```
∂f/∂x = y, ∂²f/∂x² = 0
∂f/∂y = x, ∂²f/∂y² = 0
∂²f/∂x∂y = 1
```

**Using RR:**
```
Computation graph:
x, y → Mul → f

Forward pass:
- z = x·y
- dz/dx = y, dz/dy = x
- d²z/dx² = 0, d²z/dy² = 0, d²z/dxdy = 1

First reverse pass (gradient):
- v̅ₒᵤₜ = 1.0
- v̅ₓ = v̅ₒᵤₜ·y = y
- v̅ᵧ = v̅ₒᵤₜ·x = x

Second reverse pass (Hessian):
Initialize: H̅ₒᵤₜₒᵤₜ = 0

Node (Mul):
H̅ₓₓ = H̅ₒᵤₜ·y² + v̅ₒᵤₜ·0 = 0
H̅ᵧᵧ = H̅ₒᵤₜ·x² + v̅ₒᵤₜ·0 = 0
H̅ₓᵧ = H̅ₒᵤₜ·xy + v̅ₒᵤₜ·1 = 0 + 1.0·1 = 1

Result: H = [[0, 1], [1, 0]] ✓
```

### Example 3: Composition with Trigonometric Functions

**Function:** f(x, y) = sin(x) + cos(y)

**Expected Hessian:**
```
H = [[-sin(x), 0], [0, -cos(y)]]
```

**Manual computation:**
```
∂f/∂x = cos(x), ∂²f/∂x² = -sin(x)
∂f/∂y = -sin(y), ∂²f/∂y² = -cos(y)
∂²f/∂x∂y = 0
```

**Using RR:**
```
Computation graph:
x → Sin → z₁
y → Cos → z₂
z₁ + z₂ → f

Forward pass:
- z₁ = sin(x), dz₁/dx = cos(x), d²z₁/dx² = -sin(x)
- z₂ = cos(y), dz₂/dy = -sin(y), d²z₂/dy² = -cos(y)
- z₃ = z₁ + z₂
  dz₃/dz₁ = 1, dz₃/dz₂ = 1
  d²z₃/dz₁² = 0, d²z₃/dz₂² = 0, d²z₃/dz₁dz₂ = 0

First reverse pass:
- v̅₃ = 1.0
- v̅₂ = 1.0, v̅₁ = 1.0

Second reverse pass:
Initialize: H̅₃₃ = 0

Node 3 (Add):
H̅₂₂ = 0, H̅₁₁ = 0, H̅₁₂ = 0

Node 2 (Cos):
H̅ᵧᵧ = H̅₂₂·(-sin(y))² + v̅₂·(-cos(y)) = 0 + 1.0·(-cos(y)) = -cos(y)

Node 1 (Sin):
H̅ₓₓ = H̅₁₁·(cos(x))² + v̅₁·(-sin(x)) = 0 + 1.0·(-sin(x)) = -sin(x)

Result: H = [[-sin(x), 0], [0, -cos(y)]] ✓
```

### Example 4: Complex Function

**Function:** f(x, y) = exp(sin(x)·cos(y))

**Expected Hessian (at x=1, y=1):**
```
Let s = sin(1), c = cos(1), e = exp(s·c)

∂f/∂x = e·c·cos(1) = e·c²
∂f/∂y = e·s·(-sin(1)) = -e·s²

∂²f/∂x² = e·c²·c² + e·(-s)·c² = e·c²(c² - s)
∂²f/∂y² = e·s²·s² + e·(-c)·s² = e·s²(s² - c)
∂²f/∂x∂y = e·c²·s² + e·(-c)·s·c = e·c·s(s·c - 1)
```

This demonstrates how complex compositions are handled correctly by the chain rule.

---

## Edge Cases and Numerical Considerations

### Domain Restrictions

**Ln:**
- Requires x > 0
- Implementation: return `NaN` if x ≤ 0
- Alternative: use `ln(x.max(1e-10))` for robustness

**Sqrt:**
- Requires x ≥ 0
- Implementation: return `NaN` if x < 0
- Alternative: use `sqrt(x.abs())` for complex derivatives (not mathematically correct)

**Tan:**
- Singularities at π/2 + k·π
- Implementation: detect singularities, return `NaN`

### Division by Zero

**Division operation:**
- Check if denominator is near zero (|v| < ε)
- Return `NaN` or infinity
- Consider gradient-based regularization

**Example:**
```rust
if v.abs() < 1e-12 {
    return f64::NAN;
}
```

### Numerical Stability

**Exponential overflow:**
- exp(x) → ∞ for large x
- Use `exp(x).min(f64::MAX)` or `exp(x).clamp(...)`

**Loss of precision:**
- For very large/small numbers, use log-space arithmetic
- Example: compute x·y as exp(ln(x) + ln(y))

### Symmetry Enforcement

Mathematically, the Hessian should be symmetric:
```
Hᵢⱼ = Hⱼᵢ
```

Due to floating-point errors, small asymmetries may occur:
```
|Hᵢⱼ - Hⱼᵢ| < ε_machine
```

**Enforcement strategies:**
1. Compute only upper triangle, mirror to lower
2. Average asymmetric entries: Hᵢⱼ = (Hᵢⱼ + Hⱼᵢ)/2
3. Ignore small asymmetries (< 1e-10)

### Verification

**Compare with finite differences:**
```rust
let exact = compute_hessian_rr(x);
let fd = compute_hessian_fd(x, h=1e-5);

assert!((exact - fd).norm() < 1e-7);
```

**Symmetry check:**
```rust
for i in 0..n {
    for j in 0..n {
        assert!((hessian[i][j] - hessian[j][i]).abs() < 1e-10);
    }
}
```

---

## Performance and Complexity

### Time Complexity

**RR method:**
- Forward pass: O(p) where p = number of operations
- First reverse pass (gradient): O(p)
- Second reverse pass (Hessian): O(n²·p)

Total: **O(n²·p)**

where n = number of input variables.

**For sparse Hessians:**
- If only m entries are non-zero: O(m·p)
- Common for structured problems (e.g., neural networks)

### Space Complexity

**RR method:**
- Forward storage: O(p) for values and derivatives
- Reverse state: O(n²) for Hessian

Total: **O(p + n²)**

**Optimizations:**
- Store only upper triangular Hessian: O(n·(n+1)/2)
- Sparse storage for structured problems

### Comparison with Other Methods

| Method | Time | Space | Notes |
|--------|------|-------|-------|
| **RR** | O(n²·p) | O(p + n²) | General purpose, exact |
| **FR** | O(n²·p) | O(p + n²) | Similar to RR |
| **RF** | O(n²·p) | O(p + n²) | Similar to RR |
| **Finite-diff** | O(n²·p) | O(p) | Approximate, no extra storage |

### Benchmarking Considerations

**Factors affecting performance:**
1. Number of variables (n)
2. Number of operations (p)
3. Operation types (unary vs binary)
4. Hessian sparsity
5. Memory bandwidth vs compute

**Optimization strategies:**
1. Vectorization (SIMD) for parallel operations
2. Memory pooling to reduce allocations
3. Sparse matrix representations
4. Compiler optimizations (-O3, LTO)

---

## Applications

### Optimization

**Newton's method:**
```
x_{k+1} = x_k - H(f)(x_k)⁻¹ · ∇f(x_k)
```

Requires computing both gradient and Hessian at each iteration.

**Quasi-Newton methods (BFGS, L-BFGS):**
- Approximate Hessian using gradient information
- Still need exact Hessian for comparison/validation

### Machine Learning

**Second-order optimization:**
- Natural gradient descent
- Gauss-Newton method
- Levenberg-Marquardt algorithm

**Neural network training:**
- Analyzing curvature of loss landscape
- Identifying saddle points
- Adaptive learning rates based on Hessian

### Physics and Engineering

**Structural analysis:**
- Hessian represents stiffness matrix
- Used in finite element analysis

**Control theory:**
- Linearizing nonlinear systems
- Computing observability/controllability

### Statistics and Data Science

**Fisher information matrix:**
- Related to Hessian of log-likelihood
- Used in maximum likelihood estimation

**Gaussian processes:**
- Kernel matrices and uncertainty quantification
- Involves Hessian computations

### Uncertainty Quantification

**Laplace approximation:**
```
p(θ|data) ≈ N(θ̂, H⁻¹)
```

where H is the Hessian of the negative log-likelihood.

**Cramér-Rao bound:**
- Lower bound on estimator variance
- Uses Fisher information (Hessian)

---

## References

### Automatic Differentiation

1. **Griewank, A., & Walther, A.** (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.
2. **Naumann, U.** (2012). *The Art of Differentiating Computer Programs*. Cambridge University Press.
3. **Bischof, C. H., Bücker, H. M., Rasch, A., Slusanschi, E., & Lang, B.** (2002). "Second-Order Derivatives with ADIC". *International Conference on Computational Science*.

### Second-Order AD

4. **Giles, M. B.** (2008). "Collected matrix derivative results for forward and reverse mode algorithmic differentiation". *ACM Transactions on Mathematical Software*, 35(2), 1-18.
5. **Pearlmutter, B. A.** (1994). "Fast exact multiplication by the Hessian". *Neural Computation*, 6(1), 147-160.
6. **Gebremedhin, A. H., Manne, F., & Pothen, A.** (2002). "What color is your Jacobian? Graph coloring for computing derivatives". *SIAM Review*, 47(4), 629-705.

### Multivariate Calculus

7. **Spivak, M.** (1965). *Calculus on Manifolds*. W. A. Benjamin.
8. **Hubbard, J. H., & Hubbard, B. B.** (2009). *Vector Calculus, Linear Algebra, and Differential Forms: A Unified Approach* (4th ed.). Matrix Editions.

### Optimization

9. **Nocedal, J., & Wright, S. J.** (2006). *Numerical Optimization* (2nd ed.). Springer.
10. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.

### Machine Learning

11. **Martens, J.** (2020). "New insights and perspectives on the natural gradient method". *Journal of Machine Learning Research*, 21, 1-76.
12. **Pascanu, R., Dauphin, Y. N., Ganguli, S., & Bengio, Y.** (2014). "On the number of response regions of deep feed forward networks with piece-wise linear activations". *ICLR*.

### Numerical Methods

13. **Higham, N. J.** (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
14. **Golub, G. H., & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.

---

## Appendix A: Hessian Derivative Formulas Summary

### Unary Operations

| Operation | f(x) | f'(x) | f''(x) |
|-----------|------|-------|--------|
| Inp | x | 1 | 0 |
| Sin | sin(x) | cos(x) | -sin(x) |
| Cos | cos(x) | -sin(x) | -cos(x) |
| Exp | exp(x) | exp(x) | exp(x) |
| Ln | ln(x) | 1/x | -1/x² |
| Sqrt | √x | 1/(2√x) | -1/(4x^(3/2)) |
| Tan | tan(x) | sec²(x) | 2·sin(x)/cos³(x) |
| Neg | -x | -1 | 0 |

### Binary Operations

| Operation | f(u,v) | ∂f/∂u | ∂f/∂v | ∂²f/∂u² | ∂²f/∂v² | ∂²f/∂u∂v |
|-----------|--------|-------|-------|--------|--------|----------|
| Add | u+v | 1 | 1 | 0 | 0 | 0 |
| Sub | u-v | 1 | -1 | 0 | 0 | 0 |
| Mul | u·v | v | u | 0 | 0 | 1 |
| Div | u/v | 1/v | -u/v² | 0 | 2u/v³ | -1/v² |
| Pow | u^v | v·u^(v-1) | u^v·ln(u) | v(v-1)u^(v-2) | u^v(ln u)² | u^(v-1)(1+v·ln u) |

---

## Appendix B: Algorithm Pseudocode

### RR Method for Two Variables

```
function compute_hessian_rr(operations, x, y):
    # Forward pass
    nodes = []
    value_u = x
    value_v = y

    for op in operations:
        if op is unary:
            (value_u, grad_u, hess_u) = op.forward_d2(value_u)
            nodes.append((value_u, grad_u, None, hess_u, None, None))
        else:  # binary
            (value, grad_u, grad_v, hess_uu, hess_vv, hess_uv) = op.forward_d2(value_u, value_v)
            nodes.append((value, grad_u, grad_v, hess_uu, hess_vv, hess_uv))

    # First reverse pass (gradient)
    adjoints_u = 0.0
    adjoints_v = 0.0
    adjoint_out = 1.0

    for node in reversed(nodes):
        (value, grad_u, grad_v, hess_uu, hess_vv, hess_uv) = node
        if grad_v is None:  # unary
            adjoints_u += adjoint_out * grad_u
        else:  # binary
            adjoints_u += adjoint_out * grad_u
            adjoints_v += adjoint_out * grad_v
        adjoint_out = adjoint  # Move to next node

    # Second reverse pass (Hessian)
    hessian = [[0.0, 0.0], [0.0, 0.0]]
    hessian_out = 0.0

    for node in reversed(nodes):
        (value, grad_u, grad_v, hess_uu, hess_vv, hess_uv) = node

        if grad_v is None:  # unary
            hessian[0][0] += hessian_out * grad_u * grad_u + adjoints_u * hess_uu
        else:  # binary
            hessian[0][0] += hessian_out * grad_u * grad_u + adjoints_u * hess_uu
            hessian[1][1] += hessian_out * grad_v * grad_v + adjoints_v * hess_vv
            hessian[0][1] += hessian_out * grad_u * grad_v + adjoints_u * hess_uv

        hessian_out = hessian_temp

    return hessian
```

---

## Appendix C: Implementation Checklist

### Required Components

- [x] Theory document (this file)
- [x] MultiAD2RR enum with operations
- [x] forward_d2 method for each operation
- [x] compute_hessian method implementing RR
- [x] MultiAD2FR enum with operations (exact dual forward + dual reverse)
- [x] MultiAD2RF enum with operations (exact dual forward + dual reverse)
- [x] Test suite (basic, analytical, edge cases)
- [x] Documentation and examples
- [x] Performance benchmarks

### Testing Checklist

- [x] Basic correctness tests (each operation)
- [x] Analytical comparison tests (verify against hand-derived formulas)
- [x] Edge case tests (division by zero, domain restrictions)
- [x] Symmetry verification (H[i][j] = H[j][i])
- [x] Comparison with finite differences
- [x] Performance benchmarks
- [x] Trigonometric tests (Sin, Cos)
- [x] Exponential tests (Exp)

### Documentation Checklist

- [x] Module-level documentation
- [x] Function-level documentation with formulas
- [x] Inline comments explaining chain rule
- [x] Examples with verification
- [x] Cross-references to theory document
- [x] API documentation
- [x] FR/RF exact dual-adjoint implementation documentation
- [x] Method trade-offs comparison

### Implementation Status (January 2026)

**Completed:**
- ✅ MultiAD2RR (Reverse-over-Reverse) - Exact symbolic differentiation
- ✅ MultiAD2FR (Forward-over-Reverse) - Exact dual forward + dual reverse
- ✅ MultiAD2RF (Reverse-over-Forward) - Exact dual forward + dual reverse
- ✅ Comprehensive test suite covering exact, finite-difference, and error-path behavior
- ✅ Performance benchmark suite (RR/FR/RF comparison)
- ✅ Documentation with mathematical theory and implementation details

**Test Coverage:**
- 6 tests per method (RR, FR, RF)
- Operations tested: Add, Mul, Sin, Cos, Exp
- All tests passing with 100% success rate
- Exact methods (RR): Machine precision tolerance (1e-12)
- Finite difference methods (FR/RF): 1e-3 tolerance

---

*Last updated: January 2026*
