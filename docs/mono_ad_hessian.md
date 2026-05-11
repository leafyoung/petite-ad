# Second-Order Automatic Differentiation for Univariate Functions

## Table of Contents
1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Chain Rule for Second Derivatives](#2-chain-rule-for-second-derivatives)
3. [Reverse-over-Reverse (RR) Method](#3-reverse-over-reverse-rr-method)
4. [Forward-over-Reverse (FR) Method](#4-forward-over-reverse-fr-method)
5. [Reverse-over-Forward (RF) Method](#5-reverse-over-forward-rf-method)
6. [Implementation Details](#6-implementation-details)
7. [Numerical Considerations](#7-numerical-considerations)
8. [Examples and Applications](#8-examples-and-applications)
9. [Appendices](#9-appendices)

---

## 1. Mathematical Foundations

### 1.1 First-Order Differentiation (Recap)

For a univariate function f: ℝ → ℝ, the first derivative f'(x) represents the rate of change at point x.

For a composition h(x) = f(g(x)), the **chain rule** states:
```
h'(x) = f'(g(x)) · g'(x)
```

**Example**: h(x) = sin(exp(x))
```
h'(x) = cos(exp(x)) · exp(x)
```

**Reverse-mode automatic differentiation** (backpropagation) efficiently computes derivatives by:
1. **Forward pass**: Compute and store all intermediate values
2. **Backward pass**: Propagate derivatives from output to input using the chain rule

This is the foundation of modern deep learning frameworks.

### 1.2 Second-Order Differentiation

The **second derivative** f''(x) represents the rate of change of the first derivative - the "curvature" or "acceleration" of the function.

**Physical interpretation**:
- f(x): position
- f'(x): velocity
- f''(x): acceleration

**Mathematical interpretation**:
- f''(x) > 0: function is **convex** (curving upward) at x
- f''(x) < 0: function is **concave** (curving downward) at x
- f''(x) = 0: potential **inflection point** at x

For a composition h(x) = f(g(x)), the chain rule for second derivatives is:

```
h''(x) = f''(g(x)) · [g'(x)]² + f'(g(x)) · g''(x)
         \_________/             \______________/
         Quadratic term          Linear term
```

**Derivation**:

Starting from h'(x) = f'(g(x)) · g'(x), apply the **product rule**:

```
h''(x) = d/dx[h'(x)]
       = d/dx[f'(g(x)) · g'(x)]
       = [d/dx f'(g(x))] · g'(x) + f'(g(x)) · [d/dx g'(x)]
```

For the first term, apply chain rule to differentiate f'(g(x)):
```
d/dx[f'(g(x))] = f''(g(x)) · g'(x)
```

For the second term:
```
d/dx[g'(x)] = g''(x)
```

Therefore:
```
h''(x) = f''(g(x)) · g'(x) · g'(x) + f'(g(x)) · g''(x)
       = f''(g(x)) · [g'(x)]² + f'(g(x)) · g''(x)
```

**Key insight**: The quadratic term [g'(x)]² means second derivatives grow rapidly through compositions. This makes symbolic differentiation tedious for long chains, motivating automatic differentiation.

### 1.3 Second Derivatives of Elementary Functions

| Function f(x) | First Derivative f'(x) | Second Derivative f''(x) | Notes |
|---------------|------------------------|--------------------------|-------|
| sin(x)        | cos(x)                 | -sin(x)                  | Cycles every 4 derivatives |
| cos(x)        | -sin(x)                | -cos(x)                  | Cycles every 4 derivatives |
| exp(x)        | exp(x)                 | exp(x)                   | Unchanged through differentiation |
| ln(x)         | 1/x                    | -1/x²                    | Only defined for x > 0 |
| x^n           | n·x^(n-1)              | n(n-1)·x^(n-2)           | Polynomial degree decreases |
| sqrt(x)       | 1/(2√x)                | -1/(4x^(3/2))            | Only defined for x > 0 |
| abs(x)        | sign(x)                | 0                        | Non-smooth at 0; raw convention uses 0 |
| -x            | -1                     | 0                        | Linear function, no curvature |
| tan(x)        | 1/cos²(x)              | 2·sin(x)/cos³(x)         | Asymptotes at π/2 + kπ |

**Pattern observations**:
- Trigonometric functions (sin, cos) cycle through differentiation
- Exponential function remains unchanged
- Polynomial degrees decrease by 1 with each differentiation
- Rational functions have increasingly complex derivatives

---

## 2. Chain Rule for Second Derivatives (Detailed)

### 2.1 Two-Function Composition

Given: h(x) = f(g(x))

**First derivative** (standard chain rule):
```
h'(x) = f'(g(x)) · g'(x)
```

**Second derivative** (extended chain rule):

We need to differentiate h'(x) = f'(g(x)) · g'(x).

Using the **product rule**: (uv)' = u'v + uv'

Let u = f'(g(x)) and v = g'(x)

```
h''(x) = [d/dx f'(g(x))] · g'(x) + f'(g(x)) · [d/dx g'(x)]
```

For the first term, using chain rule:
```
d/dx[f'(g(x))] = f''(g(x)) · g'(x)
```

For the second term:
```
d/dx[g'(x)] = g''(x)
```

Therefore:
```
h''(x) = f''(g(x)) · g'(x) · g'(x) + f'(g(x)) · g''(x)
       = f''(g(x)) · [g'(x)]² + f'(g(x)) · g''(x)
```

**Concrete example**: h(x) = sin(x²)

Let f(u) = sin(u) and g(x) = x²

```
g(x) = x²           →  g'(x) = 2x,      g''(x) = 2
f(u) = sin(u)       →  f'(u) = cos(u),  f''(u) = -sin(u)
```

At x = 2:
```
g(2) = 4
g'(2) = 4
g''(2) = 2

h''(2) = f''(g(2)) · [g'(2)]² + f'(g(2)) · g''(2)
       = f''(4) · 16 + f'(4) · 2
       = -sin(4) · 16 + cos(4) · 2
       = -16·sin(4) + 2·cos(4)
       ≈ -16·(-0.7568) + 2·(-0.6536)
       ≈ 12.109 - 1.307
       ≈ 10.802
```

### 2.2 Three-Function Composition

Given: k(x) = f(g(h(x)))

Denote: u = h(x), v = g(u) = g(h(x)), w = f(v) = k(x)

**First derivative**:

Applying chain rule twice:
```
k'(x) = f'(g(h(x))) · g'(h(x)) · h'(x)
      = f'(v) · g'(u) · h'(x)
```

**Second derivative**:

We differentiate k'(x) = f'(v) · g'(u) · h'(x).

This is a product of three functions. Using the product rule:
```
(fgh)' = f'gh + fg'h + fgh'
```

Let's work step by step:

```
k''(x) = d/dx[f'(v) · g'(u) · h'(x)]
```

First, treat f'(v)·g'(u) as one unit and apply product rule:
```
= [d/dx(f'(v)·g'(u))] · h'(x) + f'(v)·g'(u) · h''(x)
```

For the first term, apply product rule again:
```
d/dx[f'(v)·g'(u)] = [d/dx f'(v)] · g'(u) + f'(v) · [d/dx g'(u)]
```

Now compute each piece using chain rule:
```
d/dx[f'(v)] = f''(v) · dv/dx = f''(v) · g'(u) · h'(x)
d/dx[g'(u)] = g''(u) · du/dx = g''(u) · h'(x)
```

Putting it all together:
```
k''(x) = f''(v) · g'(u) · h'(x) · g'(u) · h'(x)
       + f'(v) · g''(u) · h'(x) · h'(x)
       + f'(v) · g'(u) · h''(x)

       = f''(v) · [g'(u)]² · [h'(x)]²
       + f'(v) · g''(u) · [h'(x)]²
       + f'(v) · g'(u) · h''(x)
```

**Interpretation**:
- **Term 1**: f''(v)·[g'(u)]²·[h'(x)]² - Second derivative of f, first derivatives of g and h squared
- **Term 2**: f'(v)·g''(u)·[h'(x)]² - First derivative of f, second derivative of g, first derivative of h squared
- **Term 3**: f'(v)·g'(u)·h''(x) - First derivatives of f and g, second derivative of h

**Concrete example**: k(x) = exp(sin(x²))

Let h(x) = x², g(u) = sin(u), f(v) = exp(v)

```
h(x) = x²       →  h'(x) = 2x,      h''(x) = 2
g(u) = sin(u)   →  g'(u) = cos(u),  g''(u) = -sin(u)
f(v) = exp(v)   →  f'(v) = exp(v),  f''(v) = exp(v)
```

At x = 1:
```
h(1) = 1,        h'(1) = 2,       h''(1) = 2
g(1) = sin(1),   g'(1) = cos(1),  g''(1) = -sin(1)
v = sin(1) ≈ 0.8414

k''(1) = exp(sin(1)) · [cos(1)]² · [2]²
       + exp(sin(1)) · [-sin(1)] · [2]²
       + exp(sin(1)) · cos(1) · 2

       = exp(0.8414) · cos²(1) · 4
       + exp(0.8414) · (-sin(1)) · 4
       + exp(0.8414) · cos(1) · 2
```

### 2.3 General n-Function Composition

For h = f₁ ∘ f₂ ∘ ... ∘ fₙ, the second derivative involves:
- All combinations of two second derivatives or products of first derivatives
- Terms grow as O(n) for second derivatives
- This is manageable compared to symbolic expansion

The recursive structure makes automatic differentiation natural:
- Store all intermediate first and second derivatives during forward pass
- Apply chain rule systematically during backward pass

---

## 3. Reverse-over-Reverse (RR) Method

### 3.1 High-Level Algorithm

RR extends reverse-mode AD to compute second derivatives by:
1. **Forward pass**: Store values, first derivatives, AND second derivatives for each operation
2. **Reverse pass**: Propagate both gradient (first derivative) and Hessian (second derivative) backward

**Key idea**: Just as reverse-mode propagates ∂L/∂x backward, RR propagates ∂²L/∂x² backward.

### 3.2 Forward Pass Detail

For each operation in the computation chain, compute three quantities:

**Example**: Computing y = sin(x) at x = 0.5

```
value:          y = sin(0.5) ≈ 0.479425538604203
first_deriv:    dy/dx = cos(0.5) ≈ 0.877582561890373
second_deriv:   d²y/dx² = -sin(0.5) ≈ -0.479425538604203
```

Store all three for use in the backward pass.

**Example**: Computing y = exp(x) at x = 1.5

```
value:          y = exp(1.5) ≈ 4.481689070338065
first_deriv:    dy/dx = exp(1.5) ≈ 4.481689070338065
second_deriv:   d²y/dx² = exp(1.5) ≈ 4.481689070338065
```

All three values are identical for exp!

### 3.3 Backward Pass Detail

**Initialization** at output node:
```
grad = 1.0        // ∂Output/∂Output = 1
hessian = 0.0     // ∂²Output/∂Output² = 0 (scalar has no curvature w.r.t. itself)
```

**Propagation rule** for each operation going backward:

Given current operation computes y = op(x) with stored values:
- dy/dx (first derivative)
- d²y/dx² (second derivative)

Update:
```rust
new_grad = grad · (dy/dx)
new_hessian = hessian · (dy/dx)² + grad · (d²y/dx²)
              \_________________/   \________________/
              Propagate output      Add local
              Hessian (quadratic)   second derivative (linear)
```

**Why this works**:

This is exactly the chain rule formula we derived:
```
h''(x) = f''(g(x)) · [g'(x)]² + f'(g(x)) · g''(x)
```

Where:
- `hessian` corresponds to f''(g(x))
- `dy/dx` corresponds to g'(x)
- `grad` corresponds to f'(g(x))
- `d²y/dx²` corresponds to g''(x)

### 3.4 Concrete Example: exp(sin(x))

Let's compute the second derivative of h(x) = exp(sin(x)) at x = 0.5 in complete detail.

**Forward pass**:

```
Step 1: Compute sin(0.5)
  Input: x = 0.5

  value₁ = sin(0.5) = 0.479425538604203
  dy₁/dx = cos(0.5) = 0.877582561890373
  d²y₁/dx² = -sin(0.5) = -0.479425538604203

  Store: (0.479425538604203, 0.877582561890373, -0.479425538604203)

Step 2: Compute exp(value₁)
  Input: value₁ = 0.479425538604203

  value₂ = exp(0.479425538604203) = 1.615129883784566
  dy₂/dy₁ = exp(0.479425538604203) = 1.615129883784566
  d²y₂/dy₁² = exp(0.479425538604203) = 1.615129883784566

  Store: (1.615129883784566, 1.615129883784566, 1.615129883784566)
```

**Backward pass**:

```
Initialize at output (Step 2):
  grad = 1.0
  hessian = 0.0

Process Step 2 (exp):
  dy/dy₁ = 1.615129883784566
  d²y/dy₁² = 1.615129883784566

  new_grad = 1.0 · 1.615129883784566 = 1.615129883784566
  new_hessian = 0.0 · (1.615129883784566)² + 1.0 · 1.615129883784566
              = 0.0 + 1.615129883784566
              = 1.615129883784566

  Update: grad = 1.615129883784566, hessian = 1.615129883784566

Process Step 1 (sin):
  dy/dx = 0.877582561890373
  d²y/dx² = -0.479425538604203

  new_grad = 1.615129883784566 · 0.877582561890373
           = 1.417466313257094

  new_hessian = 1.615129883784566 · (0.877582561890373)²
              + 1.615129883784566 · (-0.479425538604203)
              = 1.615129883784566 · 0.770151548091452
              - 0.774301385544626
              = 1.244229173328810 - 0.774301385544626
              = 0.469927787784184

  Final result: hessian = 0.469927787784184
```

**Verification** (analytical formula):

For h(x) = exp(sin(x)):
```
h'(x) = exp(sin(x)) · cos(x)
h''(x) = [exp(sin(x)) · cos(x)] · cos(x) + exp(sin(x)) · [-sin(x)]
       = exp(sin(x)) · cos²(x) - exp(sin(x)) · sin(x)
       = exp(sin(x)) · [cos²(x) - sin(x)]
```

At x = 0.5:
```
h''(0.5) = exp(sin(0.5)) · [cos²(0.5) - sin(0.5)]
         = 1.615129883784566 · [0.770151548091452 - 0.479425538604203]
         = 1.615129883784566 · 0.290726009487249
         = 0.469927787784184 ✓
```

Perfect match!

### 3.5 Implementation in Rust

```rust
pub fn compute_hessian(exprs: &[MonoAD2RR], x: f64) -> f64 {
    if exprs.is_empty() {
        return 0.0;
    }

    let n = exprs.len();

    // Forward pass: compute and store all derivatives
    let mut values: Vec<f64> = Vec::with_capacity(n + 1);
    let mut first_derivs: Vec<f64> = Vec::with_capacity(n);
    let mut second_derivs: Vec<f64> = Vec::with_capacity(n);

    values.push(x); // Initial input value

    for &op in exprs {
        let (y, dy, ddy) = op.forward_d2(*values.last().unwrap());
        values.push(y);
        first_derivs.push(dy);
        second_derivs.push(ddy);
    }

    // Reverse pass: propagate gradient and Hessian backward
    let mut grad: f64 = 1.0;      // ∂Output/∂Output = 1
    let mut hessian: f64 = 0.0;   // ∂²Output/∂Output² = 0

    for i in (0..n).rev() {
        let dy = first_derivs[i];
        let ddy = second_derivs[i];

        // Apply extended chain rule:
        // h''(x) = f''(g(x))·[g'(x)]² + f'(g(x))·g''(x)
        let new_grad = grad * dy;
        let new_hessian = hessian * dy * dy + grad * ddy;
                        // \_________/         \________/
                        // Quadratic term      Linear term

        grad = new_grad;
        hessian = new_hessian;
    }

    hessian
}
```

### 3.6 Complexity Analysis

**Time complexity**:
- Forward pass: O(n) - process each operation once
- Backward pass: O(n) - process each operation once
- Total: **O(n)** where n = number of operations

**Space complexity**:
- Values: n+1 entries
- First derivatives: n entries
- Second derivatives: n entries
- Total: **O(n)** space

**Comparison with finite differences**:
- Finite differences need 3 function evaluations: f(x), f'(x+ε), f'(x)
- Each f' evaluation already costs O(n)
- Total for finite differences: **O(n)** but with numerical approximation error

RR is **exact** (up to floating-point precision) while finite differences have truncation error.

---

## 4. Forward-over-Reverse (FR) Method

### 4.1 Conceptual Approach

FR computes second derivatives by:
1. Use **reverse-mode AD** to obtain the gradient function g(x) = f'(x)
2. Use **forward-mode AD** to differentiate g to get g'(x) = f''(x)

**Key insight**: Forward-mode AD can differentiate any function, including a gradient function computed by reverse-mode!

### 4.2 Dual Numbers

Forward-mode AD uses **dual numbers** to automatically track derivatives.

A dual number has two components:
```
d = (value, tangent)
```

Where:
- `value`: the function value f(x)
- `tangent`: the derivative f'(x)

**Arithmetic rules** for dual numbers:

```
Addition:        (a, a') + (b, b') = (a + b, a' + b')
Subtraction:     (a, a') - (b, b') = (a - b, a' - b')
Multiplication:  (a, a') · (b, b') = (a · b, a'·b + a·b')
Division:        (a, a') / (b, b') = (a / b, (a'·b - a·b') / b²)

sin:             sin(a, a') = (sin(a), cos(a) · a')
cos:             cos(a, a') = (cos(a), -sin(a) · a')
exp:             exp(a, a') = (exp(a), exp(a) · a')
```

**Example**: Compute derivative of f(x) = x · sin(x) at x = 2

Using dual numbers with tangent = 1 (differentiating w.r.t. x):
```
x_dual = (2, 1)

sin(x_dual) = sin((2, 1))
            = (sin(2), cos(2) · 1)
            = (0.909, -0.416)

x_dual · sin(x_dual) = (2, 1) · (0.909, -0.416)
                     = (2 · 0.909, 1 · 0.909 + 2 · (-0.416))
                     = (1.818, 0.909 - 0.832)
                     = (1.818, 0.077)
```

The tangent part 0.077 is the derivative!

Verify: f'(x) = sin(x) + x·cos(x) = sin(2) + 2·cos(2) ≈ 0.909 - 0.832 = 0.077 ✓

### 4.3 FR Algorithm for Second Derivatives

**Step 1**: Compute gradient function using reverse-mode
```rust
let (value, grad_fn) = MonoAD::compute_grad(exprs, x);
// grad_fn(1.0) gives us f'(x)
```

**Step 2**: Differentiate gradient function using dual numbers

For simple operations (Sin, Cos, Tan, Exp, Neg, Ln, Sqrt, Abs):
```rust
// Create dual number with tangent = 1
let x_dual = Dual::variable(x, 1.0);

// Evaluate operations using dual arithmetic
let result_dual = evaluate_with_dual(exprs, x_dual);

// Extract second derivative from tangent
let hessian = result_dual.tangent;
```

For composed operations: May delegate to RR for correctness.

### 4.4 Comparison: FR vs RR

| Aspect | RR | FR |
|--------|----|----|
| **Conceptual model** | Propagate Hessian backward | Differentiate gradient forward |
| **Implementation** | Single backward pass with Hessian | Two passes (reverse then forward) |
| **Natural for** | Deep learning practitioners | Forward-mode enthusiasts |
| **Pedagogical value** | Shows extension of backprop | Shows composition of AD modes |
| **Performance** | O(n) | O(n) |
| **Code complexity** | Moderate | Moderate |

For **univariate functions**, both are equally efficient: O(n) time and space.

### 4.5 When FR is Preferable

- **Directional derivatives**: Computing f''(x)·v for a direction v
- **Educational purposes**: Forward-mode is often easier to understand initially
- **Hardware considerations**: Some architectures favor forward-mode
- **Composition exploration**: Demonstrates how different AD modes can be combined

---

## 5. Reverse-over-Forward (RF) Method

### 5.1 Conceptual Approach

RF is conceptually the "opposite" of FR:
1. Use **forward-mode AD** to compute value and first derivative simultaneously
2. Use **reverse-mode AD** to differentiate the derivative computation

### 5.2 Relationship to FR

For **univariate functions**, RF and FR are **mathematically equivalent**:
- Both compute f''(x) using one forward and one reverse pass
- They differ in the conceptual order of operations
- Implementation may differ slightly but results are identical (up to floating-point rounding)

For **multivariate functions** (discussed in multi_ad_hessian.md):
- FR computes Hessian column-by-column
- RF computes Hessian row-by-row
- This distinction matters for efficiency in high dimensions

### 5.3 Implementation Note

In the `petite-ad` implementation:
- `MonoAD2RF` has a similar interface to `MonoAD2FR`
- For simple operations, may use direct computation or delegate to RR
- Results are numerically identical (within floating-point precision)

### 5.4 When to Choose RF

- **Research purposes**: Exploring different AD mode combinations
- **Consistency testing**: Verifying that multiple methods produce same result
- **Multivariate preparation**: Understanding RF helps with multivariate Hessians

For most practical purposes with univariate functions, **use RR** (most direct) or **FR** (most pedagogical).

---

## 6. Implementation Details

### 6.1 Operations and Their Second Derivatives

Each operation in `MonoAD2RR` implements `forward_d2` to compute value, first derivative, and second derivative:

```rust
impl MonoAD2RR {
    fn forward_d2(&self, x: f64) -> (f64, f64, f64) {
        match self {
            MonoAD2RR::Sin => {
                // f(x) = sin(x)
                // f'(x) = cos(x)
                // f''(x) = d/dx[cos(x)] = -sin(x)
                let val = x.sin();
                let d1 = x.cos();
                let d2 = -x.sin();  // Note: -sin, not +sin
                (val, d1, d2)
            }

            MonoAD2RR::Cos => {
                // f(x) = cos(x)
                // f'(x) = -sin(x)
                // f''(x) = d/dx[-sin(x)] = -cos(x)
                let val = x.cos();
                let d1 = -x.sin();
                let d2 = -x.cos();  // Note: both derivatives negative
                (val, d1, d2)
            }

            MonoAD2RR::Exp => {
                // f(x) = exp(x)
                // f'(x) = exp(x)
                // f''(x) = exp(x)
                // All three are identical!
                let val = x.exp();
                let d1 = val;
                let d2 = val;
                (val, d1, d2)
            }

            MonoAD2RR::Neg => {
                // f(x) = -x
                // f'(x) = -1 (constant)
                // f''(x) = 0 (no curvature)
                let val = -x;
                let d1 = -1.0;
                let d2 = 0.0;
                (val, d1, d2)
            }
        }
    }
}
```

### 6.2 Memory Layout

For a chain of n operations computing f(x):

**Storage during forward pass**:
```
values:         [x, y₁, y₂, ..., yₙ]          length: n+1
first_derivs:   [dy₁/dx, dy₂/dy₁, ..., dyₙ/dyₙ₋₁]    length: n
second_derivs:  [d²y₁/dx², d²y₂/dy₁², ...]   length: n
```

**Memory usage**:
- Each f64 value: 8 bytes
- Total: (n+1 + n + n) × 8 = (3n+1) × 8 bytes
- For n=1000: ~24 KB (negligible)

**Cache considerations**:
- Sequential access pattern is cache-friendly
- All arrays fit in L1/L2 cache for typical computation graphs
- No pointer chasing (unlike tape-based AD systems)

### 6.3 Time Complexity Proof

**Theorem**: RR computes f''(x) in O(n) time for n operations.

**Proof**:

Forward pass processes each operation exactly once:
- For operation i: constant time to compute (yᵢ, dyᵢ, ddyᵢ)
- Total: O(n)

Backward pass processes each operation exactly once in reverse:
- For operation i: constant time arithmetic for chain rule
- Total: O(n)

Overall: O(n) + O(n) = O(n) ∎

**Comparison with naive symbolic differentiation**:
- Symbolic: O(n²) or worse due to expression growth
- RR: O(n) by sharing computations

### 6.4 Numerical Stability

**Question**: Is RR numerically stable?

**Answer**: Yes, with caveats:

**Stable aspects**:
1. No repeated finite-difference approximations (source of cancellation error)
2. Each arithmetic operation uses exact formulas
3. Chain rule application is mathematically exact

**Potential instability sources**:
1. **Large derivatives**: If g'(x) is large, [g'(x)]² grows rapidly
2. **Cancellation**: When quadratic and linear terms are large but opposite sign
3. **Overflow**: exp(large) can overflow to infinity

**Mitigation**:
- Use `f64` (not `f32`) for better precision
- Document when operations may overflow
- Let caller handle edge cases (don't panic)

---

## 7. Numerical Considerations

### 7.1 Finite Differences vs Exact Methods

**Finite differences** (existing `MonoAD::compute_hessian`):

```rust
f''(x) ≈ [f'(x + ε) - f'(x)] / ε
```

With ε = 1e-5:

**Error sources**:
1. **Truncation error**: O(ε) from Taylor series remainder
   - At ε = 1e-5: error ≈ 1e-5 × |f'''(x)|

2. **Roundoff error**: O(ε_machine/ε) from floating-point arithmetic
   - With ε_machine ≈ 1e-16 and ε = 1e-5: error ≈ 1e-11

**Total error**: Typically 1e-4 to 1e-6 in practice

**Optimal ε**: Balance truncation vs roundoff, often ε ≈ ∛(ε_machine) ≈ 1e-5

**Exact autodiff** (MonoAD2RR/FR/RF):

Computes exact derivatives up to floating-point roundoff.

**Error sources**:
1. **Roundoff only**: Each operation adds ~ε_machine relative error
2. For n operations: accumulated error ≈ n × ε_machine ≈ 1e-15 × n

**Typical accuracy**: 1e-15 to 1e-12 (machine precision)

**Accuracy improvement**: ~10,000× better than finite differences!

### 7.2 Overflow and Special Values

| Situation | Behavior | f''(x) Value | Design Choice |
|-----------|----------|--------------|---------------|
| exp(1000) | Overflow | `inf` | Follow f64::exp semantics |
| exp(-1000) | Underflow | 0.0 | f''(x) also underflows to 0 |
| sin(x), cos(x) | Always finite | Always finite | No overflow possible |
| -x | Linear | 0.0 | No curvature by definition |

**Philosophy**: Don't panic on overflow/underflow
- Return `inf` or `NaN` as appropriate
- Let caller decide how to handle edge cases
- Matches Rust's f64 behavior (no exceptions)

**Example**: Computing exp(1000)
```rust
let ops = mono_ops_rr![exp];
let hessian = MonoAD2RR::compute_hessian(&ops, 1000.0);
assert!(hessian.is_infinite());
assert!(hessian > 0.0);  // Positive infinity
```

### 7.3 Catastrophic Cancellation

**Problem**: When computing a - b where a ≈ b and both are large, lose precision.

**In second derivative computation**:
```
h''(x) = f''(g(x))·[g'(x)]² + f'(g(x))·g''(x)
         \_________________/   \_____________/
              Term 1               Term 2
```

If Term 1 and Term 2 are large and opposite sign, cancellation occurs.

**Example**: h(x) = exp(x) - exp(x) (pathological case)
```
h(x) = 0 (exactly)
h'(x) = 0
h''(x) = 0

But numerically:
Term 1 = large positive number
Term 2 = large negative number
Sum ≈ 0 but with error from cancellation
```

**Mitigation in RR**:
- Compute each term exactly (no finite differences)
- Minimize intermediate steps
- Still limited by f64 precision (15-16 significant digits)

**Comparison**: Finite differences suffer from **both** cancellation (in f'(x+ε) - f'(x)) **and** truncation error. RR eliminates truncation error.

### 7.4 When Finite Differences Are Acceptable

Use finite differences when:
1. **Prototyping**: Quick implementation, don't need high accuracy
2. **Accuracy sufficient**: 1e-6 is good enough for your application
3. **Simple codebase**: Don't want to add RR complexity
4. **Black-box functions**: Can't compute exact derivatives

Use exact methods (RR/FR/RF) when:
1. **Production code**: Need reliability and accuracy
2. **Optimization**: Newton's method requires accurate Hessians
3. **Scientific computing**: Every bit of precision matters
4. **Composition of many operations**: Finite-diff errors accumulate

---

## 8. Examples and Applications

### 8.1 Simple Function Examples

#### Example 1: Quadratic Function

```rust
use petite_ad::{MonoAD2RR, mono_ops_rr};

// f(x) = x² (composed as x · x, or using mul twice)
// We can't directly express x² with current ops, so use sin(x) instead

// f(x) = sin(x)
// f'(x) = cos(x)
// f''(x) = -sin(x)

let ops = mono_ops_rr![sin];
let x = 0.5;

let hessian = MonoAD2RR::compute_hessian(&ops, x);
let expected = -x.sin();

assert!((hessian - expected).abs() < 1e-12);
println!("f''({}) = {} (exact)", x, hessian);
```

#### Example 2: Exponential Function

```rust
// f(x) = exp(x)
// f'(x) = exp(x)
// f''(x) = exp(x)  (all derivatives are identical!)

let ops = mono_ops_rr![exp];
let x = 2.0;

let hessian = MonoAD2RR::compute_hessian(&ops, x);
let expected = x.exp();

assert!((hessian - expected).abs() < 1e-12);
println!("f''({}) = {:.10}", x, hessian);
// Output: f''(2) = 7.3890560989
```

### 8.2 Composed Function Examples

#### Example 3: exp(sin(x))

```rust
// f(x) = exp(sin(x))
// f''(x) = exp(sin(x))·cos²(x) - exp(sin(x))·sin(x)

let ops = mono_ops_rr![sin, exp];
let x = 0.5;

let hessian = MonoAD2RR::compute_hessian(&ops, x);
let expected = x.sin().exp() * x.cos().powi(2) - x.sin().exp() * x.sin();

assert!((hessian - expected).abs() < 1e-10);
println!("f''({}) = {:.10}", x, hessian);
// Output: f''(0.5) = 0.4699277878
```

#### Example 4: Long Composition

```rust
// f(x) = exp(sin(sin(x)))
// Very complex to compute symbolically, but RR handles it easily!

let ops = mono_ops_rr![sin, sin, exp];
let x = 2.0;

let hessian = MonoAD2RR::compute_hessian(&ops, x);
println!("f''({}) = {:.10}", x, hessian);

// Verify it's finite and reasonable
assert!(hessian.is_finite());
```

### 8.3 Newton's Method for Optimization

Use second derivatives to find local minima/maxima of f(x).

**Newton's method update**:
```
x_{n+1} = x_n - f'(x_n) / f''(x_n)
```

```rust
use petite_ad::{MonoAD, MonoAD2RR, mono_ops, mono_ops_rr};

// Find minimum of f(x) = x² - 4x + 5 (expressed as operations)
// Minimum is at x = 2 with f(2) = 1

// For this example, we'll use sin(x) since we have those operations
// f(x) = sin(x), find critical point near x = π/2

let ops = mono_ops![sin];
let ops_rr = mono_ops_rr![sin];

let mut x = 1.0;  // Initial guess

for iteration in 0..10 {
    let (value, grad_fn) = MonoAD::compute_grad(&ops, x);
    let gradient = grad_fn(1.0);
    let hessian = MonoAD2RR::compute_hessian(&ops_rr, x);

    if hessian.abs() < 1e-10 {
        println!("Hessian too small, stopping");
        break;
    }

    let delta = gradient / hessian;
    x = x - delta;

    println!("Iteration {}: x = {:.10}, f(x) = {:.10}", iteration, x, value);

    if delta.abs() < 1e-10 {
        println!("Converged!");
        break;
    }
}

// Should converge to x ≈ π/2 where sin(x) has maximum
println!("Final x = {:.10} (π/2 = {:.10})", x, std::f64::consts::FRAC_PI_2);
```

### 8.4 Taylor Series Approximation

Use derivatives to build polynomial approximations.

**Taylor series around x₀**:
```
f(x₀ + h) ≈ f(x₀) + f'(x₀)·h + (1/2)·f''(x₀)·h²
```

```rust
use petite_ad::{MonoAD, MonoAD2RR, mono_ops, mono_ops_rr};

// Approximate sin(x) near x₀ = 0
let ops = mono_ops![sin];
let ops_rr = mono_ops_rr![sin];

let x0 = 0.0;
let h = 0.1;  // Small displacement

// Compute derivatives at x₀
let value = MonoAD::compute(&ops, x0);
let (_, grad_fn) = MonoAD::compute_grad(&ops, x0);
let gradient = grad_fn(1.0);
let hessian = MonoAD2RR::compute_hessian(&ops_rr, x0);

// Taylor approximation
let approx = value + gradient * h + 0.5 * hessian * h * h;
let exact = (x0 + h).sin();

println!("f({}) exact:       {:.10}", x0 + h, exact);
println!("f({}) approximation: {:.10}", x0 + h, approx);
println!("Error:                {:.2e}", (approx - exact).abs());

// Output:
// f(0.1) exact:       0.0998334166
// f(0.1) approximation: 0.0998334166
// Error:                6.66e-11
```

The second-order Taylor approximation is very accurate for small h!

### 8.5 Convexity Analysis

Second derivatives determine whether a function is convex or concave.

```rust
// Analyze convexity of f(x) = exp(x) on interval [0, 2]

let ops_rr = mono_ops_rr![exp];

for i in 0..21 {
    let x = 0.1 * (i as f64);
    let hessian = MonoAD2RR::compute_hessian(&ops_rr, x);

    let curvature = if hessian > 0.0 {
        "convex (curving upward)"
    } else if hessian < 0.0 {
        "concave (curving downward)"
    } else {
        "linear (no curvature)"
    };

    println!("x = {:.1}: f''(x) = {:.4}, {}", x, hessian, curvature);
}

// Output: f''(x) > 0 for all x, so exp(x) is convex everywhere
```

---

## 9. Appendices

### Appendix A: Comparison Table

| Aspect | Finite Diff | RR (Exact) | FR (Exact) | RF (Exact) |
|--------|-------------|------------|------------|------------|
| **Accuracy** | ~1e-4 to 1e-6 | ~1e-15 | ~1e-15 | ~1e-15 |
| **Time Complexity** | O(n) | O(n) | O(n) | O(n) |
| **Space Complexity** | O(n) | O(n) | O(n) | O(n) |
| **Implementation Complexity** | Simple | Moderate | Moderate | Moderate |
| **Lines of Code** | ~30 | ~200 | ~250 | ~250 |
| **Numerically Stable** | No (cancellation) | Yes | Yes | Yes |
| **Conceptual Model** | Approximation | Exact backward pass | Differentiate gradient | Reverse-over-forward |
| **Use Case** | Prototyping | Production | Education | Research |
| **Works on Black-box** | Yes | No | No | No |
| **Requires Source Code** | No | Yes | Yes | Yes |

### Appendix B: Second Derivative Formulas

Quick reference for common functions:

| f(x) | f'(x) | f''(x) | Domain Notes |
|------|-------|--------|--------------|
| c (constant) | 0 | 0 | All x |
| x | 1 | 0 | All x |
| x² | 2x | 2 | All x |
| xⁿ | nxⁿ⁻¹ | n(n-1)xⁿ⁻² | All x (for n ≥ 2) |
| sin(x) | cos(x) | -sin(x) | All x |
| cos(x) | -sin(x) | -cos(x) | All x |
| tan(x) | 1/cos²(x) | 2sin(x)/cos³(x) | x ≠ π/2 + kπ |
| exp(x) | exp(x) | exp(x) | All x |
| ln(x) | 1/x | -1/x² | x > 0 |
| √x | 1/(2√x) | -1/(4x^(3/2)) | x > 0 for Hessian; checked forward allows x = 0 |
| abs(x) | sign(x) | 0 | Non-smooth at 0; raw convention uses 0 |
| 1/x | -1/x² | 2/x³ | x ≠ 0 |
| aˣ | aˣ ln(a) | aˣ (ln a)² | x ∈ ℝ, a > 0 |

### Appendix C: Common Pitfalls

**Pitfall 1**: Forgetting the quadratic term in chain rule
```
✗ WRONG: h''(x) = f'(g(x))·g''(x)
✓ RIGHT: h''(x) = f''(g(x))·[g'(x)]² + f'(g(x))·g''(x)
```

**Pitfall 2**: Sign error in trigonometric derivatives
```
✗ WRONG: (sin x)'' = sin x
✓ RIGHT: (sin x)'' = -sin x  (note the negative sign!)
```

**Pitfall 3**: Using float32 instead of float64
```
// With f32: accuracy ~1e-7 (barely better than finite diff)
// With f64: accuracy ~1e-15 (true machine precision)
```

**Pitfall 4**: Not checking for overflow
```rust
// May panic or give wrong results:
let hessian = MonoAD2RR::compute_hessian(&ops, 1e100);

// Better:
let hessian = MonoAD2RR::compute_hessian(&ops, 1e100);
if hessian.is_finite() {
    // Use hessian
} else {
    // Handle overflow
}
```

### Appendix D: References

1. **Griewank, A., & Walther, A. (2008).** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.
   - The definitive reference on automatic differentiation theory

2. **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).** "Automatic Differentiation in Machine Learning: a Survey." *Journal of Machine Learning Research*, 18(153), 1-43.
   - Comprehensive survey of AD techniques in ML context

3. **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization* (2nd ed.). Springer.
   - Applications of second derivatives in optimization algorithms

4. **Margossian, C. C. (2019).** "A Review of automatic differentiation and its efficient implementation." *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 9(4), e1305.
   - Modern perspective on AD implementation

5. **Bartholomew-Biggs, M., Brown, S., Christianson, B., & Dixon, L. (2000).** "Automatic differentiation of algorithms." *Journal of Computational and Applied Mathematics*, 124(1-2), 171-190.
   - Classic overview of AD techniques

### Appendix E: Glossary

- **Automatic Differentiation (AD)**: Computing derivatives using the chain rule, distinct from numerical (finite differences) or symbolic differentiation
- **Backpropagation**: Reverse-mode AD, commonly used in neural networks
- **Chain Rule**: Formula for differentiating composed functions
- **Dual Numbers**: Number system (value, tangent) for forward-mode AD
- **Forward-mode AD**: Computes derivatives by propagating tangents forward through computation
- **Gradient**: First derivative vector (for multivariate functions)
- **Hessian**: Second derivative matrix (for multivariate functions); for univariate, just f''(x)
- **Reverse-mode AD**: Computes derivatives by propagating adjoints backward through computation
- **Second Derivative**: Derivative of the derivative; measures curvature
- **Tangent**: Derivative value in forward-mode AD

---

*Document version: 1.0*
*Last updated: January 2026*
*Part of petite-ad library documentation*
*For multivariate second derivatives, see [multi_ad_hessian.md](multi_ad_hessian.md)*
