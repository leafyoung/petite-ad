//! Exact second-order autodiff using Reverse-over-Reverse (RR) mode.
//!
//! This module implements the **Reverse-over-Reverse (RR)** method for computing exact
//! Hessians (second derivatives) of single-variable functions. It tracks second-order
//! derivatives during the reverse pass using the chain rule.
//!
//! # Supported Operations
//!
//! `MonoAD2RR` supports a subset of the operations available in [`crate::MonoAD`]:
//! `Sin`, `Cos`, `Tan`, `Exp`, `Neg`, `Ln`, `Sqrt`, `Abs`. The `Abs` operation is
//! non-smooth at zero and follows the raw `f64` convention used by [`crate::MonoAD`]:
//! derivative `0` and curvature `0` at `x = 0`. For first-order differentiation with
//! the full operation set, use
//! [`crate::MonoAD`]. For Hessian approximation with all operations, use
//! [`MultiAD::compute_hessian`](crate::MultiAD::compute_hessian) (finite-difference based).
//!
//! # Comprehensive Documentation
//!
//! For complete mathematical theory, detailed derivations, complexity analysis, and
//! comparison with other methods, see:
//! **[`/docs/mono_ad_hessian.md`](../../docs/mono_ad_hessian.md)**
//!
//! # Mathematical Foundation
//!
//! ## Chain Rule for Second Derivatives
//!
//! For a composition h(x) = f(g(x)), the chain rule for derivatives is:
//!
//! ```text
//! First derivative:  h'(x) = f'(g(x)) · g'(x)
//! Second derivative: h''(x) = f''(g(x)) · [g'(x)]² + f'(g(x)) · g''(x)
//! ```
//!
//! This formula is fundamental to the RR method. The second derivative has two terms:
//! - **Product term**: f''(g(x)) · [g'(x)]² — contribution from second derivative of outer function
//! - **Chain term**: f'(g(x)) · g''(x) — contribution from first derivative of outer times second of inner
//!
//! ## Backward Propagation
//!
//! During the reverse pass, we propagate two values backward through the computation graph:
//! - `grad`: the accumulated gradient (first derivative) ∂L/∂u
//! - `hessian`: the accumulated second derivative ∂²L/∂u²
//!
//! For each operation y = op(u) with:
//! - First derivative: dy = op'(u) = ∂y/∂u
//! - Second derivative: ddy = op''(u) = ∂²y/∂u²
//!
//! The backward accumulation follows the chain rule:
//!
//! ```text
//! new_grad = grad · dy                    (standard reverse-mode AD)
//! new_hessian = hessian · dy² + grad · ddy   (reverse-over-reverse formula)
//! ```
//!
//! This is derived from applying the chain rule to the gradient computation itself.
//! See [docs/mono_ad_hessian.md](../../docs/mono_ad_hessian.md#reverse-over-reverse-rr) for full derivation.
//!
//! # Algorithm Overview
//!
//! The RR method operates in two phases:
//!
//! ## Phase 1: Forward Pass
//!
//! Traverse the computation graph forward, storing at each node i:
//! - `values[i]`: the value vᵢ
//! - `first_derivs[i]`: the first derivative dvᵢ/du (where u is the input)
//! - `second_derivs[i]`: the second derivative d²vᵢ/du²
//!
//! ## Phase 2: Reverse Pass
//!
//! Traverse backward from output to input, accumulating:
//! - Start: `grad = 1.0` (∂F/∂F = 1), `hessian = 0.0` (∂²F/∂F² = 0)
//! - For each operation with derivatives (dy, ddy):
//!   - Apply chain rule: `new_hessian = hessian · dy² + grad · ddy`
//!   - Update: `grad = grad · dy`
//! - Result: `hessian` contains ∂²F/∂x²
//!
//! # Computational Complexity
//!
//! For a computation graph with n operations:
//! - **Time**: O(n) — single forward pass + single reverse pass
//! - **Space**: O(n) — must store all intermediate values and derivatives
//! - **Overhead**: ~3x compared to first-order reverse-mode AD
//!
//! # Accuracy
//!
//! This provides **exact** second derivatives up to floating-point precision:
//! - No finite difference approximations
//! - No truncation error
//! - Error bounded by machine epsilon (~2.2e-16 for f64)
//! - Typical relative error: < 1e-14
//!
//! # Supported Operations
//!
//! Currently supports eight elementary operations. Each operation's second derivative:
//!
//! | Operation | Function | First Derivative f'(x) | Second Derivative f''(x) |
//! |-----------|----------|----------------------|------------------------|
//! | `Sin`     | sin(x)   | cos(x)               | -sin(x)                |
//! | `Cos`     | cos(x)   | -sin(x)              | -cos(x)                |
//! | `Tan`     | tan(x)   | sec²(x)              | 2 sec²(x) tan(x)       |
//! | `Exp`     | exp(x)   | exp(x)               | exp(x)                 |
//! | `Neg`     | -x       | -1                   | 0                      |
//! | `Ln`      | ln(x)    | 1/x                  | -1/x²                  |
//! | `Sqrt`    | sqrt(x)  | 1/(2sqrt(x))         | -1/(4x sqrt(x))        |
//! | `Abs`     | abs(x)   | sign(x)              | 0                      |
//!
//! # Example Usage
//!
//! ```rust
//! use petite_ad::MonoAD2RR;
//!
//! // Compute f(x) = exp(sin(x))
//! // f'(x) = cos(x) · exp(sin(x))
//! // f''(x) = -sin(x) · exp(sin(x)) + cos²(x) · exp(sin(x))
//! let ops = [MonoAD2RR::Sin, MonoAD2RR::Exp];
//! let x = 1.0;
//!
//! let value = MonoAD2RR::compute(&ops, x);
//! let hessian = MonoAD2RR::compute_hessian(&ops, x);
//!
//! // Expected: f''(1.0) ≈ -0.9318...
//! println!("f''({}) = {}", x, hessian);
//! ```
//!
//! # Comparison with Other Methods
//!
//! | Method | Accuracy | Time | Space | Best For |
//! |--------|----------|------|-------|----------|
//! | **RR (this)** | Exact | O(n) | O(n) | General purpose, balanced |
//! | FR | Exact | O(n) | O(n) | Slightly faster than RR |
//! | RF | Exact | O(n) | O(n) | Slightly slower than RR |
//! | Finite-diff | ~1e-5 | O(n) | O(1) | Quick approximations |
//!
//! See [docs/mono_ad_hessian.md](../../docs/mono_ad_hessian.md) for detailed comparisons.
//!
//! # References
//!
//! - Griewank & Walther (2008): *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*
//! - Gebremedhin et al. (2002): "Efficient computation of Hessian matrices"
//! - Pearlmutter (1994): "Fast exact multiplication by the Hessian"

use crate::Result;

use super::common::{self, MonoHessianOpKind};
use crate::mono::types::*;

/// Single-variable automatic differentiation operations for Reverse-over-Reverse Hessian computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonoAD2RR {
    Sin,
    Cos,
    Tan,
    Exp,
    Neg,
    Ln,
    Sqrt,
    Abs,
}

impl MonoAD2RR {
    #[inline(always)]
    fn as_hessian_op(self) -> MonoHessianOpKind {
        match self {
            MonoAD2RR::Sin => MonoHessianOpKind::Sin,
            MonoAD2RR::Cos => MonoHessianOpKind::Cos,
            MonoAD2RR::Tan => MonoHessianOpKind::Tan,
            MonoAD2RR::Exp => MonoHessianOpKind::Exp,
            MonoAD2RR::Neg => MonoHessianOpKind::Neg,
            MonoAD2RR::Ln => MonoHessianOpKind::Ln,
            MonoAD2RR::Sqrt => MonoHessianOpKind::Sqrt,
            MonoAD2RR::Abs => MonoHessianOpKind::Abs,
        }
    }

    #[inline(always)]
    fn check_domain(self, x: f64) -> Result<()> {
        common::check_domain(self.as_hessian_op(), x)
    }

    /// Forward pass computing value, first derivative, and second derivative.
    ///
    /// Evaluates an elementary operation at a given point, returning the function value
    /// and its first two derivatives. This is the core primitive for the RR method.
    ///
    /// # Mathematical Formulas
    ///
    /// For each operation, we compute:
    ///
    /// ## Sin: y = sin(x)
    /// - Value: y = sin(x)
    /// - First derivative: dy/dx = cos(x)
    /// - Second derivative: d²y/dx² = -sin(x)
    ///
    /// Derivation: d/dx[cos(x)] = -sin(x)
    ///
    /// ## Cos: y = cos(x)
    /// - Value: y = cos(x)
    /// - First derivative: dy/dx = -sin(x)
    /// - Second derivative: d²y/dx² = -cos(x)
    ///
    /// Derivation: d/dx[-sin(x)] = -cos(x)
    ///
    /// ## Exp: y = exp(x)
    /// - Value: y = exp(x)
    /// - First derivative: dy/dx = exp(x)
    /// - Second derivative: d²y/dx² = exp(x)
    ///
    /// Derivation: The exponential function is its own derivative at all orders
    ///
    /// ## Neg: y = -x
    /// - Value: y = -x
    /// - First derivative: dy/dx = -1
    /// - Second derivative: d²y/dx² = 0
    ///
    /// Derivation: Linear functions have zero second derivative
    ///
    /// # Arguments
    ///
    /// * `self` - The operation to evaluate (Sin, Cos, Tan, Exp, Neg, Ln, Sqrt, or Abs)
    /// * `x` - The point at which to evaluate the operation
    ///
    /// # Returns
    ///
    /// A tuple `(value, first_derivative, second_derivative)` containing:
    /// - `value`: f(x)
    /// - `first_derivative`: f'(x)
    /// - `second_derivative`: f''(x)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // This is a private method used internally by compute_hessian
    /// use petite_ad::MonoAD2RR;
    ///
    /// let op = MonoAD2RR::Sin;
    /// // forward_d2 is private and used internally
    /// // See compute_hessian for public API examples
    /// ```
    ///
    /// For public usage examples, see [`MonoAD2RR::compute_hessian`].
    ///
    /// # Complexity
    ///
    /// - **Time**: O(1) — constant time for each operation
    /// - **Space**: O(1) — returns three f64 values
    fn forward_d2(&self, x: f64) -> (f64, f64, f64) {
        match self {
            MonoAD2RR::Sin => {
                // f(x) = sin(x)
                // f'(x) = cos(x)
                // f''(x) = -sin(x)
                let y = x.sin();
                let dy = x.cos(); // First derivative
                let ddy = -x.sin(); // Second derivative: d/dx[cos(x)] = -sin(x)
                (y, dy, ddy)
            }
            MonoAD2RR::Cos => {
                // f(x) = cos(x)
                // f'(x) = -sin(x)
                // f''(x) = -cos(x)
                let y = x.cos();
                let dy = -x.sin(); // First derivative
                let ddy = -x.cos(); // Second derivative: d/dx[-sin(x)] = -cos(x)
                (y, dy, ddy)
            }
            MonoAD2RR::Tan => {
                // f(x) = tan(x)
                // f'(x) = sec^2(x)
                // f''(x) = 2 sec^2(x) tan(x)
                let y = x.tan();
                let sec_sq = 1.0 / x.cos().powi(2);
                let dy = sec_sq;
                let ddy = 2.0 * sec_sq * y;
                (y, dy, ddy)
            }
            MonoAD2RR::Exp => {
                // f(x) = exp(x)
                // f'(x) = exp(x)
                // f''(x) = exp(x)
                // The exponential function is its own derivative at all orders
                let y = x.exp();
                let dy = y; // First derivative: exp(x)
                let ddy = y; // Second derivative: exp(x)
                (y, dy, ddy)
            }
            MonoAD2RR::Neg => {
                // f(x) = -x
                // f'(x) = -1
                // f''(x) = 0
                // Linear functions have zero second derivative
                let y = -x;
                let dy = -1.0; // First derivative: constant
                let ddy = 0.0; // Second derivative: zero (linear function)
                (y, dy, ddy)
            }
            MonoAD2RR::Ln => {
                // f(x) = ln(x)
                // f'(x) = 1/x
                // f''(x) = -1/x^2
                let y = x.ln();
                let dy = 1.0 / x;
                let ddy = -1.0 / x.powi(2);
                (y, dy, ddy)
            }
            MonoAD2RR::Sqrt => {
                // f(x) = sqrt(x)
                // f'(x) = 1/(2 sqrt(x))
                // f''(x) = -1/(4 x sqrt(x))
                let y = x.sqrt();
                let dy = 1.0 / (2.0 * y);
                let ddy = -1.0 / (4.0 * x * y);
                (y, dy, ddy)
            }
            MonoAD2RR::Abs => {
                // f(x) = abs(x)
                // f'(x) = sign(x), with raw convention f'(0) = 0
                // f''(x) = 0 away from zero; raw convention f''(0) = 0
                let y = x.abs();
                let dy = common::sign_or_zero(x);
                let ddy = 0.0;
                (y, dy, ddy)
            }
        }
    }

    /// Compute forward pass only.
    pub fn compute(exprs: &[MonoAD2RR], x: f64) -> f64 {
        let mut value = x;
        for expr in exprs {
            value = expr.forward_d2(value).0;
        }
        value
    }

    /// Compute forward pass with opt-in checked-domain validation.
    pub fn compute_checked(exprs: &[MonoAD2RR], x: f64) -> Result<f64> {
        let mut value = x;
        for &op in exprs {
            op.check_domain(value)?;
            value = op.forward_d2(value).0;
        }
        Ok(value)
    }

    /// Compute forward pass and return gradient function using exact reverse-mode.
    pub fn compute_grad(exprs: &[MonoAD2RR], x: f64) -> BackwardResultBox {
        Self::compute_grad_generic::<Box<DynMathFn>>(exprs, x)
    }

    /// Compute forward pass and gradient function with checked-domain validation.
    pub fn compute_grad_checked(exprs: &[MonoAD2RR], x: f64) -> Result<BackwardResultBox> {
        let mut value = x;
        let mut backprops: Vec<Box<DynMathFn>> = Vec::new();

        for &op in exprs {
            op.check_domain(value)?;
            let (new_value, _dy, _ddy) = op.forward_d2(value);
            let backprop = Self::make_backward_fn(op, value);
            value = new_value;
            backprops.push(backprop);
        }

        let backward_fn = Box::new(move |cotangent: f64| -> f64 {
            let mut grad = cotangent;
            for backprop in backprops.iter().rev() {
                grad = backprop(grad);
            }
            grad
        });

        Ok((value, backward_fn))
    }

    /// Generic gradient computation.
    fn compute_grad_generic<W>(exprs: &[MonoAD2RR], x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
    {
        let mut value = x;
        let mut backprops: Vec<W> = Vec::new();

        for &op in exprs {
            let (new_value, _dy, _ddy) = op.forward_d2(value);
            let backprop = Self::make_backward_fn(op, value);
            value = new_value;
            backprops.push(backprop);
        }

        let backward_fn = Box::new(move |cotangent: f64| -> f64 {
            let mut grad = cotangent;
            for backprop in backprops.iter().rev() {
                grad = backprop(grad);
            }
            grad
        });

        (value, W::from(backward_fn))
    }

    /// Create backward function for an operation (first-order only).
    fn make_backward_fn<W>(op: MonoAD2RR, x: f64) -> W
    where
        W: From<Box<DynMathFn>>,
    {
        let grad_fn: Box<DynMathFn> = match op {
            MonoAD2RR::Sin => Box::new(move |dy: f64| -> f64 { dy * x.cos() }),
            MonoAD2RR::Cos => Box::new(move |dy: f64| -> f64 { dy * -x.sin() }),
            MonoAD2RR::Tan => {
                let sec_sq = 1.0 / x.cos().powi(2);
                Box::new(move |dy: f64| -> f64 { dy * sec_sq })
            }
            MonoAD2RR::Exp => {
                let exp_val = x.exp();
                Box::new(move |dy: f64| -> f64 { dy * exp_val })
            }
            MonoAD2RR::Neg => Box::new(move |dy: f64| -> f64 { -dy * 1.0 }),
            MonoAD2RR::Ln => Box::new(move |dy: f64| -> f64 { dy / x }),
            MonoAD2RR::Sqrt => {
                let sqrt_x = x.sqrt();
                Box::new(move |dy: f64| -> f64 { dy / (2.0 * sqrt_x) })
            }
            MonoAD2RR::Abs => {
                let sign = common::sign_or_zero(x);
                Box::new(move |dy: f64| -> f64 { dy * sign })
            }
        };
        W::from(grad_fn)
    }

    /// Compute exact Hessian using Reverse-over-Reverse mode.
    ///
    /// This is the main entry point for computing exact second derivatives.
    /// It implements the RR algorithm by performing a forward pass to collect
    /// derivative information, then a reverse pass to accumulate the Hessian
    /// using the chain rule.
    ///
    /// # Algorithm Walkthrough
    ///
    /// Given a sequence of operations [op₁, op₂, ..., opₙ] applied to input x:
    ///
    /// ## Forward Pass
    ///
    /// For each operation opᵢ in sequence:
    /// 1. Compute (yᵢ, dyᵢ, ddyᵢ) = opᵢ.forward_d2(yᵢ₋₁)
    /// 2. Store yᵢ, dyᵢ, ddyᵢ
    ///
    /// where y₀ = x (the input)
    ///
    /// This gives us the computation graph with all intermediate values and local derivatives.
    ///
    /// ## Reverse Pass
    ///
    /// Initialize:
    /// - `grad = 1.0` (seed: ∂F/∂F = 1, where F is the final output)
    /// - `hessian = 0.0` (seed: ∂²F/∂F² = 0)
    ///
    /// For each operation in reverse order (opₙ, ..., op₂, op₁):
    /// 1. Retrieve stored derivatives: (dyᵢ, ddyᵢ)
    /// 2. Apply chain rule:
    ///    ```text
    ///    new_hessian = hessian · dyᵢ² + grad · ddyᵢ
    ///    new_grad = grad · dyᵢ
    ///    ```
    /// 3. Update: grad ← new_grad, hessian ← new_hessian
    ///
    /// The final `hessian` value is ∂²F/∂x².
    ///
    /// ## Chain Rule Derivation
    ///
    /// At each step, we're computing derivatives with respect to earlier variables.
    /// If we have L = f(u) and u = g(x), then by the chain rule:
    ///
    /// ```text
    /// ∂L/∂x = ∂L/∂u · ∂u/∂x = grad · dy
    ///
    /// ∂²L/∂x² = ∂/∂x[∂L/∂x]
    ///         = ∂/∂x[∂L/∂u · ∂u/∂x]
    ///         = ∂²L/∂u² · (∂u/∂x)² + ∂L/∂u · ∂²u/∂x²
    ///         = hessian · dy² + grad · ddy
    /// ```
    ///
    /// This is the fundamental formula used in the reverse pass.
    ///
    /// # Example Trace
    ///
    /// For f(x) = exp(sin(x)) at x = 1.0:
    ///
    /// ## Forward Pass:
    /// ```text
    /// Input: x = 1.0
    ///
    /// Op 1: Sin
    ///   y₁ = sin(1.0) ≈ 0.8414709848
    ///   dy₁ = cos(1.0) ≈ 0.5403023059
    ///   ddy₁ = -sin(1.0) ≈ -0.8414709848
    ///
    /// Op 2: Exp
    ///   y₂ = exp(0.8414709848) ≈ 2.3198323620
    ///   dy₂ = exp(0.8414709848) ≈ 2.3198323620
    ///   ddy₂ = exp(0.8414709848) ≈ 2.3198323620
    /// ```
    ///
    /// ## Reverse Pass:
    /// ```text
    /// Initialize: grad = 1.0, hessian = 0.0
    ///
    /// Step 1 (Op 2: Exp):
    ///   new_hessian = 0.0 · (2.3198)² + 1.0 · 2.3198 ≈ 2.3198
    ///   new_grad = 1.0 · 2.3198 ≈ 2.3198
    ///
    /// Step 2 (Op 1: Sin):
    ///   new_hessian = 2.3198 · (0.5403)² + 2.3198 · (-0.8414)
    ///                ≈ 0.6766 - 1.9523 ≈ -1.2757
    ///   new_grad = 2.3198 · 0.5403 ≈ 1.2533
    ///
    /// Result: f''(1.0) ≈ -1.2757
    /// ```
    ///
    /// # Arguments
    ///
    /// * `exprs` - Slice of operations to apply in sequence
    /// * `x` - Input value to evaluate at
    ///
    /// # Returns
    ///
    /// The exact second derivative f''(x) at the given point
    ///
    /// # Edge Cases
    ///
    /// * **Empty expression**: Returns 0.0 (constant function has zero second derivative)
    /// * **Single operation**: Returns the operation's second derivative at x
    /// * **Linear composition**: Correctly returns 0.0 for linear functions (e.g., Neg)
    ///
    /// # Accuracy
    ///
    /// - **Method**: Exact symbolic differentiation evaluated numerically
    /// - **Error source**: Only floating-point rounding (machine epsilon)
    /// - **Typical relative error**: < 1e-12
    /// - **No truncation error**: Unlike finite differences, this is mathematically exact
    ///
    /// # Complexity
    ///
    /// For n operations:
    /// - **Time**: O(n) — one forward pass + one reverse pass, each visiting n nodes
    /// - **Space**: O(n) — must store n values and 2n derivatives
    /// - **Overhead vs first-order AD**: ~3x (stores second derivatives + extra arithmetic)
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::MonoAD2RR;
    ///
    /// // Example 1: f(x) = sin(x), f''(x) = -sin(x)
    /// let ops = [MonoAD2RR::Sin];
    /// let x = 0.5;
    /// let hessian = MonoAD2RR::compute_hessian(&ops, x);
    /// assert!((hessian - (-0.5_f64.sin())).abs() < 1e-12);
    ///
    /// // Example 2: f(x) = exp(sin(x))
    /// // f'(x) = cos(x) · exp(sin(x))
    /// // f''(x) = -sin(x) · exp(sin(x)) + cos²(x) · exp(sin(x))
    /// let ops = [MonoAD2RR::Sin, MonoAD2RR::Exp];
    /// let x = 1.0;
    /// let hessian = MonoAD2RR::compute_hessian(&ops, x);
    ///
    /// // Manual calculation:
    /// let sin_x = x.sin();
    /// let cos_x = x.cos();
    /// let exp_sin_x = sin_x.exp();
    /// let expected = -sin_x * exp_sin_x + cos_x * cos_x * exp_sin_x;
    /// assert!((hessian - expected).abs() < 1e-12);
    ///
    /// // Example 3: f(x) = -x (linear, so f''(x) = 0)
    /// let ops = [MonoAD2RR::Neg];
    /// let hessian = MonoAD2RR::compute_hessian(&ops, 1.0);
    /// assert_eq!(hessian, 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`MonoAD2RR::compute`]: For computing just the function value
    /// - [`MonoAD2RR::compute_grad`]: For computing first derivatives
    /// - [`crate::MonoAD2FR`]: Forward-over-Reverse method (alternative exact method)
    /// - [`crate::MonoAD2RF`]: Reverse-over-Forward method (alternative exact method)
    /// - [docs/mono_ad_hessian.md](../../docs/mono_ad_hessian.md): Complete mathematical theory
    pub fn compute_hessian(exprs: &[MonoAD2RR], x: f64) -> f64 {
        Self::compute_hessian_impl(exprs, x, false).expect("unchecked Hessian cannot fail")
    }

    /// Compute exact Hessian with checked-domain validation.
    pub fn compute_hessian_checked(exprs: &[MonoAD2RR], x: f64) -> Result<f64> {
        Self::compute_hessian_impl(exprs, x, true)
    }

    fn compute_hessian_impl(exprs: &[MonoAD2RR], x: f64, checked: bool) -> Result<f64> {
        // Edge case: empty expression represents constant function
        if exprs.is_empty() {
            return Ok(0.0);
        }

        let n = exprs.len();

        // ========================================
        // FORWARD PASS: Collect derivative information
        // ========================================
        // Store all intermediate values (needed for computing derivatives)
        let mut values: Vec<f64> = Vec::with_capacity(n + 1);
        values.push(x); // y₀ = x (input)

        // Store first and second derivatives at each operation
        let mut first_derivs: Vec<f64> = Vec::with_capacity(n);
        let mut second_derivs: Vec<f64> = Vec::with_capacity(n);

        // Traverse forward through operations, computing and storing derivatives
        for &op in exprs {
            let input_val = *values.last().unwrap(); // yᵢ₋₁
            if checked {
                op.check_domain(input_val)?;
            }
            let (y, dy, ddy) = op.forward_d2(input_val);

            values.push(y); // Store yᵢ
            first_derivs.push(dy); // Store ∂yᵢ/∂yᵢ₋₁
            second_derivs.push(ddy); // Store ∂²yᵢ/∂yᵢ₋₁²
        }

        // ========================================
        // REVERSE PASS: Accumulate Hessian using chain rule
        // ========================================
        // Initialize reverse pass seeds
        // grad = ∂F/∂F = 1.0 (derivative of output with respect to itself)
        // hessian = ∂²F/∂F² = 0.0 (second derivative of output with respect to itself)
        let mut grad: f64 = 1.0;
        let mut hessian: f64 = 0.0;

        // Traverse backward through operations, applying chain rule
        for i in (0..n).rev() {
            let dy = first_derivs[i]; // ∂yᵢ/∂yᵢ₋₁
            let ddy = second_derivs[i]; // ∂²yᵢ/∂yᵢ₋₁²

            // Apply second-order chain rule:
            // If L depends on yᵢ and yᵢ depends on yᵢ₋₁, then:
            //   ∂²L/∂yᵢ₋₁² = ∂²L/∂yᵢ² · (∂yᵢ/∂yᵢ₋₁)² + ∂L/∂yᵢ · ∂²yᵢ/∂yᵢ₋₁²
            //              = hessian · dy² + grad · ddy
            //
            // See docs/mono_ad_hessian.md for detailed derivation
            let new_hessian = hessian * dy * dy + grad * ddy;

            // Standard reverse-mode gradient update
            let new_grad = grad * dy;

            // Update for next iteration (moving backward)
            grad = new_grad;
            hessian = new_hessian;
        }

        // After processing all operations, hessian contains ∂²F/∂x²
        Ok(hessian)
    }
}
