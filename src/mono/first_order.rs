use crate::{AutodiffError, Result};

use super::types::*;

/// Single-variable automatic differentiation operations.
///
/// Represents mathematical operations that can be composed and differentiated
/// automatically using reverse-mode differentiation (backpropagation).
///
/// # Examples
///
/// ```
/// use petite_ad::{MonoAD, mono_ops};
///
/// // Compose operations: exp(cos(sin(x)))
/// let ops = mono_ops![sin, cos, exp];
/// let (value, grad_fn) = MonoAD::compute_grad(&ops, 2.0);
///
/// println!("f(2.0) = {}", value);
/// println!("f'(2.0) = {}", grad_fn(1.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonoAD {
    /// Sine function: sin(x)
    ///
    /// Derivative: cos(x)
    ///
    /// # Notes
    /// - Delegates to `f64::sin()`, which operates in radians
    /// - Returns values in the range `[-1.0, 1.0]`
    Sin,
    /// Cosine function: cos(x)
    ///
    /// Derivative: -sin(x)
    ///
    /// # Notes
    /// - Delegates to `f64::cos()`, which operates in radians
    /// - Returns values in the range `[-1.0, 1.0]`
    Cos,
    /// Tangent function: tan(x)
    ///
    /// Derivative: sec²(x) = 1 / cos²(x)
    ///
    /// # Notes
    /// - Delegates to `f64::tan()`, which operates in radians
    /// - Returns very large values near `π/2 + kπ` (asymptotes)
    Tan,
    /// Exponential function: exp(x)
    ///
    /// Derivative: exp(x)
    ///
    /// # Notes
    /// - Delegates to `f64::exp()`
    /// - Returns `inf` for very large inputs (> ~709 for f64)
    /// - Returns `0.0` for very large negative inputs (< ~-745 for f64)
    Exp,
    /// Negation: -x
    Neg,
    /// Natural logarithm: ln(x)
    ///
    /// Derivative: 1 / x
    Ln,
    /// Square root: sqrt(x)
    ///
    /// Derivative: 1 / (2 sqrt(x))
    Sqrt,
    /// Absolute value: abs(x)
    ///
    /// Derivative: sign(x), with subgradient sign(0) = 0
    Abs,
}

impl MonoAD {
    /// Compute the forward pass for a single operation.
    ///
    /// This is an internal helper that computes just the forward value
    /// without building gradient closures.
    #[inline(always)]
    fn forward(&self, x: f64) -> f64 {
        match self {
            MonoAD::Sin => x.sin(),
            MonoAD::Cos => x.cos(),
            MonoAD::Tan => x.tan(),
            MonoAD::Exp => x.exp(),
            MonoAD::Neg => -x,
            MonoAD::Ln => x.ln(),
            MonoAD::Sqrt => x.sqrt(),
            MonoAD::Abs => x.abs(),
        }
    }

    /// Validate real-domain restrictions for checked mono evaluation.
    #[inline(always)]
    fn check_domain(self, x: f64) -> Result<()> {
        match self {
            MonoAD::Ln if x <= 0.0 => Err(AutodiffError::domain("Ln", "input must be positive")),
            MonoAD::Sqrt if x < 0.0 => {
                Err(AutodiffError::domain("Sqrt", "input must be non-negative"))
            }
            _ => Ok(()),
        }
    }

    #[inline(always)]
    fn forward_checked(&self, x: f64) -> Result<f64> {
        self.check_domain(x)?;
        Ok(self.forward(x))
    }

    /// Compute the forward pass only (no gradient computation).
    ///
    /// Evaluates the composed function by applying operations sequentially.
    ///
    /// # Arguments
    ///
    /// * `exprs` - Slice of operations to apply in sequence
    /// * `x` - Input value
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::{MonoAD, mono_ops};
    ///
    /// let ops = mono_ops![sin, exp];
    /// let result = MonoAD::compute(&ops, 2.0);
    /// assert!((result - 2.0_f64.sin().exp()).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn compute(exprs: &[MonoAD], x: f64) -> f64 {
        let mut value = x;
        for expr in exprs {
            value = expr.forward(value);
        }
        value
    }

    /// Compute the forward pass with opt-in real-domain validation.
    ///
    /// Existing unchecked methods preserve raw `f64` behavior. This checked variant
    /// returns a domain error for `Ln` inputs `<= 0` and `Sqrt` inputs `< 0`.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::DomainError`] if an intermediate value violates a
    /// real-domain restriction.
    #[inline]
    pub fn compute_checked(exprs: &[MonoAD], x: f64) -> Result<f64> {
        let mut value = x;
        for expr in exprs {
            value = expr.forward_checked(value)?;
        }
        Ok(value)
    }

    // Helper that works with Box wrapper type
    // Box<dyn Fn> is the common type that all arms return
    #[inline(always)]
    fn backward_generic<W>(self, x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>>,
    {
        let (y, grad_fn): (f64, Box<dyn Fn(f64) -> f64>) = match self {
            MonoAD::Sin => {
                let y = x.sin();
                let x_cos = x.cos();
                let grad = Box::new(move |dy: f64| -> f64 { dy * x_cos });
                (y, grad)
            }
            MonoAD::Cos => {
                let y = x.cos();
                let x_sin = x.sin();
                let grad = Box::new(move |dy: f64| -> f64 { -dy * x_sin });
                (y, grad)
            }
            MonoAD::Tan => {
                let y = x.tan();
                let sec_sq = 1.0 / x.cos().powi(2);
                let grad = Box::new(move |dy: f64| -> f64 { dy * sec_sq });
                (y, grad)
            }
            MonoAD::Exp => {
                let y = x.exp();
                let grad = Box::new(move |dy: f64| -> f64 { dy * y });
                (y, grad)
            }
            MonoAD::Neg => {
                let y = -x;
                let grad = Box::new(move |dy: f64| -> f64 { -dy });
                (y, grad)
            }
            MonoAD::Ln => {
                let y = x.ln();
                let grad = Box::new(move |dy: f64| -> f64 { dy / x });
                (y, grad)
            }
            MonoAD::Sqrt => {
                let y = x.sqrt();
                let grad = Box::new(move |dy: f64| -> f64 { dy / (2.0 * y) });
                (y, grad)
            }
            MonoAD::Abs => {
                let y = x.abs();
                let sign = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let grad = Box::new(move |dy: f64| -> f64 { dy * sign });
                (y, grad)
            }
        };
        // For backward(): Box::from(boxed_closure) → returns the Box as-is (identity)
        // For backward_arc(): Arc::from(boxed_closure) → converts Box to Arc
        (y, W::from(grad_fn))
    }

    /// Compute forward pass and return gradient function.
    ///
    /// Returns a tuple of (value, gradient_function). The gradient function
    /// takes a cotangent (typically 1.0 for full derivative) and returns
    /// the gradient at the input point.
    ///
    /// The result is Box-wrapped by default. If you need Arc for sharing across threads,
    /// convert using `Arc::from(box_fn)`.
    ///
    /// # Arguments
    ///
    /// * `exprs` - Slice of operations to compose, in reverse order
    /// * `x` - Input value to evaluate at
    ///
    /// # Returns
    ///
    /// Tuple of (output_value, gradient_function)
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::{MonoAD, mono_ops};
    /// use std::sync::Arc;
    ///
    /// let ops = mono_ops![sin, cos];
    /// let (value, grad_fn) = MonoAD::compute_grad(&ops, 1.0);
    /// let gradient = grad_fn(1.0);
    ///
    /// // Convert to Arc if needed for sharing
    /// let arc_grad_fn: Arc<dyn Fn(f64) -> f64> = Arc::from(grad_fn);
    /// ```
    #[must_use = "gradient computation is expensive; discarding the result is likely a bug"]
    #[inline]
    pub fn compute_grad_generic<W>(exprs: &[MonoAD], x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
    {
        let mut value = x;
        // Pre-allocate with capacity to avoid reallocations
        let mut backprops: Vec<W> = Vec::with_capacity(exprs.len());

        // Compute backward pass for each operation
        for &op in exprs {
            let (new_value, backprop) = op.backward_generic(value);
            value = new_value;
            backprops.push(backprop);
        }

        // Chain all the backward functions
        let backward_fn = Box::new(move |cotangent: f64| -> f64 {
            let mut grad = cotangent;
            for backprop in backprops.iter().rev() {
                grad = backprop(grad);
            }
            grad
        });

        (value, W::from(backward_fn))
    }

    #[must_use = "gradient computation is expensive; discarding the result is likely a bug"]
    pub fn compute_grad(exprs: &[MonoAD], x: f64) -> BackwardResultBox {
        Self::compute_grad_generic::<Box<DynMathFn>>(exprs, x)
    }

    /// Compute forward pass and gradient function with checked-domain validation.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::DomainError`] if an intermediate value violates a
    /// real-domain restriction before its operation is evaluated.
    #[must_use = "gradient computation is expensive; discarding the result is likely a bug"]
    pub fn compute_grad_checked(exprs: &[MonoAD], x: f64) -> Result<BackwardResultBox> {
        let mut value = x;
        let mut backprops: Vec<Box<DynMathFn>> = Vec::with_capacity(exprs.len());

        for &op in exprs {
            op.check_domain(value)?;
            let (new_value, backprop) = op.backward_generic(value);
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

    /// Compute the second derivative (Hessian for single-variable functions).
    ///
    /// For a single-variable function f(x), the Hessian is a 1x1 matrix
    /// containing f''(x), which we return as a scalar value.
    ///
    /// This uses finite differences on the gradient function:
    /// f''(x) ≈ (f'(x + ε) - f'(x - ε)) / (2ε)
    ///
    /// # Arguments
    ///
    /// * `exprs` - Slice of operations to apply in sequence
    /// * `x` - Input value to evaluate at
    ///
    /// # Returns
    ///
    /// The second derivative f''(x) at the input point
    ///
    /// # Accuracy
    ///
    /// Uses central difference with ε = 1e-5, giving approximately 1e-4 accuracy.
    /// For higher precision, use analytical derivatives or smaller ε values.
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::{MonoAD, mono_ops};
    ///
    /// // f(x) = sin(x), f''(x) = -sin(x)
    /// let ops = mono_ops![sin];
    /// let second_deriv = MonoAD::compute_hessian(&ops, 0.5);
    /// // At x = 0.5: -sin(0.5) ≈ -0.4794
    /// assert!((second_deriv - (-0.5_f64.sin())).abs() < 1e-4);
    /// ```
    #[must_use = "second derivative computation is expensive; discarding the result is likely a bug"]
    pub fn compute_hessian(exprs: &[MonoAD], x: f64) -> f64 {
        let epsilon = 1e-5;

        // Compute gradient at x + ε
        let (_value_plus, grad_fn_plus) = Self::compute_grad(exprs, x + epsilon);
        let grad_plus = grad_fn_plus(1.0);

        // Compute gradient at x - ε
        let (_value_minus, grad_fn_minus) = Self::compute_grad(exprs, x - epsilon);
        let grad_minus = grad_fn_minus(1.0);

        // Central difference formula for second derivative
        (grad_plus - grad_minus) / (2.0 * epsilon)
    }

    /// Compute the finite-difference Hessian with checked-domain validation.
    ///
    /// This validates the two perturbed gradient evaluations used by the central
    /// difference. Points near a domain boundary may therefore return an error even
    /// when the unperturbed forward value is defined.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::DomainError`] if `x + ε` or `x - ε` violates a
    /// real-domain restriction at any intermediate operation.
    #[must_use = "second derivative computation is expensive; discarding the result is likely a bug"]
    pub fn compute_hessian_checked(exprs: &[MonoAD], x: f64) -> Result<f64> {
        let epsilon = 1e-5;

        let (_value_plus, grad_fn_plus) = Self::compute_grad_checked(exprs, x + epsilon)?;
        let grad_plus = grad_fn_plus(1.0);

        let (_value_minus, grad_fn_minus) = Self::compute_grad_checked(exprs, x - epsilon)?;
        let grad_minus = grad_fn_minus(1.0);

        Ok((grad_plus - grad_minus) / (2.0 * epsilon))
    }
}
