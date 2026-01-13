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
    /// Exponential function: exp(x)
    ///
    /// Derivative: exp(x)
    ///
    /// # Notes
    /// - Delegates to `f64::exp()`
    /// - Returns `inf` for very large inputs (> ~709 for f64)
    /// - Returns `0.0` for very large negative inputs (< ~-745 for f64)
    Exp,
    Neg,
}

impl MonoAD {
    /// Compute the forward pass for a single operation.
    ///
    /// This is an internal helper that computes just the forward value
    /// without building gradient closures.
    fn forward(&self, x: f64) -> f64 {
        match self {
            MonoAD::Sin => x.sin(),
            MonoAD::Cos => x.cos(),
            MonoAD::Exp => x.exp(),
            MonoAD::Neg => -x,
        }
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
    pub fn compute(exprs: &[MonoAD], x: f64) -> f64 {
        let mut value = x;
        for expr in exprs {
            value = expr.forward(value);
        }
        value
    }

    // Helper that works with Box wrapper type
    // Box<dyn Fn> is the common type that all arms return
    fn backward_generic<W>(self, x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>>,
    {
        let (y, grad_fn): (f64, Box<dyn Fn(f64) -> f64>) = match self {
            MonoAD::Sin => {
                let y = x.sin();
                let grad = Box::new(move |dy: f64| -> f64 { dy * x.cos() });
                (y, grad)
            }
            MonoAD::Cos => {
                let y = x.cos();
                let grad = Box::new(move |dy: f64| -> f64 { dy * -x.sin() });
                (y, grad)
            }
            MonoAD::Exp => {
                let y = x.exp();
                let grad = Box::new(move |dy: f64| -> f64 { dy * y });
                (y, grad)
            }
            MonoAD::Neg => {
                let y = -x;
                let grad = Box::new(move |dy: f64| -> f64 { dy * -1.0 });
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
    pub fn compute_grad_generic<W>(exprs: &[MonoAD], x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
    {
        let mut value = x;
        let mut backprops: Vec<W> = Vec::new();

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
}
