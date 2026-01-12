use super::types::*;
use crate::error::{AutodiffError, Result};

/// Multi-variable automatic differentiation operations.
///
/// Represents operations in a computational graph for functions with multiple inputs.
/// Each operation takes references to previous results via indices.
///
/// # Examples
///
/// ```
/// use petite_ad::{MultiAD, multi_ops};
///
/// // Build graph: f(x, y) = sin(x) * (x + y)
/// let exprs = multi_ops![
///     (inp, 0),    // x at index 0
///     (inp, 1),    // y at index 1
///     (add, 0, 1), // x + y at index 2
///     (sin, 0),    // sin(x) at index 3
///     (mul, 2, 3), // sin(x) * (x + y) at index 4
/// ];
///
/// let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]).unwrap();
/// let gradients = grad_fn(1.0);
/// println!("f(0.6, 1.4) = {}", value);
/// println!("∇f = {:?}", gradients);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiAD {
    /// Input placeholder - references an input variable
    Inp,
    /// Addition: a + b
    Add,
    /// Subtraction: a - b
    Sub,
    /// Multiplication: a * b
    Mul,
    /// Division: a / b
    ///
    /// # Notes
    /// - Delegates to `f64::div()`, which returns `inf` for division by zero
    /// - Returns `NaN` for `0.0 / 0.0`
    Div,
    /// Power: a^b (a raised to the power of b)
    ///
    /// # Notes
    /// - Delegates to `f64::powf()`
    /// - For `x^n` where n is an integer, consider using repeated multiplication
    Pow,
    /// Sine function: sin(x)
    ///
    /// # Notes
    /// - Delegates to `f64::sin()`, which operates in radians
    /// - Returns values in the range `[-1.0, 1.0]`
    Sin,
    /// Cosine function: cos(x)
    ///
    /// # Notes
    /// - Delegates to `f64::cos()`, which operates in radians
    /// - Returns values in the range `[-1.0, 1.0]`
    Cos,
    /// Tangent function: tan(x)
    ///
    /// # Notes
    /// - Delegates to `f64::tan()`, which operates in radians
    /// - Returns very large values near `π/2 + kπ` (asymptotes)
    Tan,
    /// Exponential function: exp(x)
    ///
    /// # Notes
    /// - Delegates to `f64::exp()`
    /// - Returns `inf` for very large inputs (> ~709 for f64)
    /// - Returns `0.0` for very large negative inputs (< ~-745 for f64)
    Exp,
    /// Natural logarithm: ln(x)
    ///
    /// # Notes
    /// - Delegates to `f64::ln()`
    /// - Returns `NaN` for negative inputs
    /// - Returns `-inf` for `ln(0.0)`
    Ln,
    /// Square root: sqrt(x)
    ///
    /// # Notes
    /// - Delegates to `f64::sqrt()`
    /// - Returns `NaN` for negative inputs
    Sqrt,
    /// Absolute value: abs(x)
    ///
    /// # Notes
    /// - Delegates to `f64::abs()`
    /// - Subgradient at x=0 is 0 (consistent with common practice)
    Abs,
}

impl MultiAD {
    /// Get the name of this operation (for error messages and arity checking)
    fn op_name(&self) -> &'static str {
        match self {
            MultiAD::Inp => "Inp",
            MultiAD::Add => "Add",
            MultiAD::Sub => "Sub",
            MultiAD::Mul => "Mul",
            MultiAD::Div => "Div",
            MultiAD::Pow => "Pow",
            MultiAD::Sin => "Sin",
            MultiAD::Cos => "Cos",
            MultiAD::Tan => "Tan",
            MultiAD::Exp => "Exp",
            MultiAD::Ln => "Ln",
            MultiAD::Sqrt => "Sqrt",
            MultiAD::Abs => "Abs",
        }
    }

    /// Get the expected arity for this operation
    fn expected_arity(&self) -> usize {
        match self {
            MultiAD::Inp | MultiAD::Sin | MultiAD::Cos | MultiAD::Tan | MultiAD::Exp | MultiAD::Ln | MultiAD::Sqrt | MultiAD::Abs => 1,
            MultiAD::Add | MultiAD::Sub | MultiAD::Mul | MultiAD::Div | MultiAD::Pow => 2,
        }
    }
    /// Forward pass: compute the output of this operation given inputs
    fn forward(&self, args: &[f64]) -> Result<f64> {
        Ok(match self {
            MultiAD::Inp => {
                AutodiffError::check_arity("Inp", 1, args.len())?;
                args[0]
            }
            MultiAD::Sin => {
                AutodiffError::check_arity("Sin", 1, args.len())?;
                args[0].sin()
            }
            MultiAD::Cos => {
                AutodiffError::check_arity("Cos", 1, args.len())?;
                args[0].cos()
            }
            MultiAD::Tan => {
                AutodiffError::check_arity("Tan", 1, args.len())?;
                args[0].tan()
            }
            MultiAD::Exp => {
                AutodiffError::check_arity("Exp", 1, args.len())?;
                args[0].exp()
            }
            MultiAD::Ln => {
                AutodiffError::check_arity("Ln", 1, args.len())?;
                args[0].ln()
            }
            MultiAD::Sqrt => {
                AutodiffError::check_arity("Sqrt", 1, args.len())?;
                args[0].sqrt()
            }
            MultiAD::Abs => {
                AutodiffError::check_arity("Abs", 1, args.len())?;
                args[0].abs()
            }
            MultiAD::Add => {
                AutodiffError::check_arity("Add", 2, args.len())?;
                args[0] + args[1]
            }
            MultiAD::Sub => {
                AutodiffError::check_arity("Sub", 2, args.len())?;
                args[0] - args[1]
            }
            MultiAD::Mul => {
                AutodiffError::check_arity("Mul", 2, args.len())?;
                args[0] * args[1]
            }
            MultiAD::Div => {
                AutodiffError::check_arity("Div", 2, args.len())?;
                args[0] / args[1]
            }
            MultiAD::Pow => {
                AutodiffError::check_arity("Pow", 2, args.len())?;
                args[0].powf(args[1])
            }
        })
    }

    /// Backward pass: compute local gradients ∂output/∂inputs
    /// Returns a boxed closure that computes gradients given a cotangent value
    fn backward_generic<W>(&self, args: &[f64]) -> Result<W>
    where
        W: From<Box<DynGradFn>>,
    {
        AutodiffError::check_arity(self.op_name(), self.expected_arity(), args.len())?;

        let backward_fn: Box<dyn Fn(f64) -> Vec<f64>> = match self {
            MultiAD::Inp => {
                Box::new(|zcotangent: f64| vec![zcotangent])
            }
            MultiAD::Sin => {
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * arg_val.cos();
                    vec![x_cotangent]
                })
            }
            MultiAD::Cos => {
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * -arg_val.sin();
                    vec![x_cotangent]
                })
            }
            MultiAD::Tan => {
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * (1.0 / arg_val.cos().powi(2));
                    vec![x_cotangent]
                })
            }
            MultiAD::Exp => {
                let exp_val = args[0].exp();
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * exp_val;
                    vec![x_cotangent]
                })
            }
            MultiAD::Ln => {
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * (1.0 / arg_val);
                    vec![x_cotangent]
                })
            }
            MultiAD::Add => {
                Box::new(|z_cotangent: f64| vec![z_cotangent, z_cotangent])
            }
            MultiAD::Sub => {
                Box::new(|z_cotangent: f64| vec![z_cotangent, -z_cotangent])
            }
            MultiAD::Mul => {
                let arg0 = args[0];
                let arg1 = args[1];
                Box::new(move |z_cotangent: f64| vec![z_cotangent * arg1, z_cotangent * arg0])
            }
            MultiAD::Div => {
                let arg0 = args[0];
                let arg1 = args[1];
                Box::new(move |z_cotangent: f64| {
                    vec![z_cotangent / arg1, -z_cotangent * arg0 / arg1.powi(2)]
                })
            }
            MultiAD::Pow => {
                let base = args[0];
                let exp = args[1];
                Box::new(move |z_cotangent: f64| {
                    // d(a^b)/da = b * a^(b-1)
                    let d_base = z_cotangent * exp * base.powf(exp - 1.0);
                    // d(a^b)/db = a^b * ln(a)
                    let d_exp = z_cotangent * base.powf(exp) * base.ln();
                    vec![d_base, d_exp]
                })
            }
            MultiAD::Sqrt => {
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    // d(sqrt(x))/dx = 1/(2*sqrt(x))
                    let x_cotangent = z_cotangent / (2.0 * arg_val.sqrt());
                    vec![x_cotangent]
                })
            }
            MultiAD::Abs => {
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    // d(|x|)/dx = sign(x) where sign(0) = 0
                    let sign = if arg_val >= 0.0 { 1.0 } else { -1.0 };
                    vec![z_cotangent * sign]
                })
            }
        };
        Ok(W::from(backward_fn))
    }

    /// Compute forward pass only (no gradient computation).
    ///
    /// Evaluates the computational graph to produce the final output value.
    ///
    /// # Arguments
    ///
    /// * `exprs` - Slice of (operation, indices) pairs defining the computation graph
    /// * `inputs` - Input values for the function
    ///
    /// # Errors
    ///
    /// Returns `Err(AutodiffError)` if an operation receives incorrect arity.
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::{MultiAD, multi_ops};
    ///
    /// let exprs = multi_ops![(inp, 0), (inp, 1), (add, 0, 1)];
    /// let result = MultiAD::compute(&exprs, &[2.0, 3.0]).unwrap();
    /// assert!((result - 5.0).abs() < 1e-10);
    /// ```
    #[must_use = "forward computation is expensive; discarding the result is likely a bug"]
    pub fn compute(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<f64> {
        let mut values: Vec<f64> = inputs.to_vec();

        for (op, arg_indices) in exprs {
            if *op == MultiAD::Inp {
                continue; // Input values are already in the values array
            }

            // Gather the argument values from the computation graph
            let arg_values: Vec<f64> = arg_indices.iter().map(|&i| values[i]).collect();

            // Compute this operation
            let value = op.forward(&arg_values)?;
            values.push(value);
        }

        // Return the final computed value
        Ok(values.last().copied().unwrap_or(0.0))
    }

    /// Compute forward pass and return gradient function.
    ///
    /// Returns a tuple of (value, gradient_function). The gradient function
    /// takes a cotangent (typically 1.0) and returns a vector of gradients
    /// with respect to each input.
    ///
    /// The result is Box-wrapped by default. If you need Arc for sharing across threads,
    /// convert using `Arc::from(box_fn)`.
    ///
    /// # Arguments
    ///
    /// * `exprs` - Computational graph as (operation, indices) pairs
    /// * `inputs` - Input values to evaluate at
    ///
    /// # Returns
    ///
    /// Tuple of (output_value, gradient_function)
    ///
    /// # Errors
    ///
    /// Returns `Err(AutodiffError)` if an operation receives incorrect arity.
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::{MultiAD, multi_ops};
    /// use std::sync::Arc;
    ///
    /// let exprs = multi_ops![
    ///     (inp, 0), (inp, 1),
    ///     (add, 0, 1), (sin, 0), (mul, 2, 3)
    /// ];
    /// let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]).unwrap();
    /// let gradients = grad_fn(1.0);
    ///
    /// // Convert to Arc if needed for sharing
    /// let arc_grad_fn: Arc<dyn Fn(f64) -> Vec<f64>> = Arc::from(grad_fn);
    /// ```
    #[must_use = "gradient computation is expensive; discarding the result is likely a bug"]
    fn compute_grad_generic<W>(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<(f64, W)>
    where
        W: From<Box<DynGradFn>> + std::ops::Deref<Target = DynGradFn> + 'static,
    {
        // Pre-allocate with capacity for better performance
        let estimated_size = inputs.len() + exprs.len();
        let mut values: Vec<f64> = Vec::with_capacity(estimated_size);
        values.extend_from_slice(inputs);

        let mut backward_ops: Vec<Box<DynGradFn>> = Vec::with_capacity(exprs.len());
        let mut arg_indices_list: Vec<Vec<usize>> = Vec::with_capacity(exprs.len());

        // Forward pass: compute all values and track backward operations
        for (op, args) in exprs {
            if *op == MultiAD::Inp {
                continue;
            }
            let arg_values: Vec<f64> = args.iter().map(|&i| values[i]).collect();
            let value = op.forward(&arg_values)?;
            values.push(value);

            // Store the backward operation (which captures necessary values)
            backward_ops.push(op.backward_generic(&arg_values)?);
            arg_indices_list.push(args.clone());
        }

        let final_value = values.last().copied().unwrap_or(0.0);

        // Clone the data we need for the backward pass
        let num_inputs = inputs.len();
        let values_clone = values;

        let backward_fn = Box::new(move |cotangent: f64| -> Vec<f64> {
            let mut cotangent_values = vec![0.0; values_clone.len()];
            cotangent_values[values_clone.len() - 1] = cotangent;

            // Backward pass: propagate cotangents from output to inputs
            for (i, (backward_op, arg_indices)) in backward_ops
                .iter()
                .zip(arg_indices_list.iter())
                .rev() // Process operations in reverse order
                .enumerate()
            {
                let output_idx = values_clone.len() - 1 - i;
                let current_cotangent_value = cotangent_values[output_idx];
                let argv_cotangents = backward_op(current_cotangent_value);

                // Accumulate gradients for each input argument
                for (arg_idx, arg_cotangent) in arg_indices.iter().zip(argv_cotangents) {
                    cotangent_values[*arg_idx] += arg_cotangent;
                }
            }

            cotangent_values[..num_inputs].to_vec()
        });

        Ok((final_value, W::from(backward_fn)))
    }

    #[must_use = "gradient computation is expensive; discarding the result is likely a bug"]
    pub fn compute_grad(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<BackwardResultBox> {
        Self::compute_grad_generic::<Box<DynGradFn>>(exprs, inputs)
    }
}
