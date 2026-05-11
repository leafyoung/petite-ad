use super::op_rules;
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// - Gradients with respect to the exponent use `ln(a)`, so real-valued
    ///   differentiability requires a positive base for non-constant exponents
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
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Rectified linear unit: max(0, x) with subgradient 0 at x = 0.
    Relu,
    /// Stable softplus: ln(1 + exp(x)).
    Log1pExp,
    /// Stable binary log-sum-exp: ln(exp(a) + exp(b)).
    LogAddExp,
    /// Negation: -x
    Neg,
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
    /// Forward pass: compute the output of this operation given inputs
    #[inline]
    pub(crate) fn forward(&self, args: &[f64]) -> Result<f64> {
        op_rules::forward_value(*self, args)
    }

    #[inline]
    pub(crate) fn forward_checked(&self, args: &[f64]) -> Result<f64> {
        op_rules::forward_value_checked(*self, args)
    }

    /// Backward pass: compute local gradients ∂output/∂inputs
    /// Returns a boxed closure that computes gradients given a cotangent value
    #[inline]
    pub(crate) fn backward_generic<W>(&self, args: &[f64]) -> Result<W>
    where
        W: From<Box<DynGradFn>>,
    {
        let value = op_rules::forward_value(*self, args)?;
        let local_grads = match self {
            MultiAD::Inp => vec![1.0],
            _ => op_rules::first_derivatives(*self, args, value)?,
        };
        let backward_fn: Box<dyn Fn(f64) -> Vec<f64>> = Box::new(move |z_cotangent: f64| {
            local_grads
                .iter()
                .map(|grad| z_cotangent * grad)
                .collect::<Vec<f64>>()
        });
        Ok(W::from(backward_fn))
    }

    /// Validate that a graph index points to an available value.
    #[inline]
    pub(crate) fn check_value_index(index: usize, values_len: usize) -> Result<()> {
        if index < values_len {
            Ok(())
        } else {
            Err(AutodiffError::IndexOutOfBounds {
                index,
                max_index: values_len.saturating_sub(1),
            })
        }
    }

    /// Validate an input marker operation.
    #[inline]
    fn check_input_marker(arg_indices: &[usize], input_len: usize) -> Result<()> {
        AutodiffError::check_arity("Inp", 1, arg_indices.len())?;
        Self::check_value_index(arg_indices[0], input_len)
    }

    /// Gather operation arguments after validating all referenced graph indices.
    #[inline]
    pub(crate) fn gather_arg_values(arg_indices: &[usize], values: &[f64]) -> Result<Vec<f64>> {
        let mut arg_values = Vec::with_capacity(arg_indices.len());
        for &index in arg_indices {
            Self::check_value_index(index, values.len())?;
            arg_values.push(values[index]);
        }
        Ok(arg_values)
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
    #[inline]
    pub fn compute(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<f64> {
        // Pre-allocate with estimated capacity
        let estimated_size = inputs.len() + exprs.len();
        let mut values: Vec<f64> = Vec::with_capacity(estimated_size);
        values.extend_from_slice(inputs);
        let mut final_output_index = inputs.len().checked_sub(1);

        for (op, arg_indices) in exprs {
            if *op == MultiAD::Inp {
                Self::check_input_marker(arg_indices, inputs.len())?;
                final_output_index = Some(arg_indices[0]);
                continue; // Input values are already in the values array
            }

            // Gather the argument values from the computation graph.
            let arg_values = Self::gather_arg_values(arg_indices, &values)?;

            // Compute this operation
            let value = op.forward(&arg_values)?;
            values.push(value);
            final_output_index = Some(values.len() - 1);
        }

        // Return the final computed value. Empty graph with no inputs is the zero scalar.
        Ok(final_output_index.map(|index| values[index]).unwrap_or(0.0))
    }

    /// Compute forward pass with opt-in checked real-domain validation.
    #[must_use = "forward computation is expensive; discarding the result is likely a bug"]
    #[inline]
    pub fn compute_checked(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<f64> {
        let estimated_size = inputs.len() + exprs.len();
        let mut values: Vec<f64> = Vec::with_capacity(estimated_size);
        values.extend_from_slice(inputs);
        let mut final_output_index = inputs.len().checked_sub(1);

        for (op, arg_indices) in exprs {
            if *op == MultiAD::Inp {
                Self::check_input_marker(arg_indices, inputs.len())?;
                final_output_index = Some(arg_indices[0]);
                continue;
            }

            let arg_values = Self::gather_arg_values(arg_indices, &values)?;
            let value = op.forward_checked(&arg_values)?;
            values.push(value);
            final_output_index = Some(values.len() - 1);
        }

        Ok(final_output_index.map(|index| values[index]).unwrap_or(0.0))
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
    #[inline]
    pub fn compute_grad_generic<W>(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
    ) -> Result<(f64, W)>
    where
        W: From<Box<DynGradFn>> + std::ops::Deref<Target = DynGradFn> + 'static,
    {
        // Pre-allocate with capacity for better performance
        let estimated_size = inputs.len() + exprs.len();
        let mut values: Vec<f64> = Vec::with_capacity(estimated_size);
        values.extend_from_slice(inputs);
        let mut final_output_index = inputs.len().checked_sub(1);

        let mut backward_ops: Vec<Box<DynGradFn>> = Vec::with_capacity(exprs.len());
        let mut arg_indices_list: Vec<Vec<usize>> = Vec::with_capacity(exprs.len());

        // Forward pass: compute all values and track backward operations
        for (op, args) in exprs {
            if *op == MultiAD::Inp {
                Self::check_input_marker(args, inputs.len())?;
                final_output_index = Some(args[0]);
                continue;
            }
            let arg_values = Self::gather_arg_values(args, &values)?;
            let value = op.forward(&arg_values)?;
            values.push(value);
            final_output_index = Some(values.len() - 1);

            // Store the backward operation (which captures necessary values)
            backward_ops.push(op.backward_generic(&arg_values)?);
            arg_indices_list.push(args.clone());
        }

        let final_value = final_output_index.map(|index| values[index]).unwrap_or(0.0);

        // Clone the data we need for the backward pass
        let num_inputs = inputs.len();
        let values_clone = values;
        let final_output_index_clone = final_output_index;

        let backward_fn = Box::new(move |cotangent: f64| -> Vec<f64> {
            let Some(final_output_index) = final_output_index_clone else {
                return Vec::new();
            };

            let mut cotangent_values = vec![0.0; values_clone.len()];
            cotangent_values[final_output_index] = cotangent;

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
    pub fn compute_grad(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
    ) -> Result<BackwardResultBox> {
        Self::compute_grad_generic::<Box<DynGradFn>>(exprs, inputs)
    }

    /// Compute forward pass and return gradient function with checked-domain validation.
    #[must_use = "gradient computation is expensive; discarding the result is likely a bug"]
    pub fn compute_grad_checked(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
    ) -> Result<BackwardResultBox> {
        let estimated_size = inputs.len() + exprs.len();
        let mut values: Vec<f64> = Vec::with_capacity(estimated_size);
        values.extend_from_slice(inputs);
        let mut final_output_index = inputs.len().checked_sub(1);

        let mut backward_ops: Vec<Box<DynGradFn>> = Vec::with_capacity(exprs.len());
        let mut arg_indices_list: Vec<Vec<usize>> = Vec::with_capacity(exprs.len());

        for (op, args) in exprs {
            if *op == MultiAD::Inp {
                Self::check_input_marker(args, inputs.len())?;
                final_output_index = Some(args[0]);
                continue;
            }
            let arg_values = Self::gather_arg_values(args, &values)?;
            let value = op.forward_checked(&arg_values)?;
            values.push(value);
            final_output_index = Some(values.len() - 1);
            backward_ops.push(op.backward_generic(&arg_values)?);
            arg_indices_list.push(args.clone());
        }

        let final_value = final_output_index.map(|index| values[index]).unwrap_or(0.0);
        let num_inputs = inputs.len();
        let values_clone = values;
        let final_output_index_clone = final_output_index;

        let backward_fn = Box::new(move |cotangent: f64| -> Vec<f64> {
            let Some(final_output_index) = final_output_index_clone else {
                return Vec::new();
            };

            let mut cotangent_values = vec![0.0; values_clone.len()];
            cotangent_values[final_output_index] = cotangent;

            for (i, (backward_op, arg_indices)) in backward_ops
                .iter()
                .zip(arg_indices_list.iter())
                .rev()
                .enumerate()
            {
                let output_idx = values_clone.len() - 1 - i;
                let current_cotangent_value = cotangent_values[output_idx];
                let argv_cotangents = backward_op(current_cotangent_value);
                for (arg_idx, arg_cotangent) in arg_indices.iter().zip(argv_cotangents) {
                    cotangent_values[*arg_idx] += arg_cotangent;
                }
            }

            cotangent_values[..num_inputs].to_vec()
        });

        Ok((final_value, backward_fn))
    }

    /// Compute the Hessian matrix using finite differences on the gradient.
    ///
    /// The Hessian is the matrix of second-order partial derivatives:
    /// H\[i\]\[j\] = ∂²f/∂xᵢ∂xⱼ
    ///
    /// This uses central finite differences on gradients to compute how each
    /// gradient component changes with respect to each input.
    ///
    /// # Arguments
    ///
    /// * `exprs` - Computational graph as (operation, indices) pairs
    /// * `inputs` - Input values to evaluate at
    ///
    /// # Returns
    ///
    /// A 2D vector representing the Hessian matrix where result\[i\]\[j\] = ∂²f/∂xᵢ∂xⱼ
    ///
    /// # Complexity
    ///
    /// O(n²) where n is the number of inputs. For each of n inputs, this
    /// computes two gradient evaluations.
    ///
    /// # Examples
    ///
    /// ```
    /// use petite_ad::{MultiAD, multi_ops};
    ///
    /// // f(x, y) = x² + y² (Hessian is [[2, 0], [0, 2]])
    /// let exprs = multi_ops![
    ///     (inp, 0), (inp, 1),
    ///     (mul, 0, 0), (mul, 1, 1),
    ///     (add, 2, 3)
    /// ];
    /// let hessian = MultiAD::compute_hessian(&exprs, &[2.0, 3.0]).unwrap();
    /// assert!((hessian[0][0] - 2.0).abs() < 1e-6);
    /// assert!((hessian[0][1] - 0.0).abs() < 1e-6);
    /// assert!((hessian[1][0] - 0.0).abs() < 1e-6);
    /// assert!((hessian[1][1] - 2.0).abs() < 1e-6);
    /// ```
    #[must_use = "Hessian computation is expensive; discarding the result is likely a bug"]
    pub fn compute_hessian(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        let num_inputs = inputs.len();
        let epsilon = 1e-5;
        let mut hessian = vec![vec![0.0; num_inputs]; num_inputs];

        if num_inputs == 0 {
            let (_value, _grad_fn) = Self::compute_grad(exprs, inputs)?;
            return Ok(hessian);
        }

        // For each input variable, compute how the gradient changes using a
        // central difference for second-order accuracy on smooth functions.
        for j in 0..num_inputs {
            let mut inputs_plus = inputs.to_vec();
            inputs_plus[j] += epsilon;

            let mut inputs_minus = inputs.to_vec();
            inputs_minus[j] -= epsilon;

            let (_value_plus, grad_fn_plus) = Self::compute_grad(exprs, &inputs_plus)?;
            let grad_plus = grad_fn_plus(1.0);

            let (_value_minus, grad_fn_minus) = Self::compute_grad(exprs, &inputs_minus)?;
            let grad_minus = grad_fn_minus(1.0);

            for i in 0..num_inputs {
                // ∂²f/∂xᵢ∂xⱼ ≈ (∂f/∂xᵢ(x + εeⱼ) - ∂f/∂xᵢ(x - εeⱼ)) / (2ε)
                hessian[i][j] = (grad_plus[i] - grad_minus[i]) / (2.0 * epsilon);
            }
        }

        Ok(hessian)
    }

    /// Compute a single row of the Hessian matrix using finite differences.
    ///
    /// Computes ∇(∂f/∂xᵢ), which is the i-th row of the Hessian.
    /// The j-th element is ∂²f/∂xᵢ∂xⱼ.
    ///
    /// # Arguments
    ///
    /// * `exprs` - Computational graph
    /// * `inputs` - Input values
    /// * `grad_idx` - Index of the gradient component to differentiate
    ///
    /// # Returns
    ///
    /// A vector representing the i-th row of the Hessian
    ///
    /// # Errors
    ///
    /// Returns `Err(AutodiffError::IndexOutOfBounds)` if `grad_idx` is not a
    /// valid input index, or if the graph references unavailable values.
    pub fn compute_hessian_row(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
        grad_idx: usize,
    ) -> Result<Vec<f64>> {
        let num_inputs = inputs.len();
        Self::check_value_index(grad_idx, num_inputs)?;

        let epsilon = 1e-5;
        let mut hessian_row = vec![0.0; num_inputs];

        // For each input variable, compute how this gradient component changes
        // using a central difference for second-order accuracy.
        for j in 0..num_inputs {
            let mut inputs_plus = inputs.to_vec();
            inputs_plus[j] += epsilon;

            let mut inputs_minus = inputs.to_vec();
            inputs_minus[j] -= epsilon;

            let (_value_plus, grad_fn_plus) = Self::compute_grad(exprs, &inputs_plus)?;
            let grad_plus = grad_fn_plus(1.0);

            let (_value_minus, grad_fn_minus) = Self::compute_grad(exprs, &inputs_minus)?;
            let grad_minus = grad_fn_minus(1.0);

            // ∂²f/∂x_grad_idx∂xⱼ ≈ (∂f/∂x_grad_idx(x + εeⱼ) - ∂f/∂x_grad_idx(x - εeⱼ)) / (2ε)
            hessian_row[j] = (grad_plus[grad_idx] - grad_minus[grad_idx]) / (2.0 * epsilon);
        }

        Ok(hessian_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_ops;
    use crate::test_utils::approx_eq_eps as approx_eq;
    use std::sync::Arc;

    // ---- forward / forward_checked ----

    #[test]
    fn test_forward_sin() {
        let val = MultiAD::Sin.forward(&[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.sin(), 1e-12));
    }

    #[test]
    fn test_forward_cos() {
        let val = MultiAD::Cos.forward(&[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.cos(), 1e-12));
    }

    #[test]
    fn test_forward_tan() {
        let val = MultiAD::Tan.forward(&[0.5]).unwrap();
        assert!(approx_eq(val, 0.5_f64.tan(), 1e-12));
    }

    #[test]
    fn test_forward_exp() {
        let val = MultiAD::Exp.forward(&[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.exp(), 1e-12));
    }

    #[test]
    fn test_forward_ln() {
        let val = MultiAD::Ln.forward(&[2.0]).unwrap();
        assert!(approx_eq(val, 2.0_f64.ln(), 1e-12));
    }

    #[test]
    fn test_forward_sqrt() {
        let val = MultiAD::Sqrt.forward(&[4.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-12));
    }

    #[test]
    fn test_forward_neg() {
        let val = MultiAD::Neg.forward(&[3.0]).unwrap();
        assert!(approx_eq(val, -3.0, 1e-12));
    }

    #[test]
    fn test_forward_abs() {
        let val = MultiAD::Abs.forward(&[-5.0]).unwrap();
        assert!(approx_eq(val, 5.0, 1e-12));
    }

    #[test]
    fn test_forward_tanh() {
        let val = MultiAD::Tanh.forward(&[1.0]).unwrap();
        assert!(approx_eq(val, 1.0_f64.tanh(), 1e-12));
    }

    #[test]
    fn test_forward_relu_positive() {
        let val = MultiAD::Relu.forward(&[2.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-12));
    }

    #[test]
    fn test_forward_relu_negative() {
        let val = MultiAD::Relu.forward(&[-2.0]).unwrap();
        assert!(approx_eq(val, 0.0, 1e-12));
    }

    #[test]
    fn test_forward_log1p_exp() {
        let val = MultiAD::Log1pExp.forward(&[0.0]).unwrap();
        assert!(approx_eq(val, 2.0_f64.ln(), 1e-6));
    }

    #[test]
    fn test_forward_log_add_exp() {
        let val = MultiAD::LogAddExp.forward(&[0.0, 0.0]).unwrap();
        assert!(approx_eq(val, 2.0_f64.ln(), 1e-6));
    }

    #[test]
    fn test_forward_add() {
        let val = MultiAD::Add.forward(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(val, 5.0, 1e-12));
    }

    #[test]
    fn test_forward_sub() {
        let val = MultiAD::Sub.forward(&[5.0, 3.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-12));
    }

    #[test]
    fn test_forward_mul() {
        let val = MultiAD::Mul.forward(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(val, 6.0, 1e-12));
    }

    #[test]
    fn test_forward_div() {
        let val = MultiAD::Div.forward(&[6.0, 3.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-12));
    }

    #[test]
    fn test_forward_pow() {
        let val = MultiAD::Pow.forward(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(val, 8.0, 1e-12));
    }

    #[test]
    fn test_forward_checked_ln_negative() {
        let result = MultiAD::Ln.forward_checked(&[-1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_checked_sqrt_negative() {
        let result = MultiAD::Sqrt.forward_checked(&[-1.0]);
        assert!(result.is_err());
    }

    // ---- check_value_index / check_input_marker error cases ----

    #[test]
    fn test_check_value_index_ok() {
        assert!(MultiAD::check_value_index(0, 5).is_ok());
        assert!(MultiAD::check_value_index(4, 5).is_ok());
    }

    #[test]
    fn test_check_value_index_out_of_bounds() {
        let result = MultiAD::check_value_index(5, 5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AutodiffError::IndexOutOfBounds { .. }
        ));
    }

    #[test]
    fn test_gather_arg_values_ok() {
        let values = vec![1.0, 2.0, 3.0];
        let result = MultiAD::gather_arg_values(&[0, 2], &values).unwrap();
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn test_gather_arg_values_out_of_bounds() {
        let values = vec![1.0, 2.0];
        let result = MultiAD::gather_arg_values(&[0, 5], &values);
        assert!(result.is_err());
    }

    // ---- compute / compute_checked edge cases ----

    #[test]
    fn test_compute_empty_graph_no_inputs() {
        // Empty graph with no inputs should return 0.0
        let result = MultiAD::compute(&[], &[]).unwrap();
        assert!(approx_eq(result, 0.0, 1e-12));
    }

    #[test]
    fn test_compute_single_input_passthrough() {
        // Just an input marker
        let exprs = multi_ops![(inp, 0)];
        let result = MultiAD::compute(&exprs, &[42.0]).unwrap();
        assert!(approx_eq(result, 42.0, 1e-12));
    }

    #[test]
    fn test_compute_checked_basic() {
        let exprs = multi_ops![(inp, 0), (inp, 1), (add, 0, 1)];
        let result = MultiAD::compute_checked(&exprs, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(result, 5.0, 1e-12));
    }

    #[test]
    fn test_compute_checked_catches_domain_error() {
        // ln(-1) should fail with checked forward
        let exprs = multi_ops![(inp, 0), (ln, 0)];
        let result = MultiAD::compute_checked(&exprs, &[-1.0]);
        assert!(result.is_err());
    }

    // ---- compute_grad edge cases ----

    #[test]
    fn test_compute_grad_empty_graph() {
        let (value, grad_fn) = MultiAD::compute_grad(&[], &[]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-12));
        let grads = grad_fn(1.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn test_compute_grad_single_input() {
        // f(x) = x * x
        let exprs = multi_ops![(inp, 0), (mul, 0, 0)];
        let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[3.0]).unwrap();
        assert!(approx_eq(value, 9.0, 1e-10));
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 6.0, 1e-10)); // d/dx(x^2) = 2x = 6
    }

    #[test]
    fn test_compute_grad_with_different_cotangent() {
        // Test backward with cotangent != 1.0
        let exprs = multi_ops![(inp, 0), (inp, 1), (mul, 0, 1)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[2.0, 3.0]).unwrap();
        let grads = grad_fn(2.0); // cotangent = 2
                                  // df/dx = y = 3, scaled by 2 = 6
                                  // df/dy = x = 2, scaled by 2 = 4
        assert!(approx_eq(grads[0], 6.0, 1e-10));
        assert!(approx_eq(grads[1], 4.0, 1e-10));
    }

    #[test]
    fn test_compute_grad_generic_arc() {
        // Test compute_grad_generic with Arc wrapper
        let exprs = multi_ops![(inp, 0), (inp, 1), (add, 0, 1)];
        let (value, grad_fn) =
            MultiAD::compute_grad_generic::<Arc<DynGradFn>>(&exprs, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 5.0, 1e-10));
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 1.0, 1e-10));
        assert!(approx_eq(grads[1], 1.0, 1e-10));
    }

    #[test]
    fn test_compute_grad_input_marker_error_wrong_arity() {
        // Inp with wrong arity
        let exprs = vec![(MultiAD::Inp, vec![0, 1])];
        let result = MultiAD::compute_grad(&exprs, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_grad_input_marker_error_out_of_bounds() {
        // Inp referencing input index beyond inputs length
        let exprs = multi_ops![(inp, 5)];
        let result = MultiAD::compute_grad(&exprs, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_grad_index_out_of_bounds_in_op() {
        // Operation referencing a non-existent value index
        let exprs = vec![(MultiAD::Add, vec![0, 99])];
        let result = MultiAD::compute_grad(&exprs, &[1.0]);
        assert!(result.is_err());
    }

    // ---- compute_grad_checked ----

    #[test]
    fn test_compute_grad_checked_basic() {
        let exprs = multi_ops![(inp, 0), (inp, 1), (mul, 0, 1)];
        let (value, grad_fn) = MultiAD::compute_grad_checked(&exprs, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 6.0, 1e-10));
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 3.0, 1e-10));
        assert!(approx_eq(grads[1], 2.0, 1e-10));
    }

    #[test]
    fn test_compute_grad_checked_empty() {
        let (value, grad_fn) = MultiAD::compute_grad_checked(&[], &[]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-12));
        let grads = grad_fn(1.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn test_compute_grad_checked_domain_error() {
        let exprs = multi_ops![(inp, 0), (ln, 0)];
        let result = MultiAD::compute_grad_checked(&exprs, &[-1.0]);
        assert!(result.is_err());
    }

    // ---- compute_hessian ----

    #[test]
    fn test_compute_hessian_quadratic() {
        // f(x, y) = x^2 + y^2 => H = [[2,0],[0,2]]
        let exprs = multi_ops![(inp, 0), (inp, 1), (mul, 0, 0), (mul, 1, 1), (add, 2, 3)];
        let h = MultiAD::compute_hessian(&exprs, &[2.0, 3.0]).unwrap();
        assert!((h[0][0] - 2.0).abs() < 1e-4);
        assert!((h[0][1]).abs() < 1e-4);
        assert!((h[1][0]).abs() < 1e-4);
        assert!((h[1][1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_hessian_empty_inputs() {
        let h = MultiAD::compute_hessian(&[], &[]).unwrap();
        assert!(h.is_empty());
    }

    // ---- compute_hessian_row ----

    #[test]
    fn test_compute_hessian_row_quadratic() {
        // f(x, y) = x^2 + y^2 => H[0] = [2, 0]
        let exprs = multi_ops![(inp, 0), (inp, 1), (mul, 0, 0), (mul, 1, 1), (add, 2, 3)];
        let row = MultiAD::compute_hessian_row(&exprs, &[2.0, 3.0], 0).unwrap();
        assert!((row[0] - 2.0).abs() < 1e-4);
        assert!((row[1]).abs() < 1e-4);
    }

    #[test]
    fn test_compute_hessian_row_out_of_bounds() {
        let exprs = multi_ops![(inp, 0), (inp, 1), (add, 0, 1)];
        let result = MultiAD::compute_hessian_row(&exprs, &[1.0, 2.0], 5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AutodiffError::IndexOutOfBounds { .. }
        ));
    }

    // ---- Various individual op gradient checks ----

    #[test]
    fn test_grad_sub() {
        // f(x, y) = x - y => df/dx = 1, df/dy = -1
        let exprs = multi_ops![(inp, 0), (inp, 1), (sub, 0, 1)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[5.0, 3.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 1.0, 1e-10));
        assert!(approx_eq(grads[1], -1.0, 1e-10));
    }

    #[test]
    fn test_grad_div() {
        // f(x, y) = x / y => df/dx = 1/y, df/dy = -x/y^2
        let exprs = multi_ops![(inp, 0), (inp, 1), (div, 0, 1)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[6.0, 3.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 1.0 / 3.0, 1e-10));
        assert!(approx_eq(grads[1], -6.0 / 9.0, 1e-10));
    }

    #[test]
    fn test_grad_sin() {
        // f(x) = sin(x) => df/dx = cos(x)
        let exprs = multi_ops![(inp, 0), (sin, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[1.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 1.0_f64.cos(), 1e-10));
    }

    #[test]
    fn test_grad_cos() {
        // f(x) = cos(x) => df/dx = -sin(x)
        let exprs = multi_ops![(inp, 0), (cos, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[1.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], -1.0_f64.sin(), 1e-10));
    }

    #[test]
    fn test_grad_exp() {
        // f(x) = exp(x) => df/dx = exp(x)
        let exprs = multi_ops![(inp, 0), (exp, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[1.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 1.0_f64.exp(), 1e-10));
    }

    #[test]
    fn test_grad_ln() {
        // f(x) = ln(x) => df/dx = 1/x
        let exprs = multi_ops![(inp, 0), (ln, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[2.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 0.5, 1e-10));
    }

    #[test]
    fn test_grad_sqrt() {
        // f(x) = sqrt(x) => df/dx = 1/(2*sqrt(x))
        let exprs = multi_ops![(inp, 0), (sqrt, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[4.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 0.25, 1e-10));
    }

    #[test]
    fn test_grad_neg() {
        // f(x) = -x => df/dx = -1
        let exprs = multi_ops![(inp, 0), (neg, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[3.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], -1.0, 1e-10));
    }

    #[test]
    fn test_grad_abs() {
        // f(x) = abs(x) => df/dx = sign(x)
        let exprs = multi_ops![(inp, 0), (abs, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[-3.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], -1.0, 1e-10));
    }

    #[test]
    fn test_grad_tanh() {
        // f(x) = tanh(x) => df/dx = 1 - tanh(x)^2
        let exprs = multi_ops![(inp, 0), (tanh, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.5]).unwrap();
        let grads = grad_fn(1.0);
        let expected = 1.0 - 0.5_f64.tanh().powi(2);
        assert!(approx_eq(grads[0], expected, 1e-10));
    }

    #[test]
    fn test_grad_relu_positive() {
        let exprs = multi_ops![(inp, 0), (relu, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[2.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 1.0, 1e-10));
    }

    #[test]
    fn test_grad_relu_negative() {
        let exprs = multi_ops![(inp, 0), (relu, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[-2.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 0.0, 1e-10));
    }

    #[test]
    fn test_grad_pow() {
        // f(x, y) = x^y => df/dx = y * x^(y-1), df/dy = x^y * ln(x)
        let exprs = multi_ops![(inp, 0), (inp, 1), (pow, 0, 1)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[2.0, 3.0]).unwrap();
        let grads = grad_fn(1.0);
        assert!(approx_eq(grads[0], 3.0 * 4.0, 1e-10)); // y*x^(y-1) = 3*4 = 12
        assert!(approx_eq(grads[1], 8.0 * 2.0_f64.ln(), 1e-10)); // x^y*ln(x)
    }

    #[test]
    fn test_grad_log1p_exp() {
        // f(x) = log1p_exp(x) = ln(1+exp(x)) => df/dx = sigmoid(x) = exp(x)/(1+exp(x))
        let exprs = multi_ops![(inp, 0), (log1p_exp, 0)];
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[1.0]).unwrap();
        let grads = grad_fn(1.0);
        let expected = 1.0_f64.exp() / (1.0 + 1.0_f64.exp());
        assert!(approx_eq(grads[0], expected, 1e-10));
    }

    #[test]
    fn test_grad_tan() {
        // f(x) = tan(x) => df/dx = 1/cos(x)^2
        let exprs = multi_ops![(inp, 0), (tan, 0)];
        let x = 0.5_f64;
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[x]).unwrap();
        let grads = grad_fn(1.0);
        let expected = 1.0 / x.cos().powi(2);
        assert!(approx_eq(grads[0], expected, 1e-10));
    }

    #[test]
    fn test_grad_log_add_exp() {
        // f(x, y) = log_add_exp(x, y) = ln(exp(x) + exp(y))
        let exprs = multi_ops![(inp, 0), (inp, 1), (log_add_exp, 0, 1)];
        let x = 1.0_f64;
        let y = 2.0_f64;
        let (_value, grad_fn) = MultiAD::compute_grad(&exprs, &[x, y]).unwrap();
        let grads = grad_fn(1.0);
        let denom = x.exp() + y.exp();
        assert!(approx_eq(grads[0], x.exp() / denom, 1e-10));
        assert!(approx_eq(grads[1], y.exp() / denom, 1e-10));
    }

    #[test]
    fn test_compute_input_index_error() {
        // Inp with wrong arity (0 args)
        let exprs = vec![(MultiAD::Inp, vec![])];
        let result = MultiAD::compute(&exprs, &[1.0]);
        assert!(result.is_err());
    }
}
