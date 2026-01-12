use super::types::*;

/// Multi-variable automatic differentiation operations.
///
/// Represents operations in a computational graph for functions with multiple inputs.
/// Each operation takes references to previous results via indices.
///
/// # Examples
///
/// ```
/// use autodiff::{MultiAD, multi_ops};
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
/// let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]);
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
    Div,
    /// Sine function: sin(x)
    Sin,
    /// Cosine function: cos(x)
    Cos,
    /// Tangent function: tan(x)
    Tan,
    /// Exponential function: exp(x)
    Exp,
    /// Natural logarithm: ln(x)
    Ln,
}

impl MultiAD {
    /// Forward pass: compute the output of this operation given inputs
    fn forward(&self, args: &[f64]) -> f64 {
        match self {
            MultiAD::Inp => {
                assert!(args.len() == 1, "Inp expects 1 argument");
                args[0]
            }
            MultiAD::Sin => {
                assert!(args.len() == 1, "Sin expects 1 argument");
                args[0].sin()
            }
            MultiAD::Cos => {
                assert!(args.len() == 1, "Cos expects 1 argument");
                args[0].cos()
            }
            MultiAD::Tan => {
                assert!(args.len() == 1, "Tan expects 1 argument");
                args[0].tan()
            }
            MultiAD::Exp => {
                assert!(args.len() == 1, "Exp expects 1 argument");
                args[0].exp()
            }
            MultiAD::Ln => {
                assert!(args.len() == 1, "Ln expects 1 argument");
                args[0].ln()
            }
            MultiAD::Add => {
                assert!(args.len() == 2, "Add expects 2 arguments");
                args[0] + args[1]
            }
            MultiAD::Sub => {
                assert!(args.len() == 2, "Sub expects 2 arguments");
                args[0] - args[1]
            }
            MultiAD::Mul => {
                assert!(args.len() == 2, "Mul expects 2 arguments");
                args[0] * args[1]
            }
            MultiAD::Div => {
                assert!(args.len() == 2, "Div expects 2 arguments");
                args[0] / args[1]
            }
        }
    }

    /// Backward pass: compute local gradients ∂output/∂inputs
    /// Returns a boxed closure that computes gradients given a cotangent value
    fn backward_generic<W>(&self, args: &[f64]) -> W
    where
        W: From<Box<DynGradFn>>,
    {
        let backward_fn: Box<dyn Fn(f64) -> Vec<f64>> = match self {
            MultiAD::Inp => {
                assert!(args.len() == 1, "Inp expects 1 argument");
                Box::new(|zcotangent: f64| vec![zcotangent])
            }
            MultiAD::Sin => {
                assert!(args.len() == 1, "Sin expects 1 argument");
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * arg_val.cos();
                    vec![x_cotangent]
                })
            }
            MultiAD::Cos => {
                assert!(args.len() == 1, "Cos expects 1 argument");
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * -arg_val.sin();
                    vec![x_cotangent]
                })
            }
            MultiAD::Tan => {
                assert!(args.len() == 1, "Tan expects 1 argument");
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * (1.0 / arg_val.cos().powi(2));
                    vec![x_cotangent]
                })
            }
            MultiAD::Exp => {
                assert!(args.len() == 1, "Exp expects 1 argument");
                let exp_val = args[0].exp();
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * exp_val;
                    vec![x_cotangent]
                })
            }
            MultiAD::Ln => {
                assert!(args.len() == 1, "Ln expects 1 argument");
                let arg_val = args[0];
                Box::new(move |z_cotangent: f64| {
                    let x_cotangent = z_cotangent * (1.0 / arg_val);
                    vec![x_cotangent]
                })
            }
            MultiAD::Add => {
                assert!(args.len() == 2, "Add expects 2 arguments");
                Box::new(|z_cotangent: f64| vec![z_cotangent, z_cotangent])
            }
            MultiAD::Sub => {
                assert!(args.len() == 2, "Sub expects 2 arguments");
                Box::new(|z_cotangent: f64| vec![z_cotangent, -z_cotangent])
            }
            MultiAD::Mul => {
                assert!(args.len() == 2, "Mul expects 2 arguments");
                let arg0 = args[0];
                let arg1 = args[1];
                Box::new(move |z_cotangent: f64| vec![z_cotangent * arg1, z_cotangent * arg0])
            }
            MultiAD::Div => {
                assert!(args.len() == 2, "Div expects 2 arguments");
                let arg0 = args[0];
                let arg1 = args[1];
                Box::new(move |z_cotangent: f64| {
                    vec![z_cotangent / arg1, -z_cotangent * arg0 / arg1.powi(2)]
                })
            }
        };
        W::from(backward_fn)
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
    /// # Examples
    ///
    /// ```
    /// use autodiff::{MultiAD, multi_ops};
    ///
    /// let exprs = multi_ops![(inp, 0), (inp, 1), (add, 0, 1)];
    /// let result = MultiAD::compute(&exprs, &[2.0, 3.0]);
    /// assert!((result - 5.0).abs() < 1e-10);
    /// ```
    pub fn compute(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> f64 {
        let mut values: Vec<f64> = inputs.to_vec();

        for (op, arg_indices) in exprs {
            if *op == MultiAD::Inp {
                continue; // Input values are already in the values array
            }

            // Gather the argument values from the computation graph
            let arg_values: Vec<f64> = arg_indices.iter().map(|&i| values[i]).collect();

            // Compute this operation
            let value = op.forward(&arg_values);
            values.push(value);
        }

        // Return the final computed value
        values.last().copied().unwrap_or(0.0)
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
    /// # Examples
    ///
    /// ```
    /// use autodiff::{MultiAD, multi_ops};
    /// use std::sync::Arc;
    ///
    /// let exprs = multi_ops![
    ///     (inp, 0), (inp, 1),
    ///     (add, 0, 1), (sin, 0), (mul, 2, 3)
    /// ];
    /// let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]);
    /// let gradients = grad_fn(1.0);
    ///
    /// // Convert to Arc if needed for sharing
    /// let arc_grad_fn: Arc<dyn Fn(f64) -> Vec<f64>> = Arc::from(grad_fn);
    /// ```
    fn compute_grad_generic<W>(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> (f64, W)
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
            let value = op.forward(&arg_values);
            values.push(value);

            // Store the backward operation (which captures necessary values)
            backward_ops.push(op.backward_generic(&arg_values));
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

        (final_value, W::from(backward_fn))
    }

    pub fn compute_grad(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> BackwardResultBox {
        Self::compute_grad_generic::<Box<DynGradFn>>(exprs, inputs)
    }
}
