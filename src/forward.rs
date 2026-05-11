//! Public forward-mode automatic differentiation APIs.
//!
//! This module exposes a small forward-mode surface on top of the library's
//! existing operation sets:
//!
//! - [`ForwardAD::differentiate`] for single-variable `MonoAD` expressions
//! - [`ForwardAD::directional_derivative`] for multivariate `MultiAD` tuple graphs
//! - [`ForwardAD::directional_derivative_graph`] for reusable [`crate::Graph`] values

use crate::multi::op_rules;
use crate::{AutodiffError, Graph, GraphNode, MonoAD, MultiAD, Result};

/// Result of a forward-mode evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardValue {
    /// Primal function value.
    pub value: f64,
    /// Forward tangent / directional derivative.
    pub tangent: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Dual {
    value: f64,
    tangent: f64,
}

impl Dual {
    #[inline(always)]
    fn new(value: f64, tangent: f64) -> Self {
        Self { value, tangent }
    }

    #[inline(always)]
    fn constant(value: f64) -> Self {
        Self {
            value,
            tangent: 0.0,
        }
    }
}

/// Public forward-mode utilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardAD;

impl ForwardAD {
    /// Fill a reusable seed buffer with the `active`-th basis vector.
    fn basis_seed_into(dim: usize, active: usize, seed: &mut Vec<f64>) {
        seed.clear();
        seed.resize(dim, 0.0);
        seed[active] = 1.0;
    }

    /// Create a fresh basis seed vector (convenience wrapper).
    #[allow(dead_code)]
    fn basis_seed(dim: usize, active: usize) -> Vec<f64> {
        let mut seed = Vec::with_capacity(dim);
        Self::basis_seed_into(dim, active, &mut seed);
        seed
    }

    /// Evaluate a mono expression and its derivative at `x`.
    #[must_use]
    pub fn differentiate(exprs: &[MonoAD], x: f64) -> ForwardValue {
        let mut dual = Dual::new(x, 1.0);
        for expr in exprs {
            dual = match expr {
                MonoAD::Sin => Dual::new(dual.value.sin(), dual.value.cos() * dual.tangent),
                MonoAD::Cos => Dual::new(dual.value.cos(), -dual.value.sin() * dual.tangent),
                MonoAD::Tan => {
                    let cos_val = dual.value.cos();
                    Dual::new(dual.value.tan(), dual.tangent / (cos_val * cos_val))
                }
                MonoAD::Exp => {
                    let exp_val = dual.value.exp();
                    Dual::new(exp_val, exp_val * dual.tangent)
                }
                MonoAD::Neg => Dual::new(-dual.value, -dual.tangent),
                MonoAD::Ln => Dual::new(dual.value.ln(), dual.tangent / dual.value),
                MonoAD::Sqrt => {
                    let sqrt_val = dual.value.sqrt();
                    Dual::new(sqrt_val, dual.tangent / (2.0 * sqrt_val))
                }
                MonoAD::Abs => {
                    let sign = if dual.value > 0.0 {
                        1.0
                    } else if dual.value < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    Dual::new(dual.value.abs(), sign * dual.tangent)
                }
            };
        }
        ForwardValue {
            value: dual.value,
            tangent: dual.tangent,
        }
    }

    /// Evaluate a mono expression and derivative with checked real-domain validation.
    pub fn differentiate_checked(exprs: &[MonoAD], x: f64) -> Result<ForwardValue> {
        let mut dual = Dual::new(x, 1.0);
        for expr in exprs {
            match expr {
                MonoAD::Ln if dual.value <= 0.0 => {
                    return Err(AutodiffError::domain("Ln", "input must be positive"));
                }
                MonoAD::Sqrt if dual.value < 0.0 => {
                    return Err(AutodiffError::domain("Sqrt", "input must be non-negative"));
                }
                _ => {}
            }
            let step = Self::differentiate(&[*expr], dual.value);
            dual = Dual::new(step.value, step.tangent * dual.tangent);
        }
        Ok(ForwardValue {
            value: dual.value,
            tangent: dual.tangent,
        })
    }

    /// Evaluate only the primal mono expression value, skipping tangent computation.
    #[must_use]
    pub fn compute(exprs: &[MonoAD], x: f64) -> f64 {
        let mut value = x;
        for expr in exprs {
            value = match expr {
                MonoAD::Sin => value.sin(),
                MonoAD::Cos => value.cos(),
                MonoAD::Tan => value.tan(),
                MonoAD::Exp => value.exp(),
                MonoAD::Neg => -value,
                MonoAD::Ln => value.ln(),
                MonoAD::Sqrt => value.sqrt(),
                MonoAD::Abs => value.abs(),
            };
        }
        value
    }

    /// Evaluate only the primal mono value with checked real-domain validation.
    pub fn compute_checked(exprs: &[MonoAD], x: f64) -> Result<f64> {
        let mut value = x;
        for expr in exprs {
            match expr {
                MonoAD::Ln if value <= 0.0 => {
                    return Err(AutodiffError::domain("Ln", "input must be positive"));
                }
                MonoAD::Sqrt if value < 0.0 => {
                    return Err(AutodiffError::domain("Sqrt", "input must be non-negative"));
                }
                _ => {}
            }
            value = match expr {
                MonoAD::Sin => value.sin(),
                MonoAD::Cos => value.cos(),
                MonoAD::Tan => value.tan(),
                MonoAD::Exp => value.exp(),
                MonoAD::Neg => -value,
                MonoAD::Ln => value.ln(),
                MonoAD::Sqrt => value.sqrt(),
                MonoAD::Abs => value.abs(),
            };
        }
        Ok(value)
    }

    /// Evaluate a multivariate tuple graph and its directional derivative.
    ///
    /// `seed` must have the same length as `inputs`; the returned tangent is the
    /// Jacobian-vector product `J(x) · seed` for scalar-valued graphs.
    pub fn directional_derivative(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
        seed: &[f64],
    ) -> Result<ForwardValue> {
        if seed.len() != inputs.len() {
            return Err(AutodiffError::InvalidGraph {
                reason: "seed vector length must match input length",
            });
        }

        let mut values: Vec<Dual> = Vec::with_capacity(inputs.len() + exprs.len());
        for (&value, &tangent) in inputs.iter().zip(seed.iter()) {
            values.push(Dual::new(value, tangent));
        }

        let mut final_output_index = inputs.len().checked_sub(1);

        for (op, arg_indices) in exprs {
            if *op == MultiAD::Inp {
                AutodiffError::check_arity("Inp", 1, arg_indices.len())?;
                MultiAD::check_value_index(arg_indices[0], inputs.len())?;
                final_output_index = Some(arg_indices[0]);
                continue;
            }

            let mut arg_values = [0.0f64; 2];
            let mut arg_duals = [Dual::constant(0.0); 2];
            AutodiffError::check_arity(
                op_rules::op_name(*op),
                op_rules::expected_arity(*op),
                arg_indices.len(),
            )?;
            for (i, &index) in arg_indices.iter().enumerate() {
                MultiAD::check_value_index(index, values.len())?;
                arg_values[i] = values[index].value;
                arg_duals[i] = values[index];
            }

            let value = op.forward(&arg_values[..arg_indices.len()])?;
            let arg_tangents = [arg_duals[0].tangent, arg_duals[1].tangent];
            let tangent = match op {
                MultiAD::Inp => unreachable!("input markers are handled earlier"),
                _ => op_rules::directional_tangent(
                    *op,
                    &arg_values[..arg_indices.len()],
                    &arg_tangents[..arg_indices.len()],
                    value,
                )?,
            };

            values.push(Dual::new(value, tangent));
            final_output_index = Some(values.len() - 1);
        }

        let Some(final_output_index) = final_output_index else {
            return Ok(ForwardValue {
                value: 0.0,
                tangent: 0.0,
            });
        };

        Ok(ForwardValue {
            value: values[final_output_index].value,
            tangent: values[final_output_index].tangent,
        })
    }

    /// Evaluate a multivariate tuple graph and directional derivative with checked domains.
    pub fn directional_derivative_checked(
        exprs: &[(MultiAD, Vec<usize>)],
        inputs: &[f64],
        seed: &[f64],
    ) -> Result<ForwardValue> {
        if seed.len() != inputs.len() {
            return Err(AutodiffError::InvalidGraph {
                reason: "seed vector length must match input length",
            });
        }

        let mut values: Vec<Dual> = Vec::with_capacity(inputs.len() + exprs.len());
        for (&value, &tangent) in inputs.iter().zip(seed.iter()) {
            values.push(Dual::new(value, tangent));
        }

        let mut final_output_index = inputs.len().checked_sub(1);

        for (op, arg_indices) in exprs {
            if *op == MultiAD::Inp {
                AutodiffError::check_arity("Inp", 1, arg_indices.len())?;
                MultiAD::check_value_index(arg_indices[0], inputs.len())?;
                final_output_index = Some(arg_indices[0]);
                continue;
            }

            let mut arg_values = [0.0f64; 2];
            let mut arg_duals = [Dual::constant(0.0); 2];
            AutodiffError::check_arity(
                op_rules::op_name(*op),
                op_rules::expected_arity(*op),
                arg_indices.len(),
            )?;
            for (i, &index) in arg_indices.iter().enumerate() {
                MultiAD::check_value_index(index, values.len())?;
                arg_values[i] = values[index].value;
                arg_duals[i] = values[index];
            }

            let value = op.forward_checked(&arg_values[..arg_indices.len()])?;
            let arg_tangents = [arg_duals[0].tangent, arg_duals[1].tangent];
            let tangent = op_rules::directional_tangent(
                *op,
                &arg_values[..arg_indices.len()],
                &arg_tangents[..arg_indices.len()],
                value,
            )?;

            values.push(Dual::new(value, tangent));
            final_output_index = Some(values.len() - 1);
        }

        let Some(final_output_index) = final_output_index else {
            return Ok(ForwardValue {
                value: 0.0,
                tangent: 0.0,
            });
        };

        Ok(ForwardValue {
            value: values[final_output_index].value,
            tangent: values[final_output_index].tangent,
        })
    }

    /// Compute the full gradient of a scalar multivariate tuple graph.
    pub fn gradient(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<Vec<f64>> {
        let mut gradient = Vec::with_capacity(inputs.len());
        let mut seed = Vec::with_capacity(inputs.len());
        for active in 0..inputs.len() {
            Self::basis_seed_into(inputs.len(), active, &mut seed);
            let value = Self::directional_derivative(exprs, inputs, &seed)?;
            gradient.push(value.tangent);
        }
        Ok(gradient)
    }

    /// Compute the full gradient of a scalar reusable graph.
    pub fn gradient_graph(graph: &Graph, inputs: &[f64]) -> Result<Vec<f64>> {
        let mut gradient = Vec::with_capacity(inputs.len());
        let mut seed = Vec::with_capacity(inputs.len());
        for active in 0..inputs.len() {
            Self::basis_seed_into(inputs.len(), active, &mut seed);
            let value = Self::directional_derivative_graph(graph, inputs, &seed)?;
            gradient.push(value.tangent);
        }
        Ok(gradient)
    }

    /// Compute a Jacobian for multiple scalar outputs represented as tuple graphs.
    ///
    /// Each entry in `outputs` is treated as one scalar component of a vector-valued
    /// function `f: R^n -> R^m`, and the returned matrix has shape `m x n`.
    pub fn jacobian(
        outputs: &[Vec<(MultiAD, Vec<usize>)>],
        inputs: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        let mut jacobian = Vec::with_capacity(outputs.len());
        for exprs in outputs {
            jacobian.push(Self::gradient(exprs, inputs)?);
        }
        Ok(jacobian)
    }

    /// Compute a Jacobian for multiple scalar outputs represented as reusable graphs.
    pub fn jacobian_graphs(outputs: &[Graph], inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut jacobian = Vec::with_capacity(outputs.len());
        for graph in outputs {
            jacobian.push(Self::gradient_graph(graph, inputs)?);
        }
        Ok(jacobian)
    }

    /// Evaluate a reusable graph and its directional derivative.
    pub fn directional_derivative_graph(
        graph: &Graph,
        inputs: &[f64],
        seed: &[f64],
    ) -> Result<ForwardValue> {
        Self::directional_derivative_graph_inner(graph, inputs, seed, false)
    }

    /// Evaluate a reusable graph and directional derivative with checked domains.
    pub fn directional_derivative_graph_checked(
        graph: &Graph,
        inputs: &[f64],
        seed: &[f64],
    ) -> Result<ForwardValue> {
        Self::directional_derivative_graph_inner(graph, inputs, seed, true)
    }

    fn directional_derivative_graph_inner(
        graph: &Graph,
        inputs: &[f64],
        seed: &[f64],
        checked: bool,
    ) -> Result<ForwardValue> {
        if inputs.len() != graph.num_inputs() {
            return Err(AutodiffError::InvalidGraph {
                reason: "graph input length must match graph.num_inputs()",
            });
        }
        if seed.len() != inputs.len() {
            return Err(AutodiffError::InvalidGraph {
                reason: "seed vector length must match input length",
            });
        }

        let output_index = graph.effective_output_node();
        let mut values: Vec<Dual> = Vec::with_capacity(graph.num_inputs() + graph.len());
        for (&value, &tangent) in inputs.iter().zip(seed.iter()) {
            values.push(Dual::new(value, tangent));
        }

        for node in graph.nodes() {
            match node {
                GraphNode::Constant(value) => {
                    values.push(Dual::constant(*value));
                }
                GraphNode::Operation { op, inputs } => {
                    let mut arg_values: Vec<f64> = Vec::with_capacity(inputs.len());
                    let mut arg_duals: Vec<Dual> = Vec::with_capacity(inputs.len());
                    for &index in inputs {
                        MultiAD::check_value_index(index, values.len())?;
                        arg_values.push(values[index].value);
                        arg_duals.push(values[index]);
                    }
                    let value = if checked {
                        op.forward_checked(&arg_values)?
                    } else {
                        op.forward(&arg_values)?
                    };
                    let arg_tangents: Vec<f64> =
                        arg_duals.iter().map(|dual| dual.tangent).collect();
                    let tangent = match op {
                        MultiAD::Inp => unreachable!("graph nodes do not store input markers"),
                        _ => op_rules::directional_tangent(*op, &arg_values, &arg_tangents, value)?,
                    };
                    values.push(Dual::new(value, tangent));
                }
            }
        }

        let Some(output_index) = output_index else {
            return Ok(ForwardValue {
                value: 0.0,
                tangent: 0.0,
            });
        };
        MultiAD::check_value_index(output_index, values.len())?;

        Ok(ForwardValue {
            value: values[output_index].value,
            tangent: values[output_index].tangent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mono_ops;
    use crate::test_utils::approx_eq_eps as approx_eq;

    #[test]
    fn test_forward_mono_derivative() {
        let exprs = mono_ops![sin, exp];
        let result = ForwardAD::differentiate(&exprs, 0.5);
        let expected_value = 0.5_f64.sin().exp();
        let expected_tangent = expected_value * 0.5_f64.cos();
        assert!(approx_eq(result.value, expected_value, 1e-10));
        assert!(approx_eq(result.tangent, expected_tangent, 1e-10));
    }

    #[test]
    fn test_forward_multi_directional_derivative() {
        let exprs = vec![
            (MultiAD::Inp, vec![0]),
            (MultiAD::Inp, vec![1]),
            (MultiAD::Mul, vec![0, 1]),
            (MultiAD::Sin, vec![2]),
        ];
        let result = ForwardAD::directional_derivative(&exprs, &[2.0, 3.0], &[1.0, -1.0]).unwrap();
        let xy = 2.0_f64 * 3.0_f64;
        let expected_value = xy.sin();
        let expected_tangent = xy.cos() * (3.0 - 2.0);
        assert!(approx_eq(result.value, expected_value, 1e-10));
        assert!(approx_eq(result.tangent, expected_tangent, 1e-10));
    }

    #[test]
    fn test_forward_graph_directional_derivative_with_constant() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let c = graph.constant(2.0);
        let x_sq = graph.mul(x, x);
        graph.add(x_sq, c);

        let result = ForwardAD::directional_derivative_graph(&graph, &[3.0], &[1.0]).unwrap();
        assert!(approx_eq(result.value, 11.0, 1e-10));
        assert!(approx_eq(result.tangent, 6.0, 1e-10));
    }

    #[test]
    fn test_forward_graph_respects_explicit_output() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let x_sq = graph.mul(x, x);
        graph.mul(x_sq, x);
        graph.set_output(x_sq).unwrap();

        let result = ForwardAD::directional_derivative_graph(&graph, &[2.0], &[1.0]).unwrap();
        assert!(approx_eq(result.value, 4.0, 1e-10));
        assert!(approx_eq(result.tangent, 4.0, 1e-10));
    }

    #[test]
    fn test_forward_gradient_matches_scalar_graph() {
        let exprs = vec![
            (MultiAD::Inp, vec![0]),
            (MultiAD::Inp, vec![1]),
            (MultiAD::Mul, vec![0, 1]),
            (MultiAD::Sin, vec![2]),
        ];
        let gradient = ForwardAD::gradient(&exprs, &[2.0, 3.0]).unwrap();
        let xy = 2.0_f64 * 3.0_f64;
        assert!(approx_eq(gradient[0], xy.cos() * 3.0, 1e-10));
        assert!(approx_eq(gradient[1], xy.cos() * 2.0, 1e-10));
    }

    #[test]
    fn test_forward_checked_domain_errors() {
        let mono_error = ForwardAD::differentiate_checked(&[MonoAD::Ln], -1.0).unwrap_err();
        assert_eq!(
            mono_error,
            AutodiffError::DomainError {
                operation: "Ln",
                reason: "input must be positive",
            }
        );

        let exprs = vec![(MultiAD::Ln, vec![0])];
        assert!(ForwardAD::directional_derivative_checked(&exprs, &[0.0], &[1.0]).is_err());

        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sqrt(x);
        assert!(ForwardAD::directional_derivative_graph_checked(&graph, &[-1.0], &[1.0]).is_err());
    }

    #[test]
    fn test_forward_reverse_gradient_table() {
        type MultiExpr = Vec<(MultiAD, Vec<usize>)>;
        let cases: Vec<(MultiExpr, Vec<f64>)> = vec![
            (vec![(MultiAD::Tan, vec![0])], vec![0.25]),
            (vec![(MultiAD::Ln, vec![0])], vec![2.0]),
            (vec![(MultiAD::Sqrt, vec![0])], vec![4.0]),
            (
                vec![
                    (MultiAD::Inp, vec![0]),
                    (MultiAD::Inp, vec![1]),
                    (MultiAD::Div, vec![0, 1]),
                ],
                vec![4.0, 2.0],
            ),
            (
                vec![
                    (MultiAD::Inp, vec![0]),
                    (MultiAD::Inp, vec![1]),
                    (MultiAD::Pow, vec![0, 1]),
                ],
                vec![2.0, 3.0],
            ),
        ];

        for (exprs, inputs) in cases {
            let reverse = MultiAD::compute_grad(&exprs, &inputs).unwrap().1(1.0);
            let forward = ForwardAD::gradient(&exprs, &inputs).unwrap();
            for (left, right) in reverse.iter().zip(forward.iter()) {
                assert!(approx_eq(*left, *right, 1e-8));
            }
        }
    }

    #[test]
    fn test_forward_jacobian_for_two_outputs() {
        let outputs = vec![
            vec![
                (MultiAD::Inp, vec![0]),
                (MultiAD::Inp, vec![1]),
                (MultiAD::Add, vec![0, 1]),
            ],
            vec![
                (MultiAD::Inp, vec![0]),
                (MultiAD::Inp, vec![1]),
                (MultiAD::Mul, vec![0, 1]),
            ],
        ];
        let jacobian = ForwardAD::jacobian(&outputs, &[2.0, 3.0]).unwrap();
        assert_eq!(jacobian.len(), 2);
        assert!(approx_eq(jacobian[0][0], 1.0, 1e-10));
        assert!(approx_eq(jacobian[0][1], 1.0, 1e-10));
        assert!(approx_eq(jacobian[1][0], 3.0, 1e-10));
        assert!(approx_eq(jacobian[1][1], 2.0, 1e-10));
    }

    // --- Additional tests for uncovered lines ---

    #[test]
    fn test_differentiate_cos() {
        let result = ForwardAD::differentiate(&[MonoAD::Cos], 0.5);
        assert!(approx_eq(result.value, 0.5_f64.cos(), 1e-10));
        assert!(approx_eq(result.tangent, -0.5_f64.sin(), 1e-10));
    }

    #[test]
    fn test_differentiate_tan() {
        let result = ForwardAD::differentiate(&[MonoAD::Tan], 0.5);
        let cos_val = 0.5_f64.cos();
        assert!(approx_eq(result.value, 0.5_f64.tan(), 1e-10));
        assert!(approx_eq(result.tangent, 1.0 / (cos_val * cos_val), 1e-10));
    }

    #[test]
    fn test_differentiate_neg() {
        let result = ForwardAD::differentiate(&[MonoAD::Neg], 2.0);
        assert!(approx_eq(result.value, -2.0, 1e-10));
        assert!(approx_eq(result.tangent, -1.0, 1e-10));
    }

    #[test]
    fn test_differentiate_ln() {
        let result = ForwardAD::differentiate(&[MonoAD::Ln], 2.0);
        assert!(approx_eq(result.value, 2.0_f64.ln(), 1e-10));
        assert!(approx_eq(result.tangent, 0.5, 1e-10));
    }

    #[test]
    fn test_differentiate_sqrt() {
        let result = ForwardAD::differentiate(&[MonoAD::Sqrt], 4.0);
        assert!(approx_eq(result.value, 2.0, 1e-10));
        assert!(approx_eq(result.tangent, 0.25, 1e-10));
    }

    #[test]
    fn test_differentiate_abs_positive() {
        let result = ForwardAD::differentiate(&[MonoAD::Abs], 3.0);
        assert!(approx_eq(result.value, 3.0, 1e-10));
        assert!(approx_eq(result.tangent, 1.0, 1e-10));
    }

    #[test]
    fn test_differentiate_abs_negative() {
        let result = ForwardAD::differentiate(&[MonoAD::Abs], -3.0);
        assert!(approx_eq(result.value, 3.0, 1e-10));
        assert!(approx_eq(result.tangent, -1.0, 1e-10));
    }

    #[test]
    fn test_differentiate_abs_zero() {
        let result = ForwardAD::differentiate(&[MonoAD::Abs], 0.0);
        assert!(approx_eq(result.value, 0.0, 1e-10));
        assert!(approx_eq(result.tangent, 0.0, 1e-10));
    }

    #[test]
    fn test_compute_value_only() {
        let value = ForwardAD::compute(&mono_ops![sin, exp], 0.5);
        assert!(approx_eq(value, 0.5_f64.sin().exp(), 1e-10));
    }

    #[test]
    fn test_compute_all_ops() {
        // Exercise all MonoAD variants in compute()
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Sin], 1.0),
            1.0_f64.sin(),
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Cos], 1.0),
            1.0_f64.cos(),
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Tan], 1.0),
            1.0_f64.tan(),
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Exp], 1.0),
            1.0_f64.exp(),
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Neg], 1.0),
            -1.0,
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Ln], 2.0),
            2.0_f64.ln(),
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Sqrt], 4.0),
            2.0,
            1e-10
        ));
        assert!(approx_eq(
            ForwardAD::compute(&[MonoAD::Abs], -3.0),
            3.0,
            1e-10
        ));
    }

    #[test]
    fn test_compute_checked_success() {
        let value = ForwardAD::compute_checked(&mono_ops![sin, exp], 0.5).unwrap();
        assert!(approx_eq(value, 0.5_f64.sin().exp(), 1e-10));
    }

    #[test]
    fn test_compute_checked_ln_error() {
        assert!(ForwardAD::compute_checked(&[MonoAD::Ln], -1.0).is_err());
    }

    #[test]
    fn test_compute_checked_sqrt_error() {
        assert!(ForwardAD::compute_checked(&[MonoAD::Sqrt], -1.0).is_err());
    }

    #[test]
    fn test_differentiate_checked_success() {
        let result = ForwardAD::differentiate_checked(&mono_ops![sin, exp], 0.5).unwrap();
        let expected_value = 0.5_f64.sin().exp();
        let expected_tangent = expected_value * 0.5_f64.cos();
        assert!(approx_eq(result.value, expected_value, 1e-10));
        assert!(approx_eq(result.tangent, expected_tangent, 1e-10));
    }

    #[test]
    fn test_differentiate_checked_sqrt_error() {
        assert!(ForwardAD::differentiate_checked(&[MonoAD::Sqrt], -1.0).is_err());
    }

    #[test]
    fn test_directional_derivative_seed_mismatch() {
        let result = ForwardAD::directional_derivative(&[], &[1.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_directional_derivative_no_inputs_no_exprs() {
        let result = ForwardAD::directional_derivative(&[], &[], &[]).unwrap();
        assert_eq!(result.value, 0.0);
        assert_eq!(result.tangent, 0.0);
    }

    #[test]
    fn test_directional_derivative_checked_success() {
        let exprs = vec![
            (MultiAD::Inp, vec![0]),
            (MultiAD::Inp, vec![1]),
            (MultiAD::Mul, vec![0, 1]),
        ];
        let result =
            ForwardAD::directional_derivative_checked(&exprs, &[2.0, 3.0], &[1.0, 0.0]).unwrap();
        assert!(approx_eq(result.value, 6.0, 1e-10));
        assert!(approx_eq(result.tangent, 3.0, 1e-10));
    }

    #[test]
    fn test_directional_derivative_checked_seed_mismatch() {
        let result = ForwardAD::directional_derivative_checked(&[], &[1.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_directional_derivative_checked_no_inputs_no_exprs() {
        let result = ForwardAD::directional_derivative_checked(&[], &[], &[]).unwrap();
        assert_eq!(result.value, 0.0);
        assert_eq!(result.tangent, 0.0);
    }

    #[test]
    fn test_directional_derivative_checked_domain_error() {
        let exprs = vec![(MultiAD::Ln, vec![0])];
        assert!(ForwardAD::directional_derivative_checked(&exprs, &[0.0], &[1.0]).is_err());
    }

    #[test]
    fn test_gradient_graph() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let grad = ForwardAD::gradient_graph(&graph, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(grad[0], 3.0, 1e-10));
        assert!(approx_eq(grad[1], 2.0, 1e-10));
    }

    #[test]
    fn test_jacobian_graphs() {
        let mut g1 = Graph::new(2);
        let x = g1.input(0);
        let y = g1.input(1);
        g1.add(x, y);

        let mut g2 = Graph::new(2);
        let x2 = g2.input(0);
        let y2 = g2.input(1);
        g2.mul(x2, y2);

        let jacobian = ForwardAD::jacobian_graphs(&[g1, g2], &[2.0, 3.0]).unwrap();
        assert_eq!(jacobian.len(), 2);
        assert!(approx_eq(jacobian[0][0], 1.0, 1e-10));
        assert!(approx_eq(jacobian[0][1], 1.0, 1e-10));
        assert!(approx_eq(jacobian[1][0], 3.0, 1e-10));
        assert!(approx_eq(jacobian[1][1], 2.0, 1e-10));
    }

    #[test]
    fn test_directional_derivative_graph_checked_success() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let result =
            ForwardAD::directional_derivative_graph_checked(&graph, &[2.0, 3.0], &[1.0, 0.0])
                .unwrap();
        assert!(approx_eq(result.value, 6.0, 1e-10));
        assert!(approx_eq(result.tangent, 3.0, 1e-10));
    }

    #[test]
    fn test_directional_derivative_graph_input_mismatch() {
        let graph = Graph::new(2);
        assert!(ForwardAD::directional_derivative_graph(&graph, &[1.0], &[1.0]).is_err());
    }

    #[test]
    fn test_directional_derivative_graph_seed_mismatch() {
        let graph = Graph::new(2);
        assert!(ForwardAD::directional_derivative_graph(&graph, &[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_directional_derivative_graph_empty() {
        let graph = Graph::new(0);
        let result = ForwardAD::directional_derivative_graph(&graph, &[], &[]).unwrap();
        assert_eq!(result.value, 0.0);
        assert_eq!(result.tangent, 0.0);
    }
}
