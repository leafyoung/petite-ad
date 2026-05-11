//! Tape-based compiled graph evaluation.

use std::sync::Arc;

use super::graph::{Graph, GraphNode, NodeId};
use super::multi_ad::MultiAD;
use super::op_rules;
use super::types::BackwardResultBox;
use crate::{AutodiffError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompiledArgRange {
    pub(super) start: usize,
    pub(super) len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CompiledNode {
    Constant(f64),
    Operation {
        op: MultiAD,
        arg_range: CompiledArgRange,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BackwardLocal {
    Unary(f64),
    Binary(f64, f64),
}

/// Reusable scratch buffers for repeated [`Tape`] evaluation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TapeWorkspace {
    pub(super) values: Vec<f64>,
    pub(super) cotangent_values: Vec<f64>,
    pub(super) gradients: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tape {
    pub(super) graph: Graph,
    pub(super) output_indices: Vec<NodeId>,
    pub(super) compiled_nodes: Arc<[CompiledNode]>,
    pub(super) arg_indices: Arc<[usize]>,
}

#[inline]
fn compiled_arg_indices(arg_indices_storage: &[usize], arg_range: CompiledArgRange) -> &[usize] {
    &arg_indices_storage[arg_range.start..arg_range.start + arg_range.len]
}

#[inline]
fn with_compiled_arg_values<T, F>(
    arg_indices_storage: &[usize],
    arg_range: CompiledArgRange,
    values: &[f64],
    f: F,
) -> Result<T>
where
    F: FnOnce(&[f64]) -> Result<T>,
{
    let arg_indices = compiled_arg_indices(arg_indices_storage, arg_range);
    match arg_indices.len() {
        0 => f(&[]),
        1 => {
            MultiAD::check_value_index(arg_indices[0], values.len())?;
            let args = [values[arg_indices[0]]];
            f(&args)
        }
        2 => {
            MultiAD::check_value_index(arg_indices[0], values.len())?;
            MultiAD::check_value_index(arg_indices[1], values.len())?;
            let args = [values[arg_indices[0]], values[arg_indices[1]]];
            f(&args)
        }
        _ => {
            let arg_values = MultiAD::gather_arg_values(arg_indices, values)?;
            f(&arg_values)
        }
    }
}

#[inline]
fn backward_local(rule: op_rules::LocalRule) -> BackwardLocal {
    match rule {
        op_rules::LocalRule::Unary { dy, .. } => BackwardLocal::Unary(dy),
        op_rules::LocalRule::Binary {
            dy_left, dy_right, ..
        } => BackwardLocal::Binary(dy_left, dy_right),
    }
}

fn reverse_accumulate_compiled(
    num_inputs: usize,
    compiled_nodes: &[CompiledNode],
    arg_indices_storage: &[usize],
    values: &[f64],
    cotangent_values: &mut [f64],
) -> Result<()> {
    for (offset, node) in compiled_nodes.iter().enumerate().rev() {
        let node_id = num_inputs + offset;
        let current_cotangent = cotangent_values[node_id];
        if current_cotangent == 0.0 {
            continue;
        }

        let CompiledNode::Operation { op, arg_range } = *node else {
            continue;
        };
        let input_indices = compiled_arg_indices(arg_indices_storage, arg_range);
        let value = values[node_id];

        with_compiled_arg_values(arg_indices_storage, arg_range, values, |args| {
            match op_rules::local_rule(op, args, value)? {
                op_rules::LocalRule::Unary { dy, .. } => {
                    cotangent_values[input_indices[0]] += current_cotangent * dy;
                }
                op_rules::LocalRule::Binary {
                    dy_left, dy_right, ..
                } => {
                    cotangent_values[input_indices[0]] += current_cotangent * dy_left;
                    cotangent_values[input_indices[1]] += current_cotangent * dy_right;
                }
            }
            Ok(())
        })?;
    }

    Ok(())
}

impl TapeWorkspace {
    /// Create an empty reusable workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all retained buffers while keeping their capacity.
    pub fn clear(&mut self) {
        self.values.clear();
        self.cotangent_values.clear();
        self.gradients.clear();
    }
}

impl Tape {
    #[inline]
    fn check_input_len(&self, inputs: &[f64]) -> Result<()> {
        if inputs.len() == self.graph.num_inputs {
            Ok(())
        } else {
            Err(AutodiffError::InvalidGraph {
                reason: "input length must match graph.num_inputs()",
            })
        }
    }

    #[inline]
    fn fill_values(&self, inputs: &[f64], workspace: &mut TapeWorkspace) -> Result<()> {
        self.fill_values_inner(inputs, workspace, false)
    }

    #[inline]
    fn fill_values_checked(&self, inputs: &[f64], workspace: &mut TapeWorkspace) -> Result<()> {
        self.fill_values_inner(inputs, workspace, true)
    }

    fn fill_values_inner(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
        checked: bool,
    ) -> Result<()> {
        self.check_input_len(inputs)?;
        workspace.values.clear();
        workspace
            .values
            .reserve(self.graph.num_inputs + self.graph.nodes.len());
        workspace.values.extend_from_slice(inputs);

        for node in self.compiled_nodes.iter() {
            match node {
                CompiledNode::Constant(value) => workspace.values.push(*value),
                CompiledNode::Operation { op, arg_range } => {
                    let value = with_compiled_arg_values(
                        &self.arg_indices,
                        *arg_range,
                        &workspace.values,
                        |args| {
                            if checked {
                                op.forward_checked(args)
                            } else {
                                op.forward(args)
                            }
                        },
                    )?;
                    workspace.values.push(value);
                }
            }
        }

        Ok(())
    }

    /// Return the underlying graph.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Create a reusable workspace sized for this tape.
    #[must_use]
    pub fn workspace(&self) -> TapeWorkspace {
        TapeWorkspace {
            values: Vec::with_capacity(self.graph.num_inputs + self.graph.nodes.len()),
            cotangent_values: Vec::with_capacity(self.graph.num_inputs + self.graph.nodes.len()),
            gradients: Vec::with_capacity(self.graph.num_inputs),
        }
    }

    /// Compute only the primary output value.
    pub fn compute(&self, inputs: &[f64]) -> Result<f64> {
        let mut workspace = self.workspace();
        self.compute_with_workspace(inputs, &mut workspace)
    }

    /// Compute all selected output values.
    pub fn compute_many(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        let mut workspace = self.workspace();
        self.compute_many_with_workspace(inputs, &mut workspace)
    }

    /// Compute all selected output values with checked-domain validation.
    pub fn compute_many_checked(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        let mut workspace = self.workspace();
        self.compute_many_with_workspace_checked(inputs, &mut workspace)
    }

    /// Compute only the primary output value with checked-domain validation.
    pub fn compute_checked(&self, inputs: &[f64]) -> Result<f64> {
        let mut workspace = self.workspace();
        self.compute_with_workspace_checked(inputs, &mut workspace)
    }

    /// Compute only the primary output value using a reusable workspace.
    pub fn compute_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
    ) -> Result<f64> {
        self.fill_values(inputs, workspace)?;
        Ok(self
            .output_indices
            .first()
            .map(|&index| workspace.values[index])
            .unwrap_or(0.0))
    }

    /// Compute all selected output values using a reusable workspace.
    pub fn compute_many_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
    ) -> Result<Vec<f64>> {
        self.fill_values(inputs, workspace)?;
        Ok(self
            .output_indices
            .iter()
            .map(|&index| workspace.values[index])
            .collect())
    }

    /// Compute all selected output values using a reusable workspace with checked-domain validation.
    pub fn compute_many_with_workspace_checked(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
    ) -> Result<Vec<f64>> {
        self.fill_values_checked(inputs, workspace)?;
        Ok(self
            .output_indices
            .iter()
            .map(|&index| workspace.values[index])
            .collect())
    }

    /// Compute only the primary output value using a reusable workspace with checked-domain validation.
    pub fn compute_with_workspace_checked(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
    ) -> Result<f64> {
        self.fill_values_checked(inputs, workspace)?;
        Ok(self
            .output_indices
            .first()
            .map(|&index| workspace.values[index])
            .unwrap_or(0.0))
    }

    /// Compute the output value and gradient closure.
    pub fn compute_grad(&self, inputs: &[f64]) -> Result<BackwardResultBox> {
        self.check_input_len(inputs)?;
        let mut values: Vec<f64> =
            Vec::with_capacity(self.graph.num_inputs + self.graph.nodes.len());
        values.extend_from_slice(inputs);

        let mut backward_locals: Vec<Option<BackwardLocal>> =
            Vec::with_capacity(self.compiled_nodes.len());

        for node in self.compiled_nodes.iter() {
            match node {
                CompiledNode::Constant(value) => {
                    values.push(*value);
                    backward_locals.push(None);
                }
                CompiledNode::Operation { op, arg_range } => {
                    let (value, backward_local) =
                        with_compiled_arg_values(&self.arg_indices, *arg_range, &values, |args| {
                            let value = op.forward(args)?;
                            let local_rule = op_rules::local_rule(*op, args, value)?;
                            Ok((value, backward_local(local_rule)))
                        })?;
                    values.push(value);
                    backward_locals.push(Some(backward_local));
                }
            }
        }

        let final_value = self
            .output_indices
            .first()
            .map(|&index| values[index])
            .unwrap_or(0.0);
        let num_inputs = self.graph.num_inputs;
        let values_len = values.len();
        let output_index = self.output_indices.first().copied();
        let compiled_nodes = Arc::clone(&self.compiled_nodes);
        let arg_indices = Arc::clone(&self.arg_indices);

        let backward_fn = Box::new(move |cotangent: f64| -> Vec<f64> {
            let Some(final_output_index) = output_index else {
                return Vec::new();
            };

            let mut cotangent_values = vec![0.0; values_len];
            cotangent_values[final_output_index] = cotangent;

            for (offset, backward_local) in backward_locals.iter().enumerate().rev() {
                let Some(local) = backward_local else {
                    continue;
                };
                let node_id = num_inputs + offset;
                let current_cotangent = cotangent_values[node_id];
                if current_cotangent == 0.0 {
                    continue;
                }

                let CompiledNode::Operation { arg_range, .. } = compiled_nodes[offset] else {
                    continue;
                };
                let input_indices = compiled_arg_indices(&arg_indices, arg_range);
                match local {
                    BackwardLocal::Unary(dy) => {
                        cotangent_values[input_indices[0]] += current_cotangent * dy;
                    }
                    BackwardLocal::Binary(dy_left, dy_right) => {
                        cotangent_values[input_indices[0]] += current_cotangent * dy_left;
                        cotangent_values[input_indices[1]] += current_cotangent * dy_right;
                    }
                }
            }

            cotangent_values[..num_inputs.min(cotangent_values.len())].to_vec()
        });

        Ok((final_value, backward_fn))
    }

    /// Compute the output value and gradient eagerly using a reusable workspace.
    pub fn gradient_with_workspace<'a>(
        &self,
        inputs: &[f64],
        workspace: &'a mut TapeWorkspace,
    ) -> Result<(f64, &'a [f64])> {
        self.fill_values(inputs, workspace)?;

        workspace.cotangent_values.clear();
        workspace
            .cotangent_values
            .resize(workspace.values.len(), 0.0);
        workspace.gradients.clear();
        workspace.gradients.resize(self.graph.num_inputs, 0.0);

        let Some(output_index) = self.output_indices.first().copied() else {
            return Ok((0.0, &workspace.gradients));
        };

        workspace.cotangent_values[output_index] = 1.0;
        reverse_accumulate_compiled(
            self.graph.num_inputs,
            &self.compiled_nodes,
            &self.arg_indices,
            &workspace.values,
            &mut workspace.cotangent_values,
        )?;

        workspace
            .gradients
            .copy_from_slice(&workspace.cotangent_values[..self.graph.num_inputs]);
        Ok((workspace.values[output_index], &workspace.gradients))
    }

    /// Compute the output value and gradient eagerly using a reusable workspace with checked-domain validation.
    pub fn gradient_with_workspace_checked<'a>(
        &self,
        inputs: &[f64],
        workspace: &'a mut TapeWorkspace,
    ) -> Result<(f64, &'a [f64])> {
        self.fill_values_checked(inputs, workspace)?;

        workspace.cotangent_values.clear();
        workspace
            .cotangent_values
            .resize(workspace.values.len(), 0.0);
        workspace.gradients.clear();
        workspace.gradients.resize(self.graph.num_inputs, 0.0);

        let Some(output_index) = self.output_indices.first().copied() else {
            return Ok((0.0, &workspace.gradients));
        };

        workspace.cotangent_values[output_index] = 1.0;
        reverse_accumulate_compiled(
            self.graph.num_inputs,
            &self.compiled_nodes,
            &self.arg_indices,
            &workspace.values,
            &mut workspace.cotangent_values,
        )?;

        workspace
            .gradients
            .copy_from_slice(&workspace.cotangent_values[..self.graph.num_inputs]);
        Ok((workspace.values[output_index], &workspace.gradients))
    }

    /// Compute the output value and gradient eagerly.
    pub fn gradient(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let mut workspace = self.workspace();
        let (value, gradient) = self.gradient_with_workspace(inputs, &mut workspace)?;
        Ok((value, gradient.to_vec()))
    }

    /// Compute the output value and gradient eagerly with checked-domain validation.
    pub fn gradient_checked(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        let mut workspace = self.workspace();
        let (value, gradient) = self.gradient_with_workspace_checked(inputs, &mut workspace)?;
        Ok((value, gradient.to_vec()))
    }

    /// Compute the Jacobian for all selected outputs.
    pub fn jacobian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut workspace = self.workspace();
        self.jacobian_with_workspace(inputs, &mut workspace)
    }

    /// Compute the Jacobian for all selected outputs with checked-domain validation.
    pub fn jacobian_checked(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut workspace = self.workspace();
        self.jacobian_with_workspace_checked(inputs, &mut workspace)
    }

    /// Compute the Jacobian for all selected outputs using a reusable workspace.
    pub fn jacobian_with_workspace(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
    ) -> Result<Vec<Vec<f64>>> {
        self.jacobian_with_workspace_inner(inputs, workspace, false)
    }

    /// Compute the Jacobian for all selected outputs using a reusable workspace with checked-domain validation.
    pub fn jacobian_with_workspace_checked(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
    ) -> Result<Vec<Vec<f64>>> {
        self.jacobian_with_workspace_inner(inputs, workspace, true)
    }

    fn jacobian_with_workspace_inner(
        &self,
        inputs: &[f64],
        workspace: &mut TapeWorkspace,
        checked: bool,
    ) -> Result<Vec<Vec<f64>>> {
        if checked {
            self.fill_values_checked(inputs, workspace)?;
        } else {
            self.fill_values(inputs, workspace)?;
        }
        let mut jacobian: Vec<Vec<f64>> = Vec::with_capacity(self.output_indices.len());

        for &output_index in &self.output_indices {
            workspace.cotangent_values.clear();
            workspace
                .cotangent_values
                .resize(workspace.values.len(), 0.0);
            workspace.gradients.clear();
            workspace.gradients.resize(self.graph.num_inputs, 0.0);
            workspace.cotangent_values[output_index] = 1.0;
            reverse_accumulate_compiled(
                self.graph.num_inputs,
                &self.compiled_nodes,
                &self.arg_indices,
                &workspace.values,
                &mut workspace.cotangent_values,
            )?;

            workspace
                .gradients
                .copy_from_slice(&workspace.cotangent_values[..self.graph.num_inputs]);
            jacobian.push(workspace.gradients.clone());
        }

        Ok(jacobian)
    }

    /// Compute a finite-difference Hessian by differentiating eager gradients.
    pub fn compute_hessian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let num_inputs = self.graph.num_inputs;
        let epsilon = 1e-5;
        let mut hessian = vec![vec![0.0; num_inputs]; num_inputs];

        if num_inputs == 0 {
            let (_value, _grad) = self.gradient(inputs)?;
            return Ok(hessian);
        }

        let mut plus_workspace = self.workspace();
        let mut minus_workspace = self.workspace();

        for j in 0..num_inputs {
            let mut inputs_plus = inputs.to_vec();
            inputs_plus[j] += epsilon;

            let mut inputs_minus = inputs.to_vec();
            inputs_minus[j] -= epsilon;

            let (_value_plus, grad_plus) =
                self.gradient_with_workspace(&inputs_plus, &mut plus_workspace)?;
            let (_value_minus, grad_minus) =
                self.gradient_with_workspace(&inputs_minus, &mut minus_workspace)?;

            for i in 0..num_inputs {
                hessian[i][j] = (grad_plus[i] - grad_minus[i]) / (2.0 * epsilon);
            }
        }

        Ok(hessian)
    }
}
