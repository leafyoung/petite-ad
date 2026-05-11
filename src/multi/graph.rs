//! Graph and tape APIs for reusable multi-variable autodiff.
//!
//! Unlike the legacy tuple-based representation, this API supports literal
//! constants and returns node handles as you build the graph.
//!
//! # Examples
//!
//! ```
//! use petite_ad::Graph;
//!
//! let mut graph = Graph::new(2);
//! let x = graph.input(0);
//! let y = graph.input(1);
//! let two = graph.constant(2.0);
//! let xy = graph.mul(x, y);
//! let out = graph.add(xy, two);
//!
//! assert_eq!(out, 4);
//!
//! let value = graph.compute(&[3.0, 4.0]).unwrap();
//! assert!((value - 14.0).abs() < 1e-10);
//!
//! // Explicitly choose a different output when needed.
//! graph.set_output(xy).unwrap();
//! let product_only = graph.compute(&[3.0, 4.0]).unwrap();
//! assert!((product_only - 12.0).abs() < 1e-10);
//!
//! // Or designate multiple outputs and get all values / a Jacobian.
//! graph.set_outputs(&[xy, out]).unwrap();
//! let values = graph.compute_many(&[3.0, 4.0]).unwrap();
//! assert_eq!(values.len(), 2);
//! let jacobian = graph.jacobian(&[3.0, 4.0]).unwrap();
//! assert_eq!(jacobian.len(), 2);
//! ```

use std::{
    cell::RefCell,
    fmt::Write,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
    sync::Arc,
};

use super::compiled::{
    BackendKind, BackendSupportReport, BatchGradients, BatchGradientsBuffer, BatchInputs,
    BatchValues, BatchValuesBuffer, CompiledGraph, CompiledGraphMetadata, CompiledWorkspace,
    DeviceBatchPlan, DeviceBufferSet, DeviceExecutionTrace, Instruction,
};
use super::multi_ad::MultiAD;
use super::multi_ad_fr::MultiAD2FR;
use super::multi_ad_rf::MultiAD2RF;
use super::op_rules;
use super::parser;
use super::types::BackwardResultBox;
use crate::{AutodiffError, Result};

/// Reusable scratch buffers for repeated [`Tape`] evaluation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TapeWorkspace {
    values: Vec<f64>,
    cotangent_values: Vec<f64>,
    gradients: Vec<f64>,
}

/// A handle to an input or computed node in a graph.
pub type NodeId = usize;

/// A node in a reusable computation graph.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum GraphNode {
    /// A literal scalar value stored directly in the graph.
    Constant(f64),
    /// An operation node whose arguments reference prior node ids or inputs.
    Operation { op: MultiAD, inputs: Vec<NodeId> },
}

/// A reusable multi-variable computation graph.
///
/// Inputs occupy node ids `0..num_inputs`. Stored nodes are appended after the
/// inputs, so the first computed constant or operation gets id `num_inputs`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    num_inputs: usize,
    nodes: Vec<GraphNode>,
    output_nodes: Vec<NodeId>,
    input_names: Vec<Option<String>>,
    output_names: Vec<(NodeId, String)>,
    #[cfg_attr(feature = "serde", serde(default))]
    parameters: Vec<NodeId>,
    #[cfg_attr(feature = "serde", serde(default))]
    parameter_names: Vec<(NodeId, String)>,
}

/// A reusable compiled view of a [`Graph`].
///
/// Today this stores the graph structure directly and provides a stable place
/// to grow future precomputation or buffer reuse without changing the public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompiledArgRange {
    start: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CompiledNode {
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

#[derive(Debug, Clone, PartialEq)]
enum ExactLocal {
    None,
    Unary {
        parent: NodeId,
        dy: f64,
        ddy: f64,
    },
    Binary {
        left: NodeId,
        right: NodeId,
        dy_left: f64,
        dy_right: f64,
        ddy_left_left: f64,
        ddy_right_right: f64,
        ddy_left_right: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tape {
    graph: Graph,
    output_indices: Vec<NodeId>,
    compiled_nodes: Arc<[CompiledNode]>,
    arg_indices: Arc<[usize]>,
}

/// One component comparison from [`Graph::check_gradient`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GradientCheckEntry {
    /// Input coordinate being checked.
    pub index: usize,
    /// Reverse-mode gradient value.
    pub autodiff: f64,
    /// Central finite-difference gradient value.
    pub finite_difference: f64,
    /// Absolute difference between the two methods.
    pub abs_error: f64,
}

/// Deterministic report returned by [`Graph::check_gradient`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GradientCheckReport {
    /// Whether every component is within tolerance.
    pub passed: bool,
    /// Requested absolute tolerance.
    pub tolerance: f64,
    /// Largest absolute error across all entries.
    pub max_abs_error: f64,
    /// Per-coordinate comparisons.
    pub entries: Vec<GradientCheckEntry>,
}

/// Domain validation policy for graph evaluation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainPolicy {
    /// Preserve raw `f64` behavior.
    Unchecked,
    /// Reject invalid primal values such as `ln(x <= 0)`.
    Checked,
    /// Reject invalid primal values and derivative singularities such as `sqrt(0)`.
    StrictDerivative,
}

/// Basic graph statistics for diagnostics and benchmark setup.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphStats {
    pub num_inputs: usize,
    pub num_constants: usize,
    pub num_ops: usize,
    pub num_edges: usize,
    pub max_depth: usize,
    pub op_counts: Vec<(MultiAD, usize)>,
}

/// Shared expression graph used for operator-overloaded graph construction.
#[derive(Debug, Clone)]
pub struct ExprGraph {
    graph: Rc<RefCell<Graph>>,
}

/// A node handle tied to an [`ExprGraph`].
#[derive(Debug, Clone)]
pub struct ExprNode {
    graph: Rc<RefCell<Graph>>,
    node: NodeId,
}

#[inline]
fn op_name(op: MultiAD) -> &'static str {
    op_rules::op_name(op)
}

#[inline]
fn expected_arity(op: MultiAD) -> usize {
    op_rules::expected_arity(op)
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

impl Graph {
    #[inline]
    fn output_name_for(&self, node_id: NodeId) -> Option<&str> {
        self.output_names
            .iter()
            .find_map(|(id, name)| (*id == node_id).then_some(name.as_str()))
    }

    #[inline]
    fn node_label(&self, node_id: NodeId, node: &GraphNode) -> String {
        let base = match node {
            GraphNode::Constant(value) => format!("Const({value})"),
            GraphNode::Operation { op, .. } => format!("{op:?}"),
        };
        match self.output_name_for(node_id) {
            Some(name) => format!("{name}: {base}"),
            None => base,
        }
    }

    /// Create an empty graph with the given number of inputs.
    #[must_use]
    pub fn new(num_inputs: usize) -> Self {
        Self {
            num_inputs,
            nodes: Vec::new(),
            output_nodes: Vec::new(),
            input_names: vec![None; num_inputs],
            output_names: Vec::new(),
            parameters: Vec::new(),
            parameter_names: Vec::new(),
        }
    }

    /// Convert a legacy tuple graph into a reusable [`Graph`].
    ///
    /// Input marker nodes are implicit in this API. Leading `Inp` markers are
    /// skipped, while a trailing `Inp(k)` marker is preserved as an explicit
    /// output selection for input `k`.
    pub fn from_operations(num_inputs: usize, ops: &[(MultiAD, Vec<usize>)]) -> Self {
        Self::try_from_operations(num_inputs, ops).unwrap_or_else(|_| {
            let mut graph = Self::new(num_inputs);
            for (op, inputs) in ops {
                if *op != MultiAD::Inp {
                    graph.push_operation(*op, inputs.clone());
                }
            }
            graph
        })
    }

    /// Convert a legacy tuple graph into a reusable [`Graph`] with validation.
    pub fn try_from_operations(num_inputs: usize, ops: &[(MultiAD, Vec<usize>)]) -> Result<Self> {
        let mut graph = Self::new(num_inputs);
        let mut pending_input_output: Option<NodeId> = None;

        for (op, inputs) in ops {
            if *op == MultiAD::Inp {
                AutodiffError::check_arity("Inp", 1, inputs.len())?;
                if inputs[0] >= num_inputs {
                    return Err(AutodiffError::IndexOutOfBounds {
                        index: inputs[0],
                        max_index: num_inputs.saturating_sub(1),
                    });
                }
                pending_input_output = Some(inputs[0]);
                continue;
            }

            pending_input_output = None;
            graph.try_push_operation(*op, inputs.clone())?;
        }

        if let Some(output) = pending_input_output {
            graph.set_output(output)?;
        }

        Ok(graph)
    }

    /// Parse a small expression into a graph.
    ///
    /// Supported syntax includes named inputs, numeric constants, `+ - * / ^`,
    /// parentheses, unary minus, and common unary functions such as `sin`,
    /// `exp`, `sqrt`, `tanh`, `relu`, `sigmoid`, `softplus`, `log1p_exp`, and `gelu`.
    pub fn parse_expression(expression: &str, input_names: &[&str]) -> Result<Self> {
        parser::parse_expression(expression, input_names)
    }

    /// Return the number of graph inputs.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Return the stored non-input nodes.
    #[must_use]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Return whether the graph has no stored nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return the number of stored non-input nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return the next allocated node id.
    #[must_use]
    pub fn next_node_id(&self) -> NodeId {
        self.num_inputs + self.nodes.len()
    }

    /// Return the explicitly selected primary output node, if one is configured.
    #[must_use]
    pub fn output_node(&self) -> Option<NodeId> {
        self.output_nodes.first().copied()
    }

    /// Return all explicitly selected output nodes.
    #[must_use]
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }

    /// Return the effective primary output node used for single-output evaluation.
    ///
    /// If no explicit outputs have been selected, this falls back to the most
    /// recently created node, or the last input for graphs without stored nodes.
    #[must_use]
    pub fn effective_output_node(&self) -> Option<NodeId> {
        self.output_node()
            .or_else(|| self.next_node_id().checked_sub(1))
    }

    /// Return the effective output nodes used for multi-output evaluation.
    #[must_use]
    pub fn effective_output_nodes(&self) -> Vec<NodeId> {
        if self.output_nodes.is_empty() {
            self.effective_output_node().into_iter().collect()
        } else {
            self.output_nodes.clone()
        }
    }

    /// Set a single output node used for evaluation.
    pub fn set_output(&mut self, output: NodeId) -> Result<&mut Self> {
        self.set_outputs(&[output])
    }

    /// Set multiple output nodes used for vector-output evaluation.
    pub fn set_outputs(&mut self, outputs: &[NodeId]) -> Result<&mut Self> {
        let max_index =
            self.next_node_id()
                .checked_sub(1)
                .ok_or(AutodiffError::IndexOutOfBounds {
                    index: outputs.first().copied().unwrap_or(0),
                    max_index: 0,
                })?;
        for &output in outputs {
            if output > max_index {
                return Err(AutodiffError::IndexOutOfBounds {
                    index: output,
                    max_index,
                });
            }
        }
        self.output_nodes = outputs.to_vec();
        Ok(self)
    }

    /// Add one more output node to the vector-output selection.
    pub fn add_output(&mut self, output: NodeId) -> Result<&mut Self> {
        let max_index =
            self.next_node_id()
                .checked_sub(1)
                .ok_or(AutodiffError::IndexOutOfBounds {
                    index: output,
                    max_index: 0,
                })?;
        if output > max_index {
            return Err(AutodiffError::IndexOutOfBounds {
                index: output,
                max_index,
            });
        }
        self.output_nodes.push(output);
        Ok(self)
    }

    /// Clear any explicit output nodes, restoring implicit last-node behavior.
    pub fn clear_output(&mut self) -> &mut Self {
        self.output_nodes.clear();
        self
    }

    /// Get an input node id.
    #[must_use]
    pub fn input(&self, input_index: usize) -> NodeId {
        input_index
    }

    /// Append a literal constant node and return its node id.
    pub fn constant(&mut self, value: f64) -> NodeId {
        let node_id = self.next_node_id();
        self.nodes.push(GraphNode::Constant(value));
        node_id
    }

    /// Append a custom operation node and return its node id.
    ///
    /// This is a low-level unchecked API. Prefer typed helpers such as
    /// [`Graph::sin`] and [`Graph::add`] when possible. Passing `MultiAD::Inp`
    /// creates an invalid reusable graph because input nodes are implicit in
    /// this API and should be obtained with [`Graph::input`].
    pub fn push_operation(&mut self, op: MultiAD, inputs: Vec<NodeId>) -> NodeId {
        let node_id = self.next_node_id();
        self.nodes.push(GraphNode::Operation { op, inputs });
        node_id
    }

    /// Append a custom operation after validating arity and input references.
    pub fn try_push_operation(&mut self, op: MultiAD, inputs: Vec<NodeId>) -> Result<NodeId> {
        if op == MultiAD::Inp {
            return Err(AutodiffError::InvalidGraph {
                reason: "Graph nodes must not contain input markers",
            });
        }
        AutodiffError::check_arity(op_name(op), expected_arity(op), inputs.len())?;
        let max_index = self.next_node_id().saturating_sub(1);
        for &input in &inputs {
            if input >= self.next_node_id() {
                return Err(AutodiffError::IndexOutOfBounds {
                    index: input,
                    max_index,
                });
            }
        }
        Ok(self.push_operation(op, inputs))
    }

    pub fn try_sin(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Sin, vec![arg])
    }

    pub fn try_cos(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Cos, vec![arg])
    }

    pub fn try_tan(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Tan, vec![arg])
    }

    pub fn try_tanh(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Tanh, vec![arg])
    }

    pub fn try_relu(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Relu, vec![arg])
    }

    pub fn try_log1p_exp(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Log1pExp, vec![arg])
    }

    pub fn try_neg(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Neg, vec![arg])
    }

    pub fn try_exp(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Exp, vec![arg])
    }

    pub fn try_ln(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Ln, vec![arg])
    }

    pub fn try_sqrt(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Sqrt, vec![arg])
    }

    pub fn try_abs(&mut self, arg: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Abs, vec![arg])
    }

    pub fn try_add(&mut self, left: NodeId, right: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Add, vec![left, right])
    }

    pub fn try_sub(&mut self, left: NodeId, right: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Sub, vec![left, right])
    }

    pub fn try_mul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Mul, vec![left, right])
    }

    pub fn try_div(&mut self, left: NodeId, right: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Div, vec![left, right])
    }

    pub fn try_pow(&mut self, base: NodeId, exp: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::Pow, vec![base, exp])
    }

    pub fn try_log_add_exp(&mut self, left: NodeId, right: NodeId) -> Result<NodeId> {
        self.try_push_operation(MultiAD::LogAddExp, vec![left, right])
    }

    /// Append a sine node.
    pub fn sin(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Sin, vec![arg])
    }

    /// Append a cosine node.
    pub fn cos(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Cos, vec![arg])
    }

    /// Append a tangent node.
    pub fn tan(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Tan, vec![arg])
    }

    /// Append a hyperbolic tangent node.
    pub fn tanh(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Tanh, vec![arg])
    }

    /// Append a rectified-linear-unit node.
    pub fn relu(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Relu, vec![arg])
    }

    /// Append a stable `ln(1 + exp(arg))` node.
    pub fn log1p_exp_node(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Log1pExp, vec![arg])
    }

    /// Append a negation node.
    pub fn neg(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Neg, vec![arg])
    }

    /// Append an exponential node.
    pub fn exp(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Exp, vec![arg])
    }

    /// Append a natural logarithm node.
    pub fn ln(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Ln, vec![arg])
    }

    /// Append a square-root node.
    pub fn sqrt(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Sqrt, vec![arg])
    }

    /// Append an absolute-value node.
    pub fn abs(&mut self, arg: NodeId) -> NodeId {
        self.push_operation(MultiAD::Abs, vec![arg])
    }

    /// Append an addition node.
    pub fn add(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.push_operation(MultiAD::Add, vec![left, right])
    }

    /// Append a subtraction node.
    pub fn sub(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.push_operation(MultiAD::Sub, vec![left, right])
    }

    /// Append a multiplication node.
    pub fn mul(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.push_operation(MultiAD::Mul, vec![left, right])
    }

    /// Append a division node.
    pub fn div(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.push_operation(MultiAD::Div, vec![left, right])
    }

    /// Append a power node.
    pub fn pow(&mut self, base: NodeId, exp: NodeId) -> NodeId {
        self.push_operation(MultiAD::Pow, vec![base, exp])
    }

    /// Append a stable binary `ln(exp(left) + exp(right))` node.
    pub fn log_add_exp(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.push_operation(MultiAD::LogAddExp, vec![left, right])
    }

    /// Append `node + constant`.
    pub fn add_const(&mut self, node: NodeId, value: f64) -> NodeId {
        let constant = self.constant(value);
        self.add(node, constant)
    }

    /// Append `node - constant`.
    pub fn sub_const(&mut self, node: NodeId, value: f64) -> NodeId {
        let constant = self.constant(value);
        self.sub(node, constant)
    }

    /// Append `node * constant`.
    pub fn mul_const(&mut self, node: NodeId, value: f64) -> NodeId {
        let constant = self.constant(value);
        self.mul(node, constant)
    }

    /// Append `node / constant`.
    pub fn div_const(&mut self, node: NodeId, value: f64) -> NodeId {
        let constant = self.constant(value);
        self.div(node, constant)
    }

    /// Append `node.powf(constant)`.
    pub fn pow_const(&mut self, node: NodeId, value: f64) -> NodeId {
        let constant = self.constant(value);
        self.pow(node, constant)
    }

    /// Append `node * node`.
    pub fn square(&mut self, node: NodeId) -> NodeId {
        self.mul(node, node)
    }

    /// Append `node * node * node`.
    pub fn cube(&mut self, node: NodeId) -> NodeId {
        let square = self.square(node);
        self.mul(square, node)
    }

    /// Append `1 / node`.
    pub fn reciprocal(&mut self, node: NodeId) -> NodeId {
        let one = self.constant(1.0);
        self.div(one, node)
    }

    /// Append the logistic sigmoid `1 / (1 + exp(-node))`.
    pub fn sigmoid(&mut self, node: NodeId) -> NodeId {
        let neg = self.neg(node);
        let exp_neg = self.exp(neg);
        let denom = self.add_const(exp_neg, 1.0);
        self.reciprocal(denom)
    }

    /// Append stable `ln(1 + exp(node))`.
    pub fn softplus(&mut self, node: NodeId) -> NodeId {
        self.log1p_exp(node)
    }

    /// Append the tanh-based GELU approximation.
    pub fn gelu(&mut self, node: NodeId) -> NodeId {
        let x_cubed = self.cube(node);
        let inner_cubic = self.mul_const(x_cubed, 0.044_715);
        let inner_sum = self.add(node, inner_cubic);
        let scaled_inner = self.mul_const(inner_sum, (2.0 / std::f64::consts::PI).sqrt());
        let tanh = self.tanh(scaled_inner);
        let one_plus = self.add_const(tanh, 1.0);
        let half_x = self.mul_const(node, 0.5);
        self.mul(half_x, one_plus)
    }

    /// Append softmax nodes for a vector of logits.
    pub fn softmax(&mut self, logits: &[NodeId]) -> Vec<NodeId> {
        let exp_nodes: Vec<NodeId> = logits.iter().map(|&node| self.exp(node)).collect();
        let Some((&first, rest)) = exp_nodes.split_first() else {
            return Vec::new();
        };
        let mut denom = first;
        for &node in rest {
            denom = self.add(denom, node);
        }
        exp_nodes
            .into_iter()
            .map(|node| self.div(node, denom))
            .collect()
    }

    /// Sum a slice of nodes.
    pub fn sum(&mut self, nodes: &[NodeId]) -> Option<NodeId> {
        let (&first, rest) = nodes.split_first()?;
        let mut acc = first;
        for &node in rest {
            acc = self.add(acc, node);
        }
        Some(acc)
    }

    /// Mean of a slice of nodes.
    pub fn mean(&mut self, nodes: &[NodeId]) -> Option<NodeId> {
        let sum = self.sum(nodes)?;
        Some(self.div_const(sum, nodes.len() as f64))
    }

    /// Dot product of two equal-length node slices.
    pub fn dot(&mut self, left: &[NodeId], right: &[NodeId]) -> Result<NodeId> {
        if left.len() != right.len() || left.is_empty() {
            return Err(AutodiffError::InvalidGraph {
                reason: "dot inputs must be non-empty and have equal lengths",
            });
        }
        let products: Vec<NodeId> = left
            .iter()
            .zip(right.iter())
            .map(|(&lhs, &rhs)| self.mul(lhs, rhs))
            .collect();
        self.sum(&products).ok_or(AutodiffError::InvalidGraph {
            reason: "dot product could not be reduced",
        })
    }

    /// Sum of squares of a slice of nodes.
    pub fn sum_squares(&mut self, nodes: &[NodeId]) -> Option<NodeId> {
        let squares: Vec<NodeId> = nodes.iter().map(|&node| self.square(node)).collect();
        self.sum(&squares)
    }

    /// Euclidean norm approximation `sqrt(sum_squares(nodes))`.
    pub fn norm2(&mut self, nodes: &[NodeId]) -> Option<NodeId> {
        let sum_squares = self.sum_squares(nodes)?;
        Some(self.sqrt(sum_squares))
    }

    /// Stable sigmoid helper using `0.5 * (tanh(0.5 * x) + 1)`.
    pub fn sigmoid_stable(&mut self, node: NodeId) -> NodeId {
        let half = self.mul_const(node, 0.5);
        let tanh = self.tanh(half);
        let shifted = self.add_const(tanh, 1.0);
        self.mul_const(shifted, 0.5)
    }

    /// Stable `log(1 + exp(x))` helper.
    pub fn log1p_exp(&mut self, node: NodeId) -> NodeId {
        self.log1p_exp_node(node)
    }

    /// Overflow-safe pairwise log-sum-exp approximation.
    pub fn logsumexp_approx(&mut self, nodes: &[NodeId]) -> Option<NodeId> {
        let (&first, rest) = nodes.split_first()?;
        let mut acc = first;
        for &node in rest {
            acc = self.log_add_exp(acc, node);
        }
        Some(acc)
    }

    /// Overflow-safe softmax approximation using pairwise log-sum-exp.
    pub fn stable_softmax_approx(&mut self, logits: &[NodeId]) -> Vec<NodeId> {
        let Some(log_denom) = self.logsumexp_approx(logits) else {
            return Vec::new();
        };
        logits
            .iter()
            .map(|&logit| {
                let centered = self.sub(logit, log_denom);
                self.exp(centered)
            })
            .collect()
    }

    /// Mean squared error loss.
    pub fn mse_loss(&mut self, predictions: &[NodeId], targets: &[NodeId]) -> Result<NodeId> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Err(AutodiffError::InvalidGraph {
                reason: "mse_loss inputs must be non-empty and have equal lengths",
            });
        }
        let squares: Vec<NodeId> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&prediction, &target)| {
                let diff = self.sub(prediction, target);
                self.square(diff)
            })
            .collect();
        self.mean(&squares).ok_or(AutodiffError::InvalidGraph {
            reason: "mse_loss could not be reduced",
        })
    }

    /// Binary cross-entropy loss for probability predictions.
    pub fn binary_cross_entropy_loss(
        &mut self,
        probabilities: &[NodeId],
        targets: &[NodeId],
    ) -> Result<NodeId> {
        if probabilities.len() != targets.len() || probabilities.is_empty() {
            return Err(AutodiffError::InvalidGraph {
                reason: "binary_cross_entropy_loss inputs must be non-empty and have equal lengths",
            });
        }
        let mut terms = Vec::with_capacity(probabilities.len());
        for (&probability, &target) in probabilities.iter().zip(targets.iter()) {
            let log_p = self.ln(probability);
            let one_minus_target = self.sub_const(target, 1.0);
            let neg_one_minus_target = self.neg(one_minus_target);
            let one_minus_p = self.sub_const(probability, 1.0);
            let neg_one_minus_p = self.neg(one_minus_p);
            let log_one_minus_p = self.ln(neg_one_minus_p);
            let positive_term = self.mul(target, log_p);
            let negative_term = self.mul(neg_one_minus_target, log_one_minus_p);
            let sum = self.add(positive_term, negative_term);
            terms.push(self.neg(sum));
        }
        self.mean(&terms).ok_or(AutodiffError::InvalidGraph {
            reason: "binary_cross_entropy_loss could not be reduced",
        })
    }

    /// Convert this reusable graph into the legacy tuple-based operation form.
    ///
    /// This succeeds only when the selected output and all nodes needed to reach it
    /// can be represented without inline constants.
    pub fn to_operations(&self) -> Result<Vec<(MultiAD, Vec<usize>)>> {
        let effective_output = self.effective_output_node();
        if self.output_nodes.len() > 1 {
            return Err(AutodiffError::InvalidGraph {
                reason: "legacy tuple graphs support only one output",
            });
        }
        let last_included_node =
            effective_output.and_then(|node| node.checked_sub(self.num_inputs));
        let node_limit = last_included_node.map(|offset| offset + 1).unwrap_or(0);

        let mut ops: Vec<(MultiAD, Vec<usize>)> = Vec::with_capacity(node_limit);
        for node in self.nodes.iter().take(node_limit) {
            match node {
                GraphNode::Constant(_) => {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "legacy tuple graphs cannot represent constant nodes",
                    });
                }
                GraphNode::Operation { op, inputs } => ops.push((*op, inputs.clone())),
            }
        }

        if let Some(output) = effective_output {
            if output < self.num_inputs {
                ops.push((MultiAD::Inp, vec![output]));
            }
        }

        Ok(ops)
    }

    /// Validate graph structure and references.
    pub fn validate(&self) -> Result<()> {
        let mut next_valid_id = self.num_inputs;

        for node in &self.nodes {
            match node {
                GraphNode::Constant(_) => {
                    next_valid_id += 1;
                }
                GraphNode::Operation { op, inputs } => {
                    if *op == MultiAD::Inp {
                        return Err(AutodiffError::InvalidGraph {
                            reason: "Graph nodes must not contain input markers",
                        });
                    }
                    AutodiffError::check_arity(op_name(*op), expected_arity(*op), inputs.len())?;
                    for &input in inputs {
                        if input >= next_valid_id {
                            return Err(AutodiffError::IndexOutOfBounds {
                                index: input,
                                max_index: next_valid_id.saturating_sub(1),
                            });
                        }
                    }
                    next_valid_id += 1;
                }
            }
        }

        let max_index = next_valid_id.saturating_sub(1);
        for &output in &self.output_nodes {
            if output >= next_valid_id {
                return Err(AutodiffError::IndexOutOfBounds {
                    index: output,
                    max_index,
                });
            }
        }
        for &parameter in &self.parameters {
            if parameter >= next_valid_id {
                return Err(AutodiffError::IndexOutOfBounds {
                    index: parameter,
                    max_index,
                });
            }
        }

        Ok(())
    }

    /// Export the graph as Mermaid flowchart syntax.
    #[must_use]
    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("flowchart LR\n");
        for input_idx in 0..self.num_inputs {
            let label = self
                .input_name(input_idx)
                .map(|name| format!("{name}: Input {input_idx}"))
                .unwrap_or_else(|| format!("Input {input_idx}"));
            let _ = writeln!(&mut out, "    n{input_idx}[\"{label}\"]");
        }
        for (offset, node) in self.nodes.iter().enumerate() {
            let node_id = self.num_inputs + offset;
            let label = if self.output_nodes.contains(&node_id) {
                format!("{} [output]", self.node_label(node_id, node))
            } else {
                self.node_label(node_id, node)
            };
            let _ = writeln!(&mut out, "    n{node_id}[\"{label}\"]");
            if let GraphNode::Operation { inputs, .. } = node {
                for &input in inputs {
                    let _ = writeln!(&mut out, "    n{input} --> n{node_id}");
                }
            }
        }
        out
    }

    /// Export the graph as Graphviz DOT syntax.
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph Graph {\n    rankdir=LR;\n");
        for input_idx in 0..self.num_inputs {
            let label = self
                .input_name(input_idx)
                .map(|name| format!("{name}: Input {input_idx}"))
                .unwrap_or_else(|| format!("Input {input_idx}"));
            let _ = writeln!(&mut out, "    n{input_idx} [label=\"{label}\", shape=box];");
        }
        for (offset, node) in self.nodes.iter().enumerate() {
            let node_id = self.num_inputs + offset;
            let shape = if self.output_nodes.contains(&node_id) {
                "doublecircle"
            } else {
                match node {
                    GraphNode::Constant(_) => "ellipse",
                    GraphNode::Operation { .. } => "oval",
                }
            };
            let label = if self.output_nodes.contains(&node_id) {
                format!("{} [output]", self.node_label(node_id, node))
            } else {
                self.node_label(node_id, node)
            };
            let _ = writeln!(
                &mut out,
                "    n{node_id} [label=\"{label}\", shape={shape}];"
            );
            if let GraphNode::Operation { inputs, .. } = node {
                for &input in inputs {
                    let _ = writeln!(&mut out, "    n{input} -> n{node_id};");
                }
            }
        }
        out.push_str("}\n");
        out
    }

    /// Set a human-readable input name used by graph exporters.
    pub fn set_input_name(
        &mut self,
        input_index: usize,
        name: impl Into<String>,
    ) -> Result<&mut Self> {
        if input_index >= self.num_inputs {
            return Err(AutodiffError::IndexOutOfBounds {
                index: input_index,
                max_index: self.num_inputs.saturating_sub(1),
            });
        }
        self.input_names[input_index] = Some(name.into());
        Ok(self)
    }

    /// Set a human-readable output name used by graph exporters.
    pub fn set_output_name(
        &mut self,
        output: NodeId,
        name: impl Into<String>,
    ) -> Result<&mut Self> {
        self.set_output(output)?;
        self.output_names.retain(|(node_id, _)| *node_id != output);
        self.output_names.push((output, name.into()));
        Ok(self)
    }

    /// Return the configured input name, if present.
    #[must_use]
    pub fn input_name(&self, input_index: usize) -> Option<&str> {
        self.input_names
            .get(input_index)
            .and_then(|name| name.as_deref())
    }

    /// Mark a node as a parameter for optimizer-oriented workflows.
    pub fn mark_parameter(&mut self, node: NodeId) -> Result<&mut Self> {
        let next_node_id = self.next_node_id();
        let max_index = next_node_id.saturating_sub(1);
        if next_node_id == 0 || node >= next_node_id {
            return Err(AutodiffError::IndexOutOfBounds {
                index: node,
                max_index,
            });
        }
        if !self.parameters.contains(&node) {
            self.parameters.push(node);
        }
        Ok(self)
    }

    /// Set a human-readable parameter name.
    pub fn set_parameter_name(
        &mut self,
        node: NodeId,
        name: impl Into<String>,
    ) -> Result<&mut Self> {
        self.mark_parameter(node)?;
        self.parameter_names.retain(|(id, _)| *id != node);
        self.parameter_names.push((node, name.into()));
        Ok(self)
    }

    /// Return parameter nodes.
    #[must_use]
    pub fn parameters(&self) -> &[NodeId] {
        &self.parameters
    }

    /// Return the configured parameter name, if present.
    #[must_use]
    pub fn parameter_name(&self, node: NodeId) -> Option<&str> {
        self.parameter_names
            .iter()
            .find(|(id, _)| *id == node)
            .map(|(_, name)| name.as_str())
    }

    /// Return all configured parameter names.
    #[must_use]
    pub fn parameter_names(&self) -> &[(NodeId, String)] {
        &self.parameter_names
    }

    /// Extract gradients for parameter nodes that are graph inputs.
    pub fn parameter_gradient(&self, inputs: &[f64]) -> Result<Vec<(NodeId, f64)>> {
        let (_value, gradient) = self.gradient(inputs)?;
        let mut result = Vec::with_capacity(self.parameters.len());
        for &parameter in &self.parameters {
            if parameter >= self.num_inputs {
                return Err(AutodiffError::InvalidGraph {
                    reason: "parameter_gradient currently supports input parameters only",
                });
            }
            result.push((parameter, gradient[parameter]));
        }
        Ok(result)
    }

    /// Return diagnostic graph statistics.
    #[must_use]
    pub fn stats(&self) -> GraphStats {
        let mut num_constants = 0;
        let mut num_ops = 0;
        let mut num_edges = 0;
        let mut depths = vec![0usize; self.next_node_id()];
        let mut op_counts: Vec<(MultiAD, usize)> = Vec::new();

        for (offset, node) in self.nodes.iter().enumerate() {
            let node_id = self.num_inputs + offset;
            match node {
                GraphNode::Constant(_) => {
                    num_constants += 1;
                    depths[node_id] = 0;
                }
                GraphNode::Operation { op, inputs } => {
                    num_ops += 1;
                    num_edges += inputs.len();
                    let parent_depth = inputs.iter().map(|&id| depths[id]).max().unwrap_or(0);
                    depths[node_id] = parent_depth + 1;
                    if let Some((_, count)) = op_counts.iter_mut().find(|(kind, _)| kind == op) {
                        *count += 1;
                    } else {
                        op_counts.push((*op, 1));
                    }
                }
            }
        }

        GraphStats {
            num_inputs: self.num_inputs,
            num_constants,
            num_ops,
            num_edges,
            max_depth: depths.into_iter().max().unwrap_or(0),
            op_counts,
        }
    }

    /// Return a graph with simple local simplifications applied.
    ///
    /// Currently performs constant folding and a few algebraic identities with
    /// literal `0` and `1` constants.
    pub fn simplify(&self) -> Result<Graph> {
        self.validate()?;
        let mut simplified = Graph::new(self.num_inputs);
        simplified.input_names = self.input_names.clone();
        let mut old_to_new: Vec<NodeId> = (0..self.num_inputs).collect();
        let mut constant_values: Vec<Option<f64>> = vec![None; self.num_inputs];

        for node in &self.nodes {
            let new_node = match node {
                GraphNode::Constant(value) => {
                    let id = simplified.constant(*value);
                    constant_values.push(Some(*value));
                    id
                }
                GraphNode::Operation { op, inputs } => {
                    let mapped_inputs: Vec<NodeId> =
                        inputs.iter().map(|&id| old_to_new[id]).collect();
                    let input_constants: Vec<Option<f64>> = inputs
                        .iter()
                        .map(|&id| constant_values.get(id).copied().flatten())
                        .collect();

                    let folded = if input_constants.iter().all(Option::is_some) {
                        let args: Vec<f64> =
                            input_constants.iter().map(|value| value.unwrap()).collect();
                        Some(op_rules::forward_value(*op, &args)?)
                    } else {
                        None
                    };

                    if let Some(value) = folded {
                        let id = simplified.constant(value);
                        constant_values.push(Some(value));
                        id
                    } else if *op == MultiAD::Add && input_constants.first() == Some(&Some(0.0)) {
                        constant_values.push(None);
                        mapped_inputs[1]
                    } else if *op == MultiAD::Add && input_constants.get(1) == Some(&Some(0.0)) {
                        constant_values.push(None);
                        mapped_inputs[0]
                    } else if *op == MultiAD::Mul && input_constants.first() == Some(&Some(1.0)) {
                        constant_values.push(None);
                        mapped_inputs[1]
                    } else if *op == MultiAD::Mul && input_constants.get(1) == Some(&Some(1.0)) {
                        constant_values.push(None);
                        mapped_inputs[0]
                    } else {
                        let id = simplified.push_operation(*op, mapped_inputs);
                        constant_values.push(None);
                        id
                    }
                }
            };
            old_to_new.push(new_node);
        }

        let outputs: Vec<NodeId> = self
            .effective_output_nodes()
            .into_iter()
            .map(|id| old_to_new[id])
            .collect();
        simplified.set_outputs(&outputs)?;
        for (old_id, name) in &self.output_names {
            simplified
                .output_names
                .push((old_to_new[*old_id], name.clone()));
        }
        simplified.parameters = self.parameters.iter().map(|&id| old_to_new[id]).collect();
        for (old_id, name) in &self.parameter_names {
            simplified
                .parameter_names
                .push((old_to_new[*old_id], name.clone()));
        }
        Ok(simplified)
    }

    /// Return a new graph containing only nodes reachable from selected outputs.
    pub fn prune_to_outputs(&self) -> Result<Graph> {
        self.validate()?;
        let outputs = self.effective_output_nodes();
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = outputs.clone();

        while let Some(node_id) = stack.pop() {
            if node_id < self.num_inputs {
                continue;
            }
            let offset = node_id - self.num_inputs;
            if offset >= self.nodes.len() || reachable[offset] {
                continue;
            }
            reachable[offset] = true;
            if let GraphNode::Operation { inputs, .. } = &self.nodes[offset] {
                stack.extend(inputs.iter().copied());
            }
        }

        let mut pruned = Graph::new(self.num_inputs);
        pruned.input_names = self.input_names.clone();
        let mut old_to_new: Vec<Option<NodeId>> = vec![None; self.next_node_id()];
        for (input_id, slot) in old_to_new.iter_mut().enumerate().take(self.num_inputs) {
            *slot = Some(input_id);
        }

        for (offset, node) in self.nodes.iter().enumerate() {
            if !reachable[offset] {
                continue;
            }
            let old_id = self.num_inputs + offset;
            let new_id = match node {
                GraphNode::Constant(value) => pruned.constant(*value),
                GraphNode::Operation { op, inputs } => {
                    let remapped_inputs: Vec<NodeId> = inputs
                        .iter()
                        .map(|&id| {
                            old_to_new[id].ok_or(AutodiffError::InvalidGraph {
                                reason: "reachable graph remapping failed",
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    pruned.push_operation(*op, remapped_inputs)
                }
            };
            old_to_new[old_id] = Some(new_id);
        }

        let remapped_outputs: Vec<NodeId> = outputs
            .iter()
            .map(|&id| {
                old_to_new[id].ok_or(AutodiffError::InvalidGraph {
                    reason: "output remapping failed",
                })
            })
            .collect::<Result<Vec<_>>>()?;
        pruned.set_outputs(&remapped_outputs)?;
        for (old_id, name) in &self.output_names {
            if let Some(Some(new_id)) = old_to_new.get(*old_id) {
                pruned.output_names.push((*new_id, name.clone()));
            }
        }
        for &old_id in &self.parameters {
            if let Some(Some(new_id)) = old_to_new.get(old_id) {
                pruned.parameters.push(*new_id);
            }
        }
        for (old_id, name) in &self.parameter_names {
            if let Some(Some(new_id)) = old_to_new.get(*old_id) {
                pruned.parameter_names.push((*new_id, name.clone()));
            }
        }
        Ok(pruned)
    }

    /// Compile the graph into closure-free instruction IR.
    pub fn compile_ir(&self) -> Result<CompiledGraph> {
        self.validate()?;
        let mut instructions = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let instruction = match node {
                GraphNode::Constant(value) => Instruction::Constant(*value),
                GraphNode::Operation { op, inputs } => match inputs.as_slice() {
                    [arg] => Instruction::Unary { op: *op, arg: *arg },
                    [left, right] => Instruction::Binary {
                        op: *op,
                        left: *left,
                        right: *right,
                    },
                    _ => {
                        return Err(AutodiffError::InvalidGraph {
                            reason: "compiled IR supports only unary and binary operations",
                        });
                    }
                },
            };
            instructions.push(instruction);
        }
        CompiledGraph::new(self.num_inputs, instructions, self.effective_output_nodes())
    }

    /// Alias for [`Graph::compile_ir`] using acceleration-oriented naming.
    pub fn compile_accelerated(&self) -> Result<CompiledGraph> {
        self.compile_ir()
    }

    /// Compile the graph into a reusable tape.
    #[must_use]
    pub fn compile(&self) -> Tape {
        let mut compiled_nodes: Vec<CompiledNode> = Vec::with_capacity(self.nodes.len());
        let mut arg_indices: Vec<usize> = Vec::new();

        for node in &self.nodes {
            match node {
                GraphNode::Constant(value) => compiled_nodes.push(CompiledNode::Constant(*value)),
                GraphNode::Operation { op, inputs } => {
                    let start = arg_indices.len();
                    arg_indices.extend_from_slice(inputs);
                    compiled_nodes.push(CompiledNode::Operation {
                        op: *op,
                        arg_range: CompiledArgRange {
                            start,
                            len: inputs.len(),
                        },
                    });
                }
            }
        }

        Tape {
            graph: self.clone(),
            output_indices: self.effective_output_nodes(),
            compiled_nodes: Arc::from(compiled_nodes),
            arg_indices: Arc::from(arg_indices),
        }
    }

    /// Compile the graph after validating its structure.
    pub fn try_compile(&self) -> Result<Tape> {
        self.validate()?;
        Ok(self.compile())
    }

    #[inline]
    fn check_graph_input_len(&self, inputs: &[f64]) -> Result<()> {
        if inputs.len() == self.num_inputs {
            Ok(())
        } else {
            Err(AutodiffError::InvalidGraph {
                reason: "input length must match graph.num_inputs()",
            })
        }
    }

    fn emit_fr_ops(
        &self,
        node_id: NodeId,
        ops: &mut Vec<MultiAD2FR>,
        values: &mut Vec<f64>,
    ) -> Result<()> {
        if node_id < self.num_inputs {
            ops.push(MultiAD2FR::Inp(node_id));
            return Ok(());
        }
        let offset = node_id - self.num_inputs;
        let node = self
            .nodes
            .get(offset)
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: node_id,
                max_index: self.next_node_id().saturating_sub(1),
            })?;
        match node {
            GraphNode::Constant(value) => {
                let input_id = self.num_inputs + values.len();
                values.push(*value);
                ops.push(MultiAD2FR::Inp(input_id));
            }
            GraphNode::Operation { op, inputs } => {
                for &input in inputs {
                    self.emit_fr_ops(input, ops, values)?;
                }
                ops.push(Self::multi_to_fr(*op)?);
            }
        }
        Ok(())
    }

    fn emit_rf_ops(
        &self,
        node_id: NodeId,
        ops: &mut Vec<MultiAD2RF>,
        values: &mut Vec<f64>,
    ) -> Result<()> {
        if node_id < self.num_inputs {
            ops.push(MultiAD2RF::Inp(node_id));
            return Ok(());
        }
        let offset = node_id - self.num_inputs;
        let node = self
            .nodes
            .get(offset)
            .ok_or(AutodiffError::IndexOutOfBounds {
                index: node_id,
                max_index: self.next_node_id().saturating_sub(1),
            })?;
        match node {
            GraphNode::Constant(value) => {
                let input_id = self.num_inputs + values.len();
                values.push(*value);
                ops.push(MultiAD2RF::Inp(input_id));
            }
            GraphNode::Operation { op, inputs } => {
                for &input in inputs {
                    self.emit_rf_ops(input, ops, values)?;
                }
                ops.push(Self::multi_to_rf(*op)?);
            }
        }
        Ok(())
    }

    fn multi_to_fr(op: MultiAD) -> Result<MultiAD2FR> {
        Ok(match op {
            MultiAD::Sin => MultiAD2FR::Sin,
            MultiAD::Cos => MultiAD2FR::Cos,
            MultiAD::Tan => MultiAD2FR::Tan,
            MultiAD::Neg => MultiAD2FR::Neg,
            MultiAD::Exp => MultiAD2FR::Exp,
            MultiAD::Ln => MultiAD2FR::Ln,
            MultiAD::Sqrt => MultiAD2FR::Sqrt,
            MultiAD::Log1pExp => MultiAD2FR::Log1pExp,
            MultiAD::Add => MultiAD2FR::Add,
            MultiAD::Sub => MultiAD2FR::Sub,
            MultiAD::Mul => MultiAD2FR::Mul,
            MultiAD::Div => MultiAD2FR::Div,
            MultiAD::Pow => MultiAD2FR::Pow,
            MultiAD::Tanh | MultiAD::Relu | MultiAD::LogAddExp | MultiAD::Abs | MultiAD::Inp => {
                return Err(AutodiffError::InvalidGraph {
                    reason:
                        "exact graph Hessian supports only the exact-Hessian smooth operation set",
                });
            }
        })
    }

    fn multi_to_rf(op: MultiAD) -> Result<MultiAD2RF> {
        Ok(match op {
            MultiAD::Sin => MultiAD2RF::Sin,
            MultiAD::Cos => MultiAD2RF::Cos,
            MultiAD::Tan => MultiAD2RF::Tan,
            MultiAD::Neg => MultiAD2RF::Neg,
            MultiAD::Exp => MultiAD2RF::Exp,
            MultiAD::Ln => MultiAD2RF::Ln,
            MultiAD::Sqrt => MultiAD2RF::Sqrt,
            MultiAD::Log1pExp => MultiAD2RF::Log1pExp,
            MultiAD::Add => MultiAD2RF::Add,
            MultiAD::Sub => MultiAD2RF::Sub,
            MultiAD::Mul => MultiAD2RF::Mul,
            MultiAD::Div => MultiAD2RF::Div,
            MultiAD::Pow => MultiAD2RF::Pow,
            MultiAD::Tanh | MultiAD::Relu | MultiAD::LogAddExp | MultiAD::Abs | MultiAD::Inp => {
                return Err(AutodiffError::InvalidGraph {
                    reason:
                        "exact graph Hessian supports only the exact-Hessian smooth operation set",
                });
            }
        })
    }

    fn crop_hessian(&self, hessian: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        hessian
            .into_iter()
            .take(self.num_inputs)
            .map(|row| row.into_iter().take(self.num_inputs).collect())
            .collect()
    }

    fn validate_strict_derivative_domain(&self, inputs: &[f64]) -> Result<()> {
        self.check_graph_input_len(inputs)?;
        let mut values = inputs.to_vec();
        for node in &self.nodes {
            match node {
                GraphNode::Constant(value) => values.push(*value),
                GraphNode::Operation {
                    op,
                    inputs: arg_indices,
                } => {
                    let args = MultiAD::gather_arg_values(arg_indices, &values)?;
                    match op {
                        MultiAD::Sqrt if args[0] <= 0.0 => {
                            return Err(AutodiffError::domain(
                                "Sqrt",
                                "input must be positive for derivative evaluation",
                            ));
                        }
                        _ => {}
                    }
                    values.push(op.forward_checked(&args)?);
                }
            }
        }
        Ok(())
    }

    fn compute_exact_hessian_native(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.check_graph_input_len(inputs)?;
        self.validate()?;
        let output = self
            .effective_output_node()
            .ok_or(AutodiffError::EmptyGraph)?;
        if self.nodes.iter().any(|node| {
            matches!(
                node,
                GraphNode::Operation {
                    op: MultiAD::Abs | MultiAD::Relu,
                    ..
                }
            )
        }) {
            return Err(AutodiffError::InvalidGraph {
                reason: "native exact Hessian does not support non-smooth operations",
            });
        }

        let value_count = self.next_node_id();
        let mut values = Vec::with_capacity(value_count);
        values.extend_from_slice(inputs);

        let mut grads = vec![vec![0.0; self.num_inputs]; value_count];
        for (input_id, grad) in grads.iter_mut().enumerate().take(self.num_inputs) {
            grad[input_id] = 1.0;
        }

        let mut locals = vec![ExactLocal::None; value_count];

        for (offset, node) in self.nodes.iter().enumerate() {
            let node_id = self.num_inputs + offset;
            match node {
                GraphNode::Constant(value) => {
                    values.push(*value);
                }
                GraphNode::Operation { op, inputs } => {
                    let arg_values = MultiAD::gather_arg_values(inputs, &values)?;
                    let value = op_rules::forward_value(*op, &arg_values)?;
                    let local_rule = op_rules::local_rule(*op, &arg_values, value)?;
                    values.push(value);
                    match local_rule {
                        op_rules::LocalRule::Unary { dy, ddy } => {
                            let parent = inputs[0];
                            let parent_grad = grads[parent].clone();
                            for (var, slot) in
                                grads[node_id].iter_mut().enumerate().take(self.num_inputs)
                            {
                                *slot = dy * parent_grad[var];
                            }
                            locals[node_id] = ExactLocal::Unary { parent, dy, ddy };
                        }
                        op_rules::LocalRule::Binary {
                            dy_left,
                            dy_right,
                            ddy_left_left,
                            ddy_right_right,
                            ddy_left_right,
                        } => {
                            let left = inputs[0];
                            let right = inputs[1];
                            let left_grad = grads[left].clone();
                            let right_grad = grads[right].clone();
                            for (var, slot) in
                                grads[node_id].iter_mut().enumerate().take(self.num_inputs)
                            {
                                *slot = dy_left * left_grad[var] + dy_right * right_grad[var];
                            }
                            locals[node_id] = ExactLocal::Binary {
                                left,
                                right,
                                dy_left,
                                dy_right,
                                ddy_left_left,
                                ddy_right_right,
                                ddy_left_right,
                            };
                        }
                    }
                }
            }
        }

        let mut adjoints = vec![0.0; value_count];
        let mut hessian = vec![vec![0.0; self.num_inputs]; self.num_inputs];
        adjoints[output] = 1.0;

        for node_id in (0..value_count).rev() {
            let adjoint = adjoints[node_id];
            if adjoint == 0.0 {
                continue;
            }
            match locals[node_id].clone() {
                ExactLocal::None => {}
                ExactLocal::Unary { parent, dy, ddy } => {
                    adjoints[parent] += adjoint * dy;
                    for row in 0..self.num_inputs {
                        for col in 0..self.num_inputs {
                            hessian[row][col] +=
                                adjoint * ddy * grads[parent][row] * grads[parent][col];
                        }
                    }
                }
                ExactLocal::Binary {
                    left,
                    right,
                    dy_left,
                    dy_right,
                    ddy_left_left,
                    ddy_right_right,
                    ddy_left_right,
                } => {
                    adjoints[left] += adjoint * dy_left;
                    adjoints[right] += adjoint * dy_right;
                    for row in 0..self.num_inputs {
                        for col in 0..self.num_inputs {
                            hessian[row][col] += adjoint
                                * (ddy_left_left * grads[left][row] * grads[left][col]
                                    + ddy_right_right * grads[right][row] * grads[right][col]
                                    + ddy_left_right
                                        * (grads[left][row] * grads[right][col]
                                            + grads[right][row] * grads[left][col]));
                        }
                    }
                }
            }
        }

        Ok(hessian)
    }

    /// Compute only the primary output value.
    pub fn compute(&self, inputs: &[f64]) -> Result<f64> {
        self.compile().compute(inputs)
    }

    /// Return the preferred backend for batch value computation.
    pub fn recommended_batch_compute_backend(&self) -> Result<BackendKind> {
        Ok(self.compile_ir()?.recommended_batch_compute_backend())
    }

    /// Return the preferred backend for batch gradient computation.
    pub fn recommended_batch_gradient_backend(&self) -> Result<BackendKind> {
        Ok(self.compile_ir()?.recommended_batch_gradient_backend())
    }

    /// Compute all selected outputs for a batch via compiled IR.
    pub fn compute_batch(&self, batch: BatchInputs<'_>) -> Result<BatchValues> {
        self.compile_ir()?.compute_batch(batch)
    }

    /// Compute all selected outputs into a reusable output buffer via compiled IR.
    pub fn compute_batch_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<()> {
        self.compile_ir()?.compute_batch_into(batch, buffer)
    }

    /// Compute all selected outputs for a batch with automatic backend dispatch.
    pub fn compute_batch_auto(&self, batch: BatchInputs<'_>) -> Result<(BackendKind, BatchValues)> {
        self.compile_ir()?.compute_batch_auto(batch)
    }

    /// Compute all selected outputs into a reusable output buffer with automatic backend dispatch.
    pub fn compute_batch_auto_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchValuesBuffer,
    ) -> Result<BackendKind> {
        self.compile_ir()?.compute_batch_auto_into(batch, buffer)
    }

    /// Compute primary-output values and gradients for a batch via compiled IR.
    pub fn gradient_batch(&self, batch: BatchInputs<'_>) -> Result<BatchGradients> {
        self.compile_ir()?.gradient_batch(batch)
    }

    /// Compute primary-output values and gradients into a reusable buffer via compiled IR.
    pub fn gradient_batch_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<()> {
        self.compile_ir()?.gradient_batch_into(batch, buffer)
    }

    /// Compute primary-output values and gradients for a batch with automatic backend dispatch.
    pub fn gradient_batch_auto(
        &self,
        batch: BatchInputs<'_>,
    ) -> Result<(BackendKind, BatchGradients)> {
        self.compile_ir()?.gradient_batch_auto(batch)
    }

    /// Compute primary-output values and gradients into a reusable buffer with automatic backend dispatch.
    pub fn gradient_batch_auto_into(
        &self,
        batch: BatchInputs<'_>,
        buffer: &mut BatchGradientsBuffer,
    ) -> Result<BackendKind> {
        self.compile_ir()?.gradient_batch_auto_into(batch, buffer)
    }

    /// Return compiled IR metadata for backend planning.
    pub fn compiled_metadata(&self) -> Result<CompiledGraphMetadata> {
        Ok(self.compile_ir()?.metadata())
    }

    /// Return backend compatibility details for a specific backend.
    pub fn backend_support_report(&self, backend: BackendKind) -> Result<BackendSupportReport> {
        self.compile_ir()?.backend_support_report(backend)
    }

    /// Return backend compatibility details for all built-in backends.
    pub fn backend_support_reports(&self) -> Result<Vec<BackendSupportReport>> {
        self.compile_ir()?.backend_support_reports()
    }

    /// Return a device-oriented batch buffer plan for a backend.
    pub fn device_batch_plan(
        &self,
        backend: BackendKind,
        batch_size: usize,
    ) -> Result<DeviceBatchPlan> {
        Ok(self.compile_ir()?.device_batch_plan(backend, batch_size))
    }

    /// Allocate mock-device buffers for this graph and batch size.
    pub fn allocate_mock_device_buffers(&self, batch_size: usize) -> Result<DeviceBufferSet> {
        Ok(self.compile_ir()?.allocate_mock_device_buffers(batch_size))
    }

    /// Execute batch value computation through mock-device buffers.
    pub fn compute_batch_mock_device_into(
        &self,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        self.compile_ir()?
            .compute_batch_mock_device_into(batch, buffers, output)
    }

    /// Execute batch gradient computation through mock-device buffers.
    pub fn gradient_batch_mock_device_into(
        &self,
        batch: BatchInputs<'_>,
        buffers: &mut DeviceBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        self.compile_ir()?
            .gradient_batch_mock_device_into(batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Allocate real WGPU buffers for this graph and batch size.
    pub fn allocate_wgpu_buffers(
        &self,
        backend: &crate::WgpuBackend,
        batch_size: usize,
    ) -> Result<crate::WgpuBufferSet> {
        self.compile_ir()?
            .allocate_wgpu_buffers(backend, batch_size)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Execute batch value computation through real WGPU buffers.
    pub fn compute_batch_wgpu_into(
        &self,
        backend: &crate::WgpuBackend,
        batch: BatchInputs<'_>,
        buffers: &mut crate::WgpuBufferSet,
        output: &mut BatchValuesBuffer,
    ) -> Result<DeviceExecutionTrace> {
        self.compile_ir()?
            .compute_batch_wgpu_into(backend, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Execute batch gradient computation through real WGPU buffers.
    pub fn gradient_batch_wgpu_into(
        &self,
        backend: &crate::WgpuBackend,
        batch: BatchInputs<'_>,
        buffers: &mut crate::WgpuBufferSet,
        output: &mut BatchGradientsBuffer,
    ) -> Result<DeviceExecutionTrace> {
        self.compile_ir()?
            .gradient_batch_wgpu_into(backend, batch, buffers, output)
    }

    #[cfg(feature = "backend-wgpu")]
    /// Return whether this graph is statically eligible for the exact-safe native WGPU compute path.
    pub fn supports_native_wgpu_batch_compute(&self, backend: &crate::WgpuBackend) -> Result<bool> {
        Ok(self
            .compile_ir()?
            .supports_native_wgpu_batch_compute(backend))
    }

    #[cfg(feature = "backend-wgpu")]
    /// Return whether this graph and concrete batch can use the exact-safe native WGPU compute path.
    pub fn supports_native_wgpu_batch_compute_for_batch(
        &self,
        backend: &crate::WgpuBackend,
        batch: BatchInputs<'_>,
    ) -> Result<bool> {
        Ok(self
            .compile_ir()?
            .supports_native_wgpu_batch_compute_for_batch(backend, batch))
    }

    /// Return SIMD backend compatibility details for this graph.
    pub fn simd_support_report(&self) -> Result<BackendSupportReport> {
        self.compile_ir()?.simd_support_report()
    }

    /// Create a compiled IR workspace for this graph.
    pub fn compiled_workspace(&self) -> Result<CompiledWorkspace> {
        Ok(self.compile_ir()?.workspace())
    }

    /// Compute all selected output values.
    pub fn compute_many(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        self.compile().compute_many(inputs)
    }

    /// Compute all selected output values with checked-domain validation.
    pub fn compute_many_checked(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        self.compile().compute_many_checked(inputs)
    }

    /// Compute only the output value with checked-domain validation.
    pub fn compute_checked(&self, inputs: &[f64]) -> Result<f64> {
        self.compile().compute_checked(inputs)
    }

    /// Compute the primary output value and gradient closure.
    pub fn compute_grad(&self, inputs: &[f64]) -> Result<BackwardResultBox> {
        self.compile().compute_grad(inputs)
    }

    /// Compute the Jacobian for all selected outputs.
    pub fn jacobian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.compile().jacobian(inputs)
    }

    /// Compute the Jacobian for all selected outputs with checked-domain validation.
    pub fn jacobian_checked(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.compile().jacobian_checked(inputs)
    }

    /// Compute all selected output values and their Jacobian.
    pub fn value_and_jacobian(&self, inputs: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let tape = self.compile();
        Ok((tape.compute_many(inputs)?, tape.jacobian(inputs)?))
    }

    /// Compute all selected output values and their Jacobian with checked domains.
    pub fn value_and_jacobian_checked(&self, inputs: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let tape = self.compile();
        Ok((
            tape.compute_many_checked(inputs)?,
            tape.jacobian_checked(inputs)?,
        ))
    }

    /// Compute the primary output value and eager gradient.
    pub fn gradient(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.compile().gradient(inputs)
    }

    /// Alias for [`Graph::gradient`] with a name common in optimization code.
    pub fn value_and_gradient(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.gradient(inputs)
    }

    /// Compute the primary output value with an explicit domain policy.
    pub fn compute_with_domain_policy(&self, inputs: &[f64], policy: DomainPolicy) -> Result<f64> {
        match policy {
            DomainPolicy::Unchecked => self.compute(inputs),
            DomainPolicy::Checked => self.compute_checked(inputs),
            DomainPolicy::StrictDerivative => {
                self.validate_strict_derivative_domain(inputs)?;
                self.compute_checked(inputs)
            }
        }
    }

    /// Compute the primary output value and eager gradient with checked-domain validation.
    pub fn gradient_checked(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.compile().gradient_checked(inputs)
    }

    /// Compute the primary output value and gradient with an explicit domain policy.
    pub fn gradient_with_domain_policy(
        &self,
        inputs: &[f64],
        policy: DomainPolicy,
    ) -> Result<(f64, Vec<f64>)> {
        match policy {
            DomainPolicy::Unchecked => self.gradient(inputs),
            DomainPolicy::Checked => self.gradient_checked(inputs),
            DomainPolicy::StrictDerivative => {
                self.validate_strict_derivative_domain(inputs)?;
                self.gradient_checked(inputs)
            }
        }
    }

    /// Compute a finite-difference Hessian using repeated gradient evaluation.
    pub fn compute_hessian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.compile().compute_hessian(inputs)
    }

    /// Compute an exact Hessian with a native graph traversal.
    ///
    /// Constant nodes are handled directly as zero-gradient graph nodes instead
    /// of being translated to synthetic inputs.
    pub fn exact_hessian_rr(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.compute_exact_hessian_native(inputs)
    }

    /// Compute an exact Hessian using the graph-to-RPN FR translation.
    pub fn exact_hessian_fr(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.check_graph_input_len(inputs)?;
        self.validate()?;
        let output = self
            .effective_output_node()
            .ok_or(AutodiffError::EmptyGraph)?;
        let mut ops = Vec::new();
        let mut extra_inputs = Vec::new();
        self.emit_fr_ops(output, &mut ops, &mut extra_inputs)?;
        let mut all_inputs = inputs.to_vec();
        all_inputs.extend(extra_inputs);
        let hessian = MultiAD2FR::compute_hessian(&ops, &all_inputs)?;
        Ok(self.crop_hessian(hessian))
    }

    /// Compute an exact Hessian using the graph-to-RPN RF translation.
    pub fn exact_hessian_rf(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.check_graph_input_len(inputs)?;
        self.validate()?;
        let output = self
            .effective_output_node()
            .ok_or(AutodiffError::EmptyGraph)?;
        let mut ops = Vec::new();
        let mut extra_inputs = Vec::new();
        self.emit_rf_ops(output, &mut ops, &mut extra_inputs)?;
        let mut all_inputs = inputs.to_vec();
        all_inputs.extend(extra_inputs);
        let hessian = MultiAD2RF::compute_hessian(&ops, &all_inputs)?;
        Ok(self.crop_hessian(hessian))
    }

    /// Compute a finite-difference Hessian-vector product.
    pub fn hessian_vector_product(&self, inputs: &[f64], vector: &[f64]) -> Result<Vec<f64>> {
        self.check_graph_input_len(inputs)?;
        if vector.len() != self.num_inputs {
            return Err(AutodiffError::InvalidGraph {
                reason: "vector length must match graph.num_inputs()",
            });
        }
        let epsilon = 1e-5;
        let inputs_plus: Vec<f64> = inputs
            .iter()
            .zip(vector)
            .map(|(x, v)| x + epsilon * v)
            .collect();
        let inputs_minus: Vec<f64> = inputs
            .iter()
            .zip(vector)
            .map(|(x, v)| x - epsilon * v)
            .collect();
        let (_value_plus, grad_plus) = self.gradient(&inputs_plus)?;
        let (_value_minus, grad_minus) = self.gradient(&inputs_minus)?;
        Ok(grad_plus
            .iter()
            .zip(grad_minus)
            .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
            .collect())
    }

    /// Return non-zero gradient entries using a default `1e-12` threshold.
    pub fn gradient_sparse(&self, inputs: &[f64]) -> Result<Vec<(usize, f64)>> {
        self.gradient_sparse_with_tolerance(inputs, 1e-12)
    }

    /// Return gradient entries whose absolute value is greater than `tolerance`.
    pub fn gradient_sparse_with_tolerance(
        &self,
        inputs: &[f64],
        tolerance: f64,
    ) -> Result<Vec<(usize, f64)>> {
        let (_value, gradient) = self.gradient(inputs)?;
        Ok(gradient
            .into_iter()
            .enumerate()
            .filter(|(_, value)| value.abs() > tolerance)
            .collect())
    }

    /// Return non-zero Hessian entries using a default `1e-12` threshold.
    pub fn hessian_sparse(&self, inputs: &[f64]) -> Result<Vec<(usize, usize, f64)>> {
        self.hessian_sparse_with_tolerance(inputs, 1e-12)
    }

    /// Return Hessian entries whose absolute value is greater than `tolerance`.
    pub fn hessian_sparse_with_tolerance(
        &self,
        inputs: &[f64],
        tolerance: f64,
    ) -> Result<Vec<(usize, usize, f64)>> {
        let hessian = self.compute_hessian(inputs)?;
        let mut sparse = Vec::new();
        for (row_index, row) in hessian.iter().enumerate() {
            for (col_index, value) in row.iter().enumerate() {
                if value.abs() > tolerance {
                    sparse.push((row_index, col_index, *value));
                }
            }
        }
        Ok(sparse)
    }

    /// Compare reverse-mode gradients to central finite differences.
    pub fn check_gradient(&self, inputs: &[f64], tolerance: f64) -> Result<GradientCheckReport> {
        self.check_graph_input_len(inputs)?;
        let (_value, autodiff_gradient) = self.gradient(inputs)?;
        let epsilon = 1e-6;
        let mut entries = Vec::with_capacity(self.num_inputs);
        let mut max_abs_error: f64 = 0.0;

        for index in 0..self.num_inputs {
            let mut plus = inputs.to_vec();
            let mut minus = inputs.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let value_plus = self.compute(&plus)?;
            let value_minus = self.compute(&minus)?;
            let finite_difference = (value_plus - value_minus) / (2.0 * epsilon);
            let abs_error = (autodiff_gradient[index] - finite_difference).abs();
            max_abs_error = max_abs_error.max(abs_error);
            entries.push(GradientCheckEntry {
                index,
                autodiff: autodiff_gradient[index],
                finite_difference,
                abs_error,
            });
        }

        Ok(GradientCheckReport {
            passed: max_abs_error <= tolerance,
            tolerance,
            max_abs_error,
            entries,
        })
    }
}

impl ExprGraph {
    /// Create an expression graph with `num_inputs` input variables.
    #[must_use]
    pub fn new(num_inputs: usize) -> Self {
        Self {
            graph: Rc::new(RefCell::new(Graph::new(num_inputs))),
        }
    }

    /// Return an expression node for an input.
    #[must_use]
    pub fn input(&self, input_index: usize) -> ExprNode {
        let node = self.graph.borrow().input(input_index);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    /// Return an expression node for a literal constant.
    pub fn constant(&self, value: f64) -> ExprNode {
        let node = self.graph.borrow_mut().constant(value);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    /// Select an expression node as the graph output.
    pub fn set_output(&self, expr: &ExprNode) -> Result<()> {
        self.graph.borrow_mut().set_output(expr.node)?;
        Ok(())
    }

    /// Clone out the underlying reusable graph.
    #[must_use]
    pub fn graph(&self) -> Graph {
        self.graph.borrow().clone()
    }
}

impl ExprNode {
    /// Return the underlying node id.
    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.node
    }

    fn same_graph(&self, other: &ExprNode) {
        assert!(
            Rc::ptr_eq(&self.graph, &other.graph),
            "ExprNode graph mismatch"
        );
    }

    fn unary(&self, op: MultiAD) -> ExprNode {
        let node = self.graph.borrow_mut().push_operation(op, vec![self.node]);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    fn binary(&self, op: MultiAD, other: &ExprNode) -> ExprNode {
        self.same_graph(other);
        let node = self
            .graph
            .borrow_mut()
            .push_operation(op, vec![self.node, other.node]);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    fn binary_const(&self, op: MultiAD, value: f64) -> ExprNode {
        let mut graph = self.graph.borrow_mut();
        let constant = graph.constant(value);
        let node = graph.push_operation(op, vec![self.node, constant]);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    /// Append `sin(self)`.
    pub fn sin(&self) -> ExprNode {
        self.unary(MultiAD::Sin)
    }

    /// Append `cos(self)`.
    pub fn cos(&self) -> ExprNode {
        self.unary(MultiAD::Cos)
    }

    /// Append `exp(self)`.
    pub fn exp(&self) -> ExprNode {
        self.unary(MultiAD::Exp)
    }

    /// Append `ln(self)`.
    pub fn ln(&self) -> ExprNode {
        self.unary(MultiAD::Ln)
    }

    /// Append `sqrt(self)`.
    pub fn sqrt(&self) -> ExprNode {
        self.unary(MultiAD::Sqrt)
    }
}

impl Add for ExprNode {
    type Output = ExprNode;

    fn add(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Add, &rhs)
    }
}

impl Add<f64> for ExprNode {
    type Output = ExprNode;

    fn add(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Add, rhs)
    }
}

impl Sub for ExprNode {
    type Output = ExprNode;

    fn sub(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Sub, &rhs)
    }
}

impl Sub<f64> for ExprNode {
    type Output = ExprNode;

    fn sub(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Sub, rhs)
    }
}

impl Mul for ExprNode {
    type Output = ExprNode;

    fn mul(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Mul, &rhs)
    }
}

impl Mul<f64> for ExprNode {
    type Output = ExprNode;

    fn mul(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Mul, rhs)
    }
}

impl Div for ExprNode {
    type Output = ExprNode;

    fn div(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Div, &rhs)
    }
}

impl Div<f64> for ExprNode {
    type Output = ExprNode;

    fn div(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Div, rhs)
    }
}

impl Neg for ExprNode {
    type Output = ExprNode;

    fn neg(self) -> Self::Output {
        self.unary(MultiAD::Neg)
    }
}

impl TryFrom<&Graph> for Vec<(MultiAD, Vec<usize>)> {
    type Error = crate::AutodiffError;

    fn try_from(graph: &Graph) -> Result<Self> {
        graph.to_operations()
    }
}

impl TryFrom<Graph> for Vec<(MultiAD, Vec<usize>)> {
    type Error = crate::AutodiffError;

    fn try_from(graph: Graph) -> Result<Self> {
        graph.to_operations()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::approx_eq_eps as approx_eq;
    use crate::{
        BackendCapabilities, BackendKind, ExecutionBackend, OpCode, ScalarBackend, SimdBackend,
    };

    #[test]
    fn test_graph_supports_constants() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let two = graph.constant(2.0);
        let x_sq = graph.mul(x, x);
        graph.add(x_sq, two);

        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 11.0, 1e-10));

        let (_value, grad_fn) = graph.compute_grad(&[3.0]).unwrap();
        let grad = grad_fn(1.0);
        assert_eq!(grad.len(), 1);
        assert!(approx_eq(grad[0], 6.0, 1e-10));
    }

    #[test]
    fn test_graph_compile_reuses_graph() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        graph.mul(sum, y);

        let tape = graph.compile();
        let value = tape.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 15.0, 1e-10));

        let (_value, grad_fn) = tape.compute_grad(&[2.0, 3.0]).unwrap();
        let grad = grad_fn(1.0);
        assert!(approx_eq(grad[0], 3.0, 1e-10));
        assert!(approx_eq(grad[1], 8.0, 1e-10));
    }

    #[test]
    fn test_tape_workspace_eager_gradient() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        graph.mul(sum, y);

        let tape = graph.compile();
        let mut workspace = tape.workspace();
        let (value, grad) = tape
            .gradient_with_workspace(&[2.0, 3.0], &mut workspace)
            .unwrap();
        assert!(approx_eq(value, 15.0, 1e-10));
        assert!(approx_eq(grad[0], 3.0, 1e-10));
        assert!(approx_eq(grad[1], 8.0, 1e-10));

        let value_again = tape
            .compute_with_workspace(&[4.0, 1.5], &mut workspace)
            .unwrap();
        assert!(approx_eq(value_again, 8.25, 1e-10));
    }

    #[test]
    fn test_graph_rejects_wrong_input_length() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.mul(x, x);

        assert!(graph.compute(&[]).is_err());
        assert!(graph.compute(&[2.0, 3.0]).is_err());
        assert!(graph.compute_checked(&[2.0, 3.0]).is_err());
        assert!(graph.compute_grad(&[2.0, 3.0]).is_err());
        assert!(graph.jacobian(&[2.0, 3.0]).is_err());

        let tape = graph.compile();
        assert!(tape.compute(&[2.0, 3.0]).is_err());
        assert!(tape.compute_grad(&[2.0, 3.0]).is_err());
        assert!(tape.gradient(&[2.0, 3.0]).is_err());
    }

    #[test]
    fn test_graph_checked_compute_and_gradient() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.ln(x);

        let error = graph.compute_checked(&[0.0]).unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::DomainError {
                operation: "Ln",
                reason: "input must be positive",
            }
        );

        let ok = graph.gradient_checked(&[2.0]).unwrap();
        assert!(approx_eq(ok.0, 2.0_f64.ln(), 1e-10));
        assert!(approx_eq(ok.1[0], 0.5, 1e-10));
    }

    #[test]
    fn test_graph_checked_multi_output_and_jacobian() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let ratio = graph.div(x, y);
        let log_y = graph.ln(y);
        graph.set_outputs(&[ratio, log_y]).unwrap();

        let values = graph.compute_many_checked(&[4.0, 2.0]).unwrap();
        assert_eq!(values.len(), 2);
        assert!(approx_eq(values[0], 2.0, 1e-10));
        assert!(approx_eq(values[1], 2.0_f64.ln(), 1e-10));

        let jacobian = graph.jacobian_checked(&[4.0, 2.0]).unwrap();
        assert_eq!(jacobian.len(), 2);
        assert!(approx_eq(jacobian[0][0], 0.5, 1e-10));
        assert!(approx_eq(jacobian[0][1], -1.0, 1e-10));
        assert!(approx_eq(jacobian[1][0], 0.0, 1e-10));
        assert!(approx_eq(jacobian[1][1], 0.5, 1e-10));

        let error = graph.compute_many_checked(&[4.0, 0.0]).unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::DomainError {
                operation: "Div",
                reason: "denominator must be non-zero",
            }
        );
    }

    #[test]
    fn test_tape_checked_workspace() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let ratio = graph.div(x, y);
        let log_y = graph.ln(y);
        graph.set_outputs(&[ratio, log_y]).unwrap();
        let tape = graph.compile();
        let mut workspace = tape.workspace();

        let values = tape
            .compute_many_with_workspace_checked(&[4.0, 2.0], &mut workspace)
            .unwrap();
        assert!(approx_eq(values[0], 2.0, 1e-10));
        assert!(approx_eq(values[1], 2.0_f64.ln(), 1e-10));

        let jacobian = tape
            .jacobian_with_workspace_checked(&[4.0, 2.0], &mut workspace)
            .unwrap();
        assert!(approx_eq(jacobian[0][0], 0.5, 1e-10));
        assert!(approx_eq(jacobian[0][1], -1.0, 1e-10));
        assert!(approx_eq(jacobian[1][0], 0.0, 1e-10));
        assert!(approx_eq(jacobian[1][1], 0.5, 1e-10));

        let error = tape
            .compute_with_workspace_checked(&[1.0, 0.0], &mut workspace)
            .unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::DomainError {
                operation: "Div",
                reason: "denominator must be non-zero",
            }
        );
    }

    #[test]
    fn test_graph_from_operations() {
        let ops = [
            (MultiAD::Inp, vec![0]),
            (MultiAD::Inp, vec![1]),
            (MultiAD::Add, vec![0, 1]),
            (MultiAD::Sin, vec![2]),
        ];
        let graph = Graph::from_operations(2, &ops);
        let value = graph.compute(&[0.5, 0.25]).unwrap();
        assert!(approx_eq(value, (0.75_f64).sin(), 1e-10));
    }

    #[test]
    fn test_graph_from_operations_preserves_trailing_input_output() {
        let ops = [(MultiAD::Add, vec![0, 1]), (MultiAD::Inp, vec![0])];
        let graph = Graph::from_operations(2, &ops);
        let value = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_graph_from_operations_preserves_input_only_output() {
        let ops = [(MultiAD::Inp, vec![0])];
        let graph = Graph::from_operations(2, &ops);
        let value = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_graph_to_operations_round_trip() {
        let ops = [
            (MultiAD::Inp, vec![0]),
            (MultiAD::Inp, vec![1]),
            (MultiAD::Add, vec![0, 1]),
            (MultiAD::Sin, vec![2]),
        ];
        let graph = Graph::from_operations(2, &ops);
        let converted = graph.to_operations().unwrap();
        assert_eq!(
            converted,
            vec![(MultiAD::Add, vec![0, 1]), (MultiAD::Sin, vec![2])]
        );

        let value_legacy = MultiAD::compute(&converted, &[0.5, 0.25]).unwrap();
        let value_graph = graph.compute(&[0.5, 0.25]).unwrap();
        assert!(approx_eq(value_legacy, value_graph, 1e-10));
    }

    #[test]
    fn test_graph_to_operations_rejects_constants() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let c = graph.constant(2.0);
        graph.add(x, c);

        let error = graph.to_operations().unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::InvalidGraph {
                reason: "legacy tuple graphs cannot represent constant nodes",
            }
        );
    }

    #[test]
    fn test_graph_validate_accepts_valid_graph() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        graph.sin(sum);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_graph_validate_rejects_bad_index() {
        let mut graph = Graph::new(1);
        graph.push_operation(MultiAD::Sin, vec![3]);
        let error = graph.validate().unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::IndexOutOfBounds {
                index: 3,
                max_index: 0,
            }
        );
    }

    #[test]
    fn test_graph_export_formats() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let neg_x = graph.neg(x);
        graph.exp(neg_x);

        let mermaid = graph.to_mermaid();
        let dot = graph.to_dot();
        assert!(mermaid.contains("flowchart LR"));
        assert!(mermaid.contains("Neg"));
        assert!(dot.contains("digraph Graph"));
        assert!(dot.contains("Exp"));
    }

    #[test]
    fn test_graph_explicit_output_uses_selected_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let x_sq = graph.mul(x, x);
        let x_cu = graph.mul(x_sq, x);
        graph.set_output(x_sq).unwrap();

        let value = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(value, 4.0, 1e-10));

        let (_value, grad_fn) = graph.compute_grad(&[2.0]).unwrap();
        let grad = grad_fn(1.0);
        assert!(approx_eq(grad[0], 4.0, 1e-10));

        let ops = graph.to_operations().unwrap();
        let legacy_value = MultiAD::compute(&ops, &[2.0]).unwrap();
        assert!(approx_eq(legacy_value, 4.0, 1e-10));

        graph.set_output(x_cu).unwrap();
        let value_cu = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(value_cu, 8.0, 1e-10));
    }

    #[test]
    fn test_graph_clear_output_restores_last_node_behavior() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let x_sq = graph.mul(x, x);
        graph.mul(x_sq, x);
        graph.set_output(x_sq).unwrap();
        assert!(approx_eq(graph.compute(&[2.0]).unwrap(), 4.0, 1e-10));
        graph.clear_output();
        assert!(approx_eq(graph.compute(&[2.0]).unwrap(), 8.0, 1e-10));
    }

    #[test]
    fn test_graph_validate_rejects_bad_explicit_output() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sin(x);
        graph.output_nodes = vec![4];
        let error = graph.validate().unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::IndexOutOfBounds {
                index: 4,
                max_index: 1,
            }
        );
    }

    #[test]
    fn test_graph_to_operations_with_input_output_marker() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);
        graph.set_output(x).unwrap();

        let ops = graph.to_operations().unwrap();
        assert_eq!(ops.last(), Some(&(MultiAD::Inp, vec![0])));
        let value = MultiAD::compute(&ops, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_graph_multi_output_values_and_jacobian() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        let product = graph.mul(x, y);
        graph.set_outputs(&[sum, product]).unwrap();

        let values = graph.compute_many(&[2.0, 3.0]).unwrap();
        assert_eq!(values.len(), 2);
        assert!(approx_eq(values[0], 5.0, 1e-10));
        assert!(approx_eq(values[1], 6.0, 1e-10));

        let jacobian = graph.jacobian(&[2.0, 3.0]).unwrap();
        assert_eq!(jacobian.len(), 2);
        assert!(approx_eq(jacobian[0][0], 1.0, 1e-10));
        assert!(approx_eq(jacobian[0][1], 1.0, 1e-10));
        assert!(approx_eq(jacobian[1][0], 3.0, 1e-10));
        assert!(approx_eq(jacobian[1][1], 2.0, 1e-10));
    }

    #[test]
    fn test_graph_to_operations_rejects_multi_output() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        let product = graph.mul(x, y);
        graph.set_outputs(&[sum, product]).unwrap();

        let error = graph.to_operations().unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::InvalidGraph {
                reason: "legacy tuple graphs support only one output",
            }
        );
    }

    #[test]
    fn test_safe_construction_and_compile() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let x_sq = graph.try_push_operation(MultiAD::Mul, vec![x, x]).unwrap();
        graph.set_output(x_sq).unwrap();
        assert!(graph.try_compile().is_ok());
        assert!(graph.try_push_operation(MultiAD::Inp, vec![0]).is_err());
        assert!(Graph::try_from_operations(1, &[(MultiAD::Sin, vec![2])]).is_err());
    }

    #[test]
    fn test_gradient_helpers_and_gradient_check() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.square(x);

        let (value, gradient) = graph.value_and_gradient(&[3.0]).unwrap();
        assert!(approx_eq(value, 9.0, 1e-10));
        assert!(approx_eq(gradient[0], 6.0, 1e-10));

        let report = graph.check_gradient(&[3.0], 1e-6).unwrap();
        assert!(report.passed);
        assert_eq!(report.entries.len(), 1);
    }

    #[test]
    fn test_constant_and_expression_helpers() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let x_sq = graph.square(x);
        let shifted = graph.add_const(x_sq, 2.0);
        let sigmoid = graph.sigmoid(x);
        graph.set_outputs(&[shifted, sigmoid]).unwrap();

        let values = graph.compute_many(&[3.0]).unwrap();
        assert!(approx_eq(values[0], 11.0, 1e-10));
        assert!(values[1] > 0.0 && values[1] < 1.0);
    }

    #[test]
    fn test_graph_builder_constant_node() {
        let mut builder = crate::GraphBuilder::new(1);
        let x = builder.input_node(0);
        let two = builder.constant_node(2.0);
        let out = builder.add_node(x, two);
        let graph = builder.build_graph_with_output(out).unwrap();
        assert!(approx_eq(graph.compute(&[3.0]).unwrap(), 5.0, 1e-10));
    }

    #[test]
    fn test_exact_hessian_hvp_and_sparse_outputs() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let x_sq = graph.square(x);
        let y_sq = graph.square(y);
        graph.add(x_sq, y_sq);

        let expected = [[2.0, 0.0], [0.0, 2.0]];
        for hessian in [
            graph.exact_hessian_rr(&[1.0, 2.0]).unwrap(),
            graph.exact_hessian_fr(&[1.0, 2.0]).unwrap(),
            graph.exact_hessian_rf(&[1.0, 2.0]).unwrap(),
        ] {
            for i in 0..2 {
                for j in 0..2 {
                    assert!(approx_eq(hessian[i][j], expected[i][j], 1e-10));
                }
            }
        }

        let hvp = graph
            .hessian_vector_product(&[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        assert!(approx_eq(hvp[0], 6.0, 1e-6));
        assert!(approx_eq(hvp[1], 8.0, 1e-6));

        let sparse_grad = graph.gradient_sparse(&[1.0, 2.0]).unwrap();
        assert_eq!(sparse_grad.len(), 2);
        let sparse_hessian = graph.hessian_sparse(&[1.0, 2.0]).unwrap();
        assert_eq!(sparse_hessian.len(), 2);
    }

    #[test]
    fn test_prune_names_stats_and_exports() {
        let mut graph = Graph::new(2);
        graph.set_input_name(0, "x").unwrap();
        let x = graph.input(0);
        let y = graph.input(1);
        let used = graph.square(x);
        let _unused = graph.square(y);
        graph.set_output_name(used, "objective").unwrap();

        let pruned = graph.prune_to_outputs().unwrap();
        assert_eq!(pruned.len(), 1);
        let stats = graph.stats();
        assert_eq!(stats.num_inputs, 2);
        assert_eq!(stats.num_ops, 2);
        assert!(graph.to_mermaid().contains("x: Input 0"));
        assert!(graph.to_dot().contains("objective"));
    }

    #[test]
    fn test_expr_graph_operator_overloads() {
        let expr_graph = ExprGraph::new(2);
        let x = expr_graph.input(0);
        let y = expr_graph.input(1);
        let out = x.clone().sin() * (x + y) + 2.0;
        expr_graph.set_output(&out).unwrap();
        let graph = expr_graph.graph();
        let value = graph.compute(&[0.6, 1.4]).unwrap();
        let expected = 0.6_f64.sin() * 2.0 + 2.0;
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_ml_activations_and_softmax() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let tanh = graph.tanh(x);
        let relu = graph.relu(x);
        let gelu = graph.gelu(x);
        let softmax = graph.softmax(&[x, y]);
        graph
            .set_outputs(&[tanh, relu, gelu, softmax[0], softmax[1]])
            .unwrap();
        let values = graph.compute_many(&[1.0, 2.0]).unwrap();
        assert!(approx_eq(values[0], 1.0_f64.tanh(), 1e-10));
        assert!(approx_eq(values[1], 1.0, 1e-10));
        assert!(values[2] > 0.0);
        assert!(approx_eq(values[3] + values[4], 1.0, 1e-10));
    }

    #[test]
    fn test_parser_simplify_vector_and_domain_policy() {
        let graph = Graph::parse_expression("sin(x) * (x + y) + 2", &["x", "y"]).unwrap();
        let (values, jacobian) = graph.value_and_jacobian(&[0.6, 1.4]).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(jacobian[0].len(), 2);

        let mut simplifiable = Graph::new(1);
        let x = simplifiable.input(0);
        let zero = simplifiable.constant(0.0);
        simplifiable.add(x, zero);
        let simplified = simplifiable.simplify().unwrap();
        assert!(approx_eq(simplified.compute(&[3.0]).unwrap(), 3.0, 1e-10));

        let mut strict = Graph::new(1);
        let x = strict.input(0);
        strict.sqrt(x);
        assert!(strict
            .gradient_with_domain_policy(&[0.0], DomainPolicy::StrictDerivative)
            .is_err());
    }

    #[test]
    fn test_compiled_ir_and_batch_match_graph() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        graph.mul(sum, y);

        let compiled = graph.compile_ir().unwrap();
        let value_graph = graph.compute(&[2.0, 3.0]).unwrap();
        let value_ir = compiled.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value_graph, value_ir, 1e-10));

        let (_value_graph, grad_graph) = graph.gradient(&[2.0, 3.0]).unwrap();
        let (_value_ir, grad_ir) = compiled.gradient(&[2.0, 3.0]).unwrap();
        assert_eq!(grad_graph.len(), grad_ir.len());
        for (left, right) in grad_graph.iter().zip(grad_ir.iter()) {
            assert!(approx_eq(*left, *right, 1e-10));
        }

        let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let batch_values = compiled.compute_batch(batch).unwrap();
        assert_eq!(batch_values.batch_size, 2);
        assert_eq!(batch_values.output_dim, 1);
        assert!(approx_eq(
            batch_values.data[0],
            graph.compute(&[2.0, 3.0]).unwrap(),
            1e-10
        ));
        assert!(approx_eq(
            batch_values.data[1],
            graph.compute(&[4.0, 5.0]).unwrap(),
            1e-10
        ));

        let mut values_buffer = BatchValuesBuffer::new();
        compiled
            .compute_batch_into(batch, &mut values_buffer)
            .unwrap();
        assert_eq!(values_buffer.data, batch_values.data);
        let first_capacity = values_buffer.data.capacity();
        compiled
            .compute_batch_into(batch, &mut values_buffer)
            .unwrap();
        assert!(values_buffer.data.capacity() >= first_capacity);

        let batch_grad = graph.gradient_batch(batch).unwrap();
        assert_eq!(batch_grad.batch_size, 2);
        assert_eq!(batch_grad.input_dim, 2);
        assert_eq!(batch_grad.gradients.len(), 4);

        let mut gradients_buffer = BatchGradientsBuffer::new();
        graph
            .gradient_batch_into(batch, &mut gradients_buffer)
            .unwrap();
        assert_eq!(gradients_buffer.values, batch_grad.values);
        assert_eq!(gradients_buffer.gradients, batch_grad.gradients);

        let metadata = graph.compiled_metadata().unwrap();
        assert_eq!(metadata.num_inputs, 2);
        assert_eq!(metadata.num_outputs, 1);
        assert_eq!(metadata.num_instructions, compiled.instructions().len());
        assert_eq!(
            metadata.value_count,
            graph.num_inputs() + compiled.instructions().len()
        );
        assert!(metadata.is_scalar_output);

        let flat = compiled.flat_instructions().unwrap();
        assert_eq!(flat.len(), compiled.instructions().len());
        assert_eq!(compiled.flat_instructions_slice().len(), flat.len());
        assert_eq!(flat[0].opcode, OpCode::Add);
        assert_eq!(flat[0].output, 2);
        assert_eq!(flat[0].left, x);
        assert_eq!(flat[0].right, y);
        assert_eq!(flat[1].opcode, OpCode::Mul);
        assert_eq!(flat[1].value, 0.0);

        let scalar_backend = ScalarBackend;
        compiled
            .validate_backend_capabilities(&scalar_backend.capabilities())
            .unwrap();
        let scalar_value = scalar_backend.compute(&compiled, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(scalar_value, value_ir, 1e-10));
        let mut scalar_values_buffer = BatchValuesBuffer::new();
        scalar_backend
            .compute_batch(&compiled, batch, &mut scalar_values_buffer)
            .unwrap();
        assert_eq!(scalar_values_buffer.data, batch_values.data);

        let simd_backend = SimdBackend;
        if simd_backend.capabilities().supports_batch_compute {
            compiled
                .validate_backend_capabilities(&simd_backend.capabilities())
                .unwrap();
            let mut simd_values_buffer = BatchValuesBuffer::new();
            simd_backend
                .compute_batch(&compiled, batch, &mut simd_values_buffer)
                .unwrap();
            assert_eq!(simd_values_buffer.batch_size, batch.batch_size);
            assert_eq!(simd_values_buffer.output_dim, batch_values.output_dim);
            for (left, right) in simd_values_buffer.data.iter().zip(batch_values.data.iter()) {
                assert!(approx_eq(*left, *right, 1e-10));
            }
            assert!(simd_backend.gradient(&compiled, &[2.0, 3.0]).is_err());
        }

        let limited_backend = BackendCapabilities {
            supports_f64: true,
            supports_f32: false,
            supports_constants: true,
            supports_unary: true,
            supports_binary: true,
            supports_multi_output: true,
            supports_reverse_gradient: true,
            supports_batch_compute: true,
            supports_batch_gradient: true,
            supported_opcodes: vec![OpCode::Add],
        };
        assert!(compiled
            .validate_backend_capabilities(&limited_backend)
            .is_err());

        let (auto_backend, auto_values) = compiled.compute_batch_auto(batch).unwrap();
        assert_eq!(auto_values.data, batch_values.data);
        let mut dispatched_values = BatchValuesBuffer::new();
        auto_backend
            .compute_batch(&compiled, batch, &mut dispatched_values)
            .unwrap();
        assert_eq!(dispatched_values.data, batch_values.data);
        let graph_auto_backend = graph.recommended_batch_compute_backend().unwrap();
        let expected_compute_backend = compiled.recommended_batch_compute_backend();
        assert_eq!(auto_backend, expected_compute_backend);
        assert_eq!(graph_auto_backend, expected_compute_backend);

        let (auto_gradient_backend, auto_gradients) = compiled.gradient_batch_auto(batch).unwrap();
        assert_eq!(auto_gradients.values, batch_grad.values);
        assert_eq!(auto_gradients.gradients, batch_grad.gradients);
        let mut dispatched_gradients = BatchGradientsBuffer::new();
        auto_gradient_backend
            .gradient_batch(&compiled, batch, &mut dispatched_gradients)
            .unwrap();
        assert_eq!(dispatched_gradients.values, batch_grad.values);
        assert_eq!(dispatched_gradients.gradients, batch_grad.gradients);
        let graph_auto_gradient_backend = graph.recommended_batch_gradient_backend().unwrap();
        let expected_gradient_backend = compiled.recommended_batch_gradient_backend();
        assert_eq!(auto_gradient_backend, expected_gradient_backend);
        assert_eq!(graph_auto_gradient_backend, expected_gradient_backend);

        let plan = graph
            .device_batch_plan(expected_compute_backend, batch.batch_size)
            .unwrap();
        assert_eq!(plan.backend, expected_compute_backend);
        assert_eq!(plan.batch_size, batch.batch_size);
        assert_eq!(plan.input_dim, 2);
        assert_eq!(plan.output_dim, 1);
        assert_eq!(plan.buffers.len(), 5);
        assert_eq!(plan.buffer_handles.len(), 5);
        assert!(plan.compute_transfer_plan.is_empty());
        assert!(plan.gradient_transfer_plan.is_empty());
        assert_eq!(plan.buffers[3].kind, crate::DeviceBufferKind::PrimaryValues);
        assert_eq!(plan.buffers[3].len, batch.batch_size);
    }

    #[test]
    fn test_auto_backend_reports_track_supported_simd_ops() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let abs_x = graph.abs(x);
        let shifted = graph.add_const(abs_x, 1.25);
        let exponent = graph.log1p_exp(x);
        let powered = graph.pow(shifted, exponent);
        let output = graph.log_add_exp(powered, x);
        graph.set_output(output).unwrap();
        let compiled = graph.compile_ir().unwrap();
        let batch = BatchInputs::new(&[1.0, 2.0, 3.0], 3, 1).unwrap();

        let expected_compute_backend = compiled.recommended_batch_compute_backend();
        let expected_gradient_backend = compiled.recommended_batch_gradient_backend();
        assert_eq!(
            graph.recommended_batch_compute_backend().unwrap(),
            expected_compute_backend
        );
        assert_eq!(
            graph.recommended_batch_gradient_backend().unwrap(),
            expected_gradient_backend
        );

        let report = graph.simd_support_report().unwrap();
        assert!(matches!(
            report.backend,
            BackendKind::SimdF64x4 | BackendKind::SimdF64x2
        ));
        assert_eq!(report.missing_opcodes, Vec::<OpCode>::new());
        assert_eq!(report.lane_width, report.backend.lane_width());
        assert_eq!(
            report.can_compute_batch(),
            expected_compute_backend != BackendKind::Scalar
        );
        assert_eq!(
            report.can_gradient_batch(),
            expected_gradient_backend != BackendKind::Scalar
        );

        let scalar_report = graph.backend_support_report(BackendKind::Scalar).unwrap();
        assert!(scalar_report.can_compute_batch());
        assert!(scalar_report.can_gradient_batch());
        assert!(scalar_report.runtime_available);
        assert_eq!(scalar_report.lane_width, 1);
        let mock_report = graph
            .backend_support_report(BackendKind::MockDeviceCpu)
            .unwrap();
        assert!(mock_report.can_compute_batch());
        assert_eq!(mock_report.backend.name(), "mock-device-cpu");
        let wgpu_report = graph.backend_support_report(BackendKind::Wgpu).unwrap();
        assert_eq!(wgpu_report.backend.name(), "wgpu");
        assert_eq!(wgpu_report.lane_width, 1);
        let reports = graph.backend_support_reports().unwrap();
        assert_eq!(reports.len(), 5);
        assert_eq!(reports[0].backend, BackendKind::Scalar);
        assert_eq!(reports[1].backend, BackendKind::MockDeviceCpu);
        assert_eq!(reports[2].backend, BackendKind::Wgpu);
        assert_eq!(reports[3].backend, BackendKind::SimdF64x4);
        assert_eq!(reports[4].backend, BackendKind::SimdF64x2);
        let simd_plan = graph
            .device_batch_plan(BackendKind::Scalar, batch.batch_size)
            .unwrap();
        assert_eq!(simd_plan.batch_size, batch.batch_size);
        assert_eq!(simd_plan.input_dim, 1);
        let mock_plan = graph
            .device_batch_plan(BackendKind::MockDeviceCpu, batch.batch_size)
            .unwrap();
        assert_eq!(mock_plan.backend, BackendKind::MockDeviceCpu);
        assert_eq!(
            mock_plan.buffer_handles[0].location,
            crate::DeviceMemoryLocation::Device
        );
        assert_eq!(mock_plan.compute_transfer_plan.len(), 2);
        assert_eq!(mock_plan.gradient_transfer_plan.len(), 3);
        let wgpu_plan = graph
            .device_batch_plan(BackendKind::Wgpu, batch.batch_size)
            .unwrap();
        assert_eq!(wgpu_plan.backend, BackendKind::Wgpu);
        assert_eq!(
            wgpu_plan.buffer_handles[0].location,
            crate::DeviceMemoryLocation::Device
        );
        assert_eq!(wgpu_plan.compute_transfer_plan.len(), 2);
        assert_eq!(wgpu_plan.gradient_transfer_plan.len(), 3);

        let (value_backend, values) = graph.compute_batch_auto(batch).unwrap();
        assert_eq!(value_backend, expected_compute_backend);
        assert_eq!(values.data, graph.compute_batch(batch).unwrap().data);

        let mut gradients = BatchGradientsBuffer::new();
        let gradient_backend = graph
            .gradient_batch_auto_into(batch, &mut gradients)
            .unwrap();
        assert_eq!(gradient_backend, expected_gradient_backend);
        assert_eq!(
            gradients.gradients,
            graph.gradient_batch(batch).unwrap().gradients
        );
    }

    #[test]
    fn test_mock_device_buffers_execute_transfer_plans() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let product = graph.mul(x, y);
        graph.set_output(product).unwrap();
        let compiled = graph.compile_ir().unwrap();
        let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 3, 2).unwrap();
        let mock = crate::MockDeviceBackend;
        let mut buffers = mock.allocate_batch_buffers(&compiled, batch.batch_size);
        assert_eq!(buffers.plan().backend, BackendKind::MockDeviceCpu);
        let input_buffer = buffers.buffer(crate::DeviceBufferKind::Inputs).unwrap();
        assert_eq!(input_buffer.data().len(), batch.data.len());
        assert_eq!(input_buffer.handle().kind, crate::DeviceBufferKind::Inputs);
        assert_eq!(buffers.buffers().len(), buffers.plan().buffers.len());
        assert!(buffers
            .upload(crate::DeviceBufferKind::Inputs, &[1.0])
            .is_err());

        let scalar_plan = compiled.device_batch_plan(BackendKind::Scalar, batch.batch_size);
        let mut scalar_buffers = crate::DeviceBufferSet::new(scalar_plan);
        let mut rejected_values = BatchValuesBuffer::new();
        assert!(mock
            .compute_batch_with_buffers(&compiled, batch, &mut scalar_buffers, &mut rejected_values)
            .is_err());
        let wrong_batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        assert!(mock
            .compute_batch_with_buffers(&compiled, wrong_batch, &mut buffers, &mut rejected_values)
            .is_err());

        let mut values = BatchValuesBuffer::new();
        let trace = compiled
            .compute_batch_mock_device_into(batch, &mut buffers, &mut values)
            .unwrap();
        assert_eq!(trace.mode, crate::DeviceExecutionMode::ComputeBatch);
        assert_eq!(trace.transfers, buffers.plan().compute_transfer_plan);
        assert!(!trace.used_native_kernel);
        assert_eq!(values.data, compiled.compute_batch(batch).unwrap().data);
        assert_eq!(
            buffers.download(crate::DeviceBufferKind::Outputs).unwrap(),
            values.data
        );

        let mut graph_buffers = graph
            .allocate_mock_device_buffers(batch.batch_size)
            .unwrap();
        let mut graph_values = BatchValuesBuffer::new();
        let graph_trace = graph
            .compute_batch_mock_device_into(batch, &mut graph_buffers, &mut graph_values)
            .unwrap();
        assert_eq!(graph_trace.mode, crate::DeviceExecutionMode::ComputeBatch);
        assert!(!graph_trace.used_native_kernel);
        assert_eq!(graph_values.data, values.data);

        let mut gradients = BatchGradientsBuffer::new();
        let trace = compiled
            .gradient_batch_mock_device_into(batch, &mut buffers, &mut gradients)
            .unwrap();
        assert_eq!(trace.mode, crate::DeviceExecutionMode::GradientBatch);
        assert_eq!(trace.transfers, buffers.plan().gradient_transfer_plan);
        assert!(!trace.used_native_kernel);
        let scalar_gradients = compiled.gradient_batch(batch).unwrap();
        assert_eq!(gradients.values, scalar_gradients.values);
        assert_eq!(gradients.gradients, scalar_gradients.gradients);
        assert_eq!(
            buffers
                .download(crate::DeviceBufferKind::PrimaryValues)
                .unwrap(),
            gradients.values
        );

        let boundary = crate::GpuBackendBoundary::new(
            crate::AcceleratorDeviceContext::mock_cpu(),
            crate::DeviceTransferPolicy::Explicit,
        );
        assert!(boundary.unsupported_execution_error::<()>().is_err());
        let cuda = crate::AcceleratorDeviceContext::cuda(2);
        assert_eq!(cuda.kind, crate::AcceleratorDeviceKind::Cuda);
        assert_eq!(cuda.name, "cuda:2");
        let wgpu = crate::AcceleratorDeviceContext::wgpu(1);
        assert_eq!(wgpu.kind, crate::AcceleratorDeviceKind::Wgpu);
        assert_eq!(wgpu.name, "wgpu:1");
    }

    #[cfg(feature = "backend-wgpu")]
    #[test]
    fn test_wgpu_backend_skeleton_buffers_and_execution() {
        let boundary = crate::GpuBackendBoundary::new(
            crate::AcceleratorDeviceContext::wgpu(0),
            crate::DeviceTransferPolicy::Explicit,
        );
        let backend = match boundary.initialize_wgpu() {
            Ok(backend) => backend,
            Err(_) => return,
        };
        assert_eq!(backend.context().kind, crate::AcceleratorDeviceKind::Wgpu);
        assert_eq!(
            backend.transfer_policy(),
            crate::DeviceTransferPolicy::Explicit
        );
        assert!(!backend.adapter_name().is_empty());
        assert_eq!(
            crate::WgpuBackend::native_batch_compute_supported_opcodes(),
            crate::WGPU_NATIVE_BATCH_COMPUTE_EXACT_SAFE_OPCODES
        );

        let mut exact_graph = Graph::new(1);
        let x = exact_graph.input(0);
        let neg_x = exact_graph.neg(x);
        let relu_neg_x = exact_graph.relu(neg_x);
        let abs_x = exact_graph.abs(x);
        exact_graph.set_outputs(&[relu_neg_x, abs_x]).unwrap();
        let compiled = exact_graph.compile_ir().unwrap();
        let batch = BatchInputs::new(&[2.0, -3.0, -0.0], 3, 1).unwrap();
        assert!(compiled.supports_native_wgpu_batch_compute(&backend));
        assert!(compiled.supports_native_wgpu_batch_compute_for_batch(&backend, batch));
        assert!(exact_graph
            .supports_native_wgpu_batch_compute(&backend)
            .unwrap());
        assert!(exact_graph
            .supports_native_wgpu_batch_compute_for_batch(&backend, batch)
            .unwrap());

        let mut buffers = backend
            .allocate_batch_buffers(&compiled, batch.batch_size)
            .unwrap();
        assert_eq!(buffers.plan().backend, BackendKind::Wgpu);
        assert_eq!(buffers.buffers().len(), buffers.plan().buffers.len());
        assert_eq!(
            buffers
                .buffer(crate::DeviceBufferKind::Inputs)
                .unwrap()
                .handle()
                .location,
            crate::DeviceMemoryLocation::Device
        );
        assert!(buffers
            .upload(&backend, crate::DeviceBufferKind::Inputs, &[1.0])
            .is_err());
        buffers
            .upload(&backend, crate::DeviceBufferKind::Inputs, batch.data)
            .unwrap();
        assert_eq!(
            buffers
                .download(&backend, crate::DeviceBufferKind::Inputs)
                .unwrap(),
            batch.data
        );

        let mut values = BatchValuesBuffer::new();
        let trace = compiled
            .compute_batch_wgpu_into(&backend, batch, &mut buffers, &mut values)
            .unwrap();
        assert_eq!(trace.backend, BackendKind::Wgpu);
        assert_eq!(trace.mode, crate::DeviceExecutionMode::ComputeBatch);
        assert_eq!(trace.transfers, buffers.plan().compute_transfer_plan);
        assert!(trace.used_native_kernel);
        assert_eq!(values.data, compiled.compute_batch(batch).unwrap().data);
        assert_eq!(
            buffers
                .download(&backend, crate::DeviceBufferKind::Outputs)
                .unwrap(),
            values.data
        );

        let mut gradients = BatchGradientsBuffer::new();
        let gradient_trace = compiled
            .gradient_batch_wgpu_into(&backend, batch, &mut buffers, &mut gradients)
            .unwrap();
        assert_eq!(gradient_trace.backend, BackendKind::Wgpu);
        assert_eq!(
            gradient_trace.mode,
            crate::DeviceExecutionMode::GradientBatch
        );
        assert!(!gradient_trace.used_native_kernel);
        let scalar_gradients = compiled.gradient_batch(batch).unwrap();
        assert_eq!(gradients.values, scalar_gradients.values);
        assert_eq!(gradients.gradients, scalar_gradients.gradients);

        let mut graph_buffers = exact_graph
            .allocate_wgpu_buffers(&backend, batch.batch_size)
            .unwrap();
        let mut graph_values = BatchValuesBuffer::new();
        let graph_trace = exact_graph
            .compute_batch_wgpu_into(&backend, batch, &mut graph_buffers, &mut graph_values)
            .unwrap();
        assert!(graph_trace.used_native_kernel);
        assert_eq!(graph_values.data, values.data);

        let inexact_batch = BatchInputs::new(&[0.1, -0.2, 1.5], 3, 1).unwrap();
        assert!(compiled.supports_native_wgpu_batch_compute(&backend));
        assert!(!compiled.supports_native_wgpu_batch_compute_for_batch(&backend, inexact_batch));
        assert!(!exact_graph
            .supports_native_wgpu_batch_compute_for_batch(&backend, inexact_batch)
            .unwrap());
        let inexact_scalar = compiled.compute_batch(inexact_batch).unwrap();
        let mut inexact_values = BatchValuesBuffer::new();
        let inexact_trace = compiled
            .compute_batch_wgpu_into(&backend, inexact_batch, &mut buffers, &mut inexact_values)
            .unwrap();
        assert!(!inexact_trace.used_native_kernel);
        assert_eq!(inexact_values.data, inexact_scalar.data);

        let mut fallback_graph = Graph::new(2);
        let left = fallback_graph.input(0);
        let right = fallback_graph.input(1);
        let product = fallback_graph.mul(left, right);
        fallback_graph.set_output(product).unwrap();
        let fallback_compiled = fallback_graph.compile_ir().unwrap();
        let fallback_batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 3, 2).unwrap();
        assert!(!fallback_compiled.supports_native_wgpu_batch_compute(&backend));
        assert!(!fallback_graph
            .supports_native_wgpu_batch_compute(&backend)
            .unwrap());
        let mut fallback_buffers = fallback_graph
            .allocate_wgpu_buffers(&backend, fallback_batch.batch_size)
            .unwrap();
        let mut fallback_values = BatchValuesBuffer::new();
        let fallback_trace = fallback_graph
            .compute_batch_wgpu_into(
                &backend,
                fallback_batch,
                &mut fallback_buffers,
                &mut fallback_values,
            )
            .unwrap();
        assert!(!fallback_trace.used_native_kernel);
        assert_eq!(
            fallback_values.data,
            fallback_compiled
                .compute_batch(fallback_batch)
                .unwrap()
                .data
        );
    }

    #[test]
    fn test_simd_scalar_unary_fallbacks_and_abs_match_scalar_when_available() {
        let compiled = {
            let mut graph = Graph::new(1);
            let x = graph.input(0);
            let sin_x = graph.sin(x);
            let cos_sin = graph.cos(sin_x);
            let exp_cos = graph.exp(cos_sin);
            let shifted = graph.add_const(exp_cos, 1.0);
            let ln_shifted = graph.ln(shifted);
            let tan_ln = graph.tan(ln_shifted);
            let tanh_tan = graph.tanh(tan_ln);
            let abs_tanh = graph.abs(tanh_tan);
            graph.set_output(abs_tanh).unwrap();
            graph.compile_ir().unwrap()
        };
        let batch = BatchInputs::new(&[-0.5, -0.0, 0.2, 0.7, 1.1], 5, 1).unwrap();
        let scalar_values = compiled.compute_batch(batch).unwrap();
        let scalar_gradients = compiled.gradient_batch(batch).unwrap();

        for backend in [BackendKind::SimdF64x4, BackendKind::SimdF64x2] {
            let simd_report = compiled.backend_support_report(backend).unwrap();
            assert!(simd_report.missing_opcodes.is_empty());
            if simd_report.can_compute_batch() {
                let mut values = BatchValuesBuffer::new();
                backend
                    .compute_batch(&compiled, batch, &mut values)
                    .unwrap();
                assert_eq!(values.data.len(), scalar_values.data.len());
                for (left, right) in values.data.iter().zip(scalar_values.data.iter()) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
            }
            if simd_report.can_gradient_batch() {
                let mut gradients = BatchGradientsBuffer::new();
                backend
                    .gradient_batch(&compiled, batch, &mut gradients)
                    .unwrap();
                assert_eq!(gradients.values.len(), scalar_gradients.values.len());
                assert_eq!(gradients.gradients.len(), scalar_gradients.gradients.len());
                for (left, right) in gradients.values.iter().zip(scalar_gradients.values.iter()) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
                for (left, right) in gradients
                    .gradients
                    .iter()
                    .zip(scalar_gradients.gradients.iter())
                {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
            }
        }
    }

    #[test]
    fn test_simd_scalar_binary_fallbacks_match_scalar_when_available() {
        let compiled = {
            let mut graph = Graph::new(2);
            let x = graph.input(0);
            let y = graph.input(1);
            let abs_x = graph.abs(x);
            let base = graph.add_const(abs_x, 1.25);
            let exponent = graph.log1p_exp(y);
            let powered = graph.pow(base, exponent);
            let mixed = graph.log_add_exp(powered, y);
            let output = graph.tanh(mixed);
            graph.set_output(output).unwrap();
            graph.compile_ir().unwrap()
        };
        let batch = BatchInputs::new(
            &[-0.5, -0.3, -0.0, 0.0, 0.2, 0.4, 0.7, -0.2, 1.1, 0.3],
            5,
            2,
        )
        .unwrap();
        let scalar_values = compiled.compute_batch(batch).unwrap();
        let scalar_gradients = compiled.gradient_batch(batch).unwrap();

        for backend in [BackendKind::SimdF64x4, BackendKind::SimdF64x2] {
            let simd_report = compiled.backend_support_report(backend).unwrap();
            assert!(simd_report.missing_opcodes.is_empty());
            if simd_report.can_compute_batch() {
                let mut values = BatchValuesBuffer::new();
                backend
                    .compute_batch(&compiled, batch, &mut values)
                    .unwrap();
                assert_eq!(values.data.len(), scalar_values.data.len());
                for (left, right) in values.data.iter().zip(scalar_values.data.iter()) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
            }
            if simd_report.can_gradient_batch() {
                let mut gradients = BatchGradientsBuffer::new();
                backend
                    .gradient_batch(&compiled, batch, &mut gradients)
                    .unwrap();
                assert_eq!(gradients.values.len(), scalar_gradients.values.len());
                assert_eq!(gradients.gradients.len(), scalar_gradients.gradients.len());
                for (left, right) in gradients.values.iter().zip(scalar_gradients.values.iter()) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
                for (left, right) in gradients
                    .gradients
                    .iter()
                    .zip(scalar_gradients.gradients.iter())
                {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
            }
        }
    }

    #[test]
    fn test_simd_backend_batch_compute_multi_output_and_fallback_support() {
        let simd_backend = SimdBackend;
        if !simd_backend.capabilities().supports_batch_compute {
            return;
        }

        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let product = graph.mul(x, y);
        let ratio = graph.div(x, y);
        let difference = graph.sub(x, y);
        let relu = graph.relu(difference);
        let sum = graph.add(x, y);
        let sqrt = graph.sqrt(sum);
        graph.set_outputs(&[product, ratio, relu, sqrt]).unwrap();
        let compiled = graph.compile_ir().unwrap();
        let batch = BatchInputs::new(&[2.0, 1.0, 3.0, 1.5, 4.0, 2.0], 3, 2).unwrap();
        let scalar_values = compiled.compute_batch(batch).unwrap();
        let mut simd_values = BatchValuesBuffer::new();
        simd_backend
            .compute_batch(&compiled, batch, &mut simd_values)
            .unwrap();
        assert_eq!(simd_values.batch_size, 3);
        assert_eq!(simd_values.output_dim, 4);
        for (left, right) in simd_values.data.iter().zip(scalar_values.data.iter()) {
            assert!(approx_eq(*left, *right, 1e-10));
        }

        let scalar_gradients = compiled.gradient_batch(batch).unwrap();
        let mut simd_gradients = BatchGradientsBuffer::new();
        simd_backend
            .gradient_batch(&compiled, batch, &mut simd_gradients)
            .unwrap();
        assert_eq!(simd_gradients.batch_size, 3);
        assert_eq!(simd_gradients.input_dim, 2);
        for (left, right) in simd_gradients
            .values
            .iter()
            .zip(scalar_gradients.values.iter())
        {
            assert!(approx_eq(*left, *right, 1e-10));
        }
        for (left, right) in simd_gradients
            .gradients
            .iter()
            .zip(scalar_gradients.gradients.iter())
        {
            assert!(approx_eq(*left, *right, 1e-10));
        }

        let mut fallback_graph = Graph::new(1);
        let z = fallback_graph.input(0);
        let abs_z = fallback_graph.abs(z);
        let shifted = fallback_graph.add_const(abs_z, 1.25);
        let exponent = fallback_graph.log1p_exp(z);
        let powered = fallback_graph.pow(shifted, exponent);
        let output = fallback_graph.log_add_exp(powered, z);
        fallback_graph.set_output(output).unwrap();
        let fallback_compiled = fallback_graph.compile_ir().unwrap();
        assert!(fallback_compiled
            .validate_backend_capabilities(&simd_backend.capabilities())
            .is_ok());
    }

    #[test]
    fn test_simd_relu_matches_scalar_edge_cases() {
        let simd_backend = SimdBackend;
        if !simd_backend.capabilities().supports_batch_compute {
            return;
        }

        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.relu(x);
        let compiled = graph.compile_ir().unwrap();
        let inputs = [f64::NAN, -0.0, 0.0, -2.0, 3.0];
        let batch = BatchInputs::new(&inputs, inputs.len(), 1).unwrap();

        let scalar_values = compiled.compute_batch(batch).unwrap();
        let mut simd_values = BatchValuesBuffer::new();
        simd_backend
            .compute_batch(&compiled, batch, &mut simd_values)
            .unwrap();
        assert_eq!(simd_values.data.len(), scalar_values.data.len());
        for (left, right) in simd_values.data.iter().zip(scalar_values.data.iter()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }

        let scalar_gradients = compiled.gradient_batch(batch).unwrap();
        let mut simd_gradients = BatchGradientsBuffer::new();
        simd_backend
            .gradient_batch(&compiled, batch, &mut simd_gradients)
            .unwrap();
        for (left, right) in simd_gradients
            .values
            .iter()
            .zip(scalar_gradients.values.iter())
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
        for (left, right) in simd_gradients
            .gradients
            .iter()
            .zip(scalar_gradients.gradients.iter())
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn test_simd_gradient_masks_inactive_nan_lanes() {
        let simd_backend = SimdBackend;
        if !simd_backend.capabilities().supports_batch_gradient {
            return;
        }

        let mut relu_sqrt_graph = Graph::new(1);
        let x = relu_sqrt_graph.input(0);
        let sqrt_x = relu_sqrt_graph.sqrt(x);
        relu_sqrt_graph.relu(sqrt_x);
        let compiled = relu_sqrt_graph.compile_ir().unwrap();
        let inputs = [-1.0, 4.0, -9.0];
        let batch = BatchInputs::new(&inputs, inputs.len(), 1).unwrap();
        let scalar = compiled.gradient_batch(batch).unwrap();
        let mut simd = BatchGradientsBuffer::new();
        simd_backend
            .gradient_batch(&compiled, batch, &mut simd)
            .unwrap();
        for (left, right) in simd.gradients.iter().zip(scalar.gradients.iter()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }

        let mut unused_div_graph = Graph::new(1);
        let x = unused_div_graph.input(0);
        let zero = unused_div_graph.constant(0.0);
        unused_div_graph.div(x, zero);
        unused_div_graph.set_output(x).unwrap();
        let compiled = unused_div_graph.compile_ir().unwrap();
        let inputs = [0.0, 1.0, -1.0];
        let batch = BatchInputs::new(&inputs, inputs.len(), 1).unwrap();
        let scalar = compiled.gradient_batch(batch).unwrap();
        let mut simd = BatchGradientsBuffer::new();
        simd_backend
            .gradient_batch(&compiled, batch, &mut simd)
            .unwrap();
        assert_eq!(simd.values, scalar.values);
        assert_eq!(simd.gradients, scalar.gradients);
    }

    #[test]
    fn test_reductions_losses_and_parameters() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mark_parameter(x).unwrap();
        graph.set_parameter_name(y, "weight").unwrap();
        assert_eq!(graph.parameters(), &[x, y]);
        assert_eq!(graph.parameter_name(y), Some("weight"));
        assert_eq!(graph.parameter_names().len(), 1);

        let dot = graph.dot(&[x, y], &[x, y]).unwrap();
        graph.set_output(dot).unwrap();
        let (_value, parameter_grad) = graph.gradient(&[3.0, 4.0]).unwrap();
        let extracted = graph.parameter_gradient(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(extracted[0].1, parameter_grad[0], 1e-10));
        assert!(approx_eq(extracted[1].1, parameter_grad[1], 1e-10));

        let mut loss_graph = Graph::new(2);
        let prediction = loss_graph.input(0);
        let target = loss_graph.input(1);
        let mse = loss_graph.mse_loss(&[prediction], &[target]).unwrap();
        loss_graph.set_output(mse).unwrap();
        assert!(approx_eq(
            loss_graph.compute(&[5.0, 3.0]).unwrap(),
            4.0,
            1e-10
        ));
    }

    #[test]
    fn test_stable_math_regressions() {
        let mut softplus_graph = Graph::new(1);
        let x = softplus_graph.input(0);
        softplus_graph.log1p_exp(x);
        let (value, gradient) = softplus_graph.gradient(&[0.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::LN_2, 1e-12));
        assert!(approx_eq(gradient[0], 0.5, 1e-12));
        assert!(softplus_graph.compute(&[1000.0]).unwrap().is_finite());

        let mut public_softplus_graph = Graph::new(1);
        let x = public_softplus_graph.input(0);
        public_softplus_graph.softplus(x);
        assert!(public_softplus_graph
            .compute(&[1000.0])
            .unwrap()
            .is_finite());

        let mut logsumexp_graph = Graph::new(2);
        let a = logsumexp_graph.input(0);
        let b = logsumexp_graph.input(1);
        let lse = logsumexp_graph.logsumexp_approx(&[a, b]).unwrap();
        logsumexp_graph.set_output(lse).unwrap();
        let lse_value = logsumexp_graph.compute(&[1000.0, 1001.0]).unwrap();
        let expected = 1001.0 + (-1.0_f64).exp().ln_1p();
        assert!(lse_value.is_finite());
        assert!(approx_eq(lse_value, expected, 1e-10));

        let mut softmax_graph = Graph::new(2);
        let a = softmax_graph.input(0);
        let b = softmax_graph.input(1);
        let softmax = softmax_graph.stable_softmax_approx(&[a, b]);
        softmax_graph.set_outputs(&softmax).unwrap();
        let values = softmax_graph.compute_many(&[1000.0, 1001.0]).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(approx_eq(values[0] + values[1], 1.0, 1e-12));
    }

    #[test]
    fn test_softplus_fr_rf_hessian_lowering() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.softplus(x);

        for input in [0.0, 1000.0] {
            let native = graph.exact_hessian_rr(&[input]).unwrap();
            let fr = graph.exact_hessian_fr(&[input]).unwrap();
            let rf = graph.exact_hessian_rf(&[input]).unwrap();

            assert!(native[0][0].is_finite());
            assert!(fr[0][0].is_finite());
            assert!(rf[0][0].is_finite());
            assert!(approx_eq(fr[0][0], native[0][0], 1e-12));
            assert!(approx_eq(rf[0][0], native[0][0], 1e-12));
        }

        let at_zero = graph.exact_hessian_rr(&[0.0]).unwrap();
        let at_large = graph.exact_hessian_rr(&[1000.0]).unwrap();
        assert!(approx_eq(at_zero[0][0], 0.25, 1e-12));
        assert!(approx_eq(at_large[0][0], 0.0, 1e-12));
    }

    #[test]
    fn test_batch_try_row_checks_bounds() {
        let batch = BatchInputs::new(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(batch.row(0), &[1.0, 2.0]);
        assert_eq!(batch.try_row(0).unwrap(), &[1.0, 2.0]);
        assert!(batch.try_row(2).is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_graph_serde_round_trip() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.square(x);
        let encoded = serde_json::to_string(&graph).unwrap();
        let decoded: Graph = serde_json::from_str(&encoded).unwrap();
        assert!(approx_eq(decoded.compute(&[3.0]).unwrap(), 9.0, 1e-10));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_graph_serde_accepts_missing_parameter_metadata() {
        let encoded = r#"{
            "num_inputs": 1,
            "nodes": [{"Operation": {"op": "Sin", "inputs": [0]}}],
            "output_nodes": [],
            "input_names": [null],
            "output_names": []
        }"#;
        let decoded: Graph = serde_json::from_str(encoded).unwrap();
        assert!(decoded.parameters().is_empty());
        assert!(approx_eq(
            decoded.compute(&[0.5]).unwrap(),
            0.5_f64.sin(),
            1e-10
        ));
    }

    // ====================
    // Additional coverage tests for graph.rs uncovered areas
    // ====================

    #[test]
    fn test_graph_cos_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.cos(x);
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 1.0, 1e-10));

        let (_v, grad) = graph.gradient(&[0.0]).unwrap();
        assert!(approx_eq(grad[0], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_tan_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.tan(x);
        let value = graph.compute(&[0.5]).unwrap();
        assert!(approx_eq(value, 0.5_f64.tan(), 1e-10));

        let (_v, grad) = graph.gradient(&[0.5]).unwrap();
        let expected_grad = 1.0 / 0.5_f64.cos().powi(2);
        assert!(approx_eq(grad[0], expected_grad, 1e-8));
    }

    #[test]
    fn test_graph_neg_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.neg(x);
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, -3.0, 1e-10));

        let (_v, grad) = graph.gradient(&[3.0]).unwrap();
        assert!(approx_eq(grad[0], -1.0, 1e-10));
    }

    #[test]
    fn test_graph_exp_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);
        let value = graph.compute(&[1.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::E, 1e-10));

        let (_v, grad) = graph.gradient(&[1.0]).unwrap();
        assert!(approx_eq(grad[0], std::f64::consts::E, 1e-8));
    }

    #[test]
    fn test_graph_ln_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.ln(x);
        let value = graph.compute(&[std::f64::consts::E]).unwrap();
        assert!(approx_eq(value, 1.0, 1e-10));

        let (_v, grad) = graph.gradient(&[std::f64::consts::E]).unwrap();
        assert!(approx_eq(grad[0], 1.0 / std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_graph_sqrt_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sqrt(x);
        let value = graph.compute(&[4.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));

        let (_v, grad) = graph.gradient(&[4.0]).unwrap();
        assert!(approx_eq(grad[0], 0.25, 1e-10));
    }

    #[test]
    fn test_graph_abs_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.abs(x);
        let value = graph.compute(&[-3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));

        let (_v, grad) = graph.gradient(&[5.0]).unwrap();
        assert!(approx_eq(grad[0], 1.0, 1e-10));
    }

    #[test]
    fn test_graph_sub_node() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.sub(x, y);
        let value = graph.compute(&[5.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));

        let (_v, grad) = graph.gradient(&[5.0, 3.0]).unwrap();
        assert!(approx_eq(grad[0], 1.0, 1e-10));
        assert!(approx_eq(grad[1], -1.0, 1e-10));
    }

    #[test]
    fn test_graph_div_node() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.div(x, y);
        let value = graph.compute(&[6.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));

        let (_v, grad) = graph.gradient(&[6.0, 3.0]).unwrap();
        assert!(approx_eq(grad[0], 1.0 / 3.0, 1e-10));
        assert!(approx_eq(grad[1], -2.0 / 3.0, 1e-10));
    }

    #[test]
    fn test_graph_tanh_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.tanh(x);
        let value = graph.compute(&[0.5]).unwrap();
        assert!(approx_eq(value, 0.5_f64.tanh(), 1e-10));
    }

    #[test]
    fn test_graph_relu_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.relu(x);

        let pos = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(pos, 3.0, 1e-10));

        let neg = graph.compute(&[-2.0]).unwrap();
        assert!(approx_eq(neg, 0.0, 1e-10));

        let (_v, grad) = graph.gradient(&[3.0]).unwrap();
        assert!(approx_eq(grad[0], 1.0, 1e-10));
    }

    #[test]
    fn test_graph_log1p_exp_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.log1p_exp(x);
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::LN_2, 1e-10));
    }

    #[test]
    fn test_graph_log_add_exp_node() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.log_add_exp(x, y);
        let value = graph.compute(&[1.0, 2.0]).unwrap();
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_graph_set_output_errors() {
        let mut graph = Graph::new(1);
        // Empty graph, no nodes - set_output should error
        let result = graph.set_output(0);
        assert!(result.is_ok()); // Input 0 is valid

        let result = graph.set_output(1);
        assert!(result.is_err()); // No nodes, so index 1 is out of bounds
    }

    #[test]
    fn test_graph_set_outputs_errors() {
        let mut graph = Graph::new(1);
        let result = graph.set_outputs(&[1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_add_output() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        let product = graph.mul(x, y);

        graph.add_output(sum).unwrap();
        graph.add_output(product).unwrap();
        assert_eq!(graph.output_nodes(), &[sum, product]);

        // Error: out of bounds
        assert!(graph.add_output(10).is_err());
    }

    #[test]
    fn test_graph_add_output_error_empty_graph() {
        let mut graph = Graph::new(0);
        assert!(graph.add_output(0).is_err());
    }

    #[test]
    fn test_graph_clear_output() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let _y = graph.mul(x, x);
        graph.set_output(x).unwrap();
        assert_eq!(graph.output_node(), Some(x));

        graph.clear_output();
        assert!(graph.output_node().is_none());

        // After clear, effective_output falls back to last node
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 9.0, 1e-10));
    }

    #[test]
    fn test_graph_effective_output_node_fallback() {
        // No stored nodes, no explicit output
        let graph = Graph::new(2);
        let effective = graph.effective_output_node();
        // Should fall back to last input
        assert_eq!(effective, Some(1));
    }

    #[test]
    fn test_graph_effective_output_nodes_empty() {
        let graph = Graph::new(2);
        let nodes = graph.effective_output_nodes();
        assert_eq!(nodes, vec![1]);
    }

    #[test]
    fn test_graph_is_empty_and_len() {
        let mut graph = Graph::new(2);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);

        let x = graph.input(0);
        graph.sin(x);
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_graph_next_node_id() {
        let mut graph = Graph::new(2);
        assert_eq!(graph.next_node_id(), 2);

        let x = graph.input(0);
        graph.sin(x);
        assert_eq!(graph.next_node_id(), 3);
    }

    #[test]
    fn test_graph_nodes_access() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sin(x);
        assert_eq!(graph.nodes().len(), 1);
    }

    #[test]
    fn test_graph_num_inputs() {
        let graph = Graph::new(3);
        assert_eq!(graph.num_inputs(), 3);
    }

    #[test]
    fn test_graph_compute_checked_valid() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);
        let value = graph.compute_checked(&[1.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_graph_gradient_checked_valid() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);
        let (value, grad) = graph.gradient_checked(&[1.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::E, 1e-10));
        assert!(approx_eq(grad[0], std::f64::consts::E, 1e-8));
    }

    #[test]
    fn test_graph_compute_many_checked_valid() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.set_outputs(&[x, y]).unwrap();
        let values = graph.compute_many_checked(&[1.0, 2.0]).unwrap();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_graph_jacobian_checked_valid() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.set_outputs(&[x, y]).unwrap();
        let jacobian = graph.jacobian_checked(&[1.0, 2.0]).unwrap();
        assert_eq!(jacobian.len(), 2);
        assert!(approx_eq(jacobian[0][0], 1.0, 1e-10));
        assert!(approx_eq(jacobian[0][1], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_value_and_jacobian() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        graph.set_output(sum).unwrap();

        let (values, jacobian) = graph.value_and_jacobian(&[2.0, 3.0]).unwrap();
        assert_eq!(values.len(), 1);
        assert!(approx_eq(values[0], 5.0, 1e-10));
        assert_eq!(jacobian.len(), 1);
    }

    #[test]
    fn test_graph_value_and_jacobian_checked() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let sum = graph.add(x, y);
        graph.set_output(sum).unwrap();

        let (values, _jacobian) = graph.value_and_jacobian_checked(&[2.0, 3.0]).unwrap();
        assert_eq!(values.len(), 1);
        assert!(approx_eq(values[0], 5.0, 1e-10));
    }

    #[test]
    fn test_graph_compile_ir_valid() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);

        let compiled = graph.compile_ir().unwrap();
        assert_eq!(compiled.num_inputs(), 2);
        assert_eq!(compiled.instructions().len(), 1);
    }

    #[test]
    fn test_graph_compile_accelerated() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);

        let compiled = graph.compile_accelerated().unwrap();
        let value = compiled.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 1.0, 1e-10));
    }

    #[test]
    fn test_graph_try_compile() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sin(x);

        let tape = graph.try_compile().unwrap();
        let value = tape.compute(&[0.5]).unwrap();
        assert!(approx_eq(value, 0.5_f64.sin(), 1e-10));
    }

    #[test]
    fn test_graph_try_compile_invalid() {
        let mut graph = Graph::new(1);
        graph.push_operation(MultiAD::Sin, vec![5]); // Bad index
        assert!(graph.try_compile().is_err());
    }

    #[test]
    fn test_graph_from_operations_fallback() {
        // Use an invalid operation to trigger the fallback path in from_operations
        let ops = [(MultiAD::Sin, vec![0])]; // Valid, should not trigger fallback
        let graph = Graph::from_operations(1, &ops);
        let value = graph.compute(&[0.5]).unwrap();
        assert!(approx_eq(value, 0.5_f64.sin(), 1e-10));
    }

    #[test]
    fn test_graph_try_from_operations_invalid_input_marker() {
        let ops = [(MultiAD::Inp, vec![5])]; // Index out of bounds
        assert!(Graph::try_from_operations(2, &ops).is_err());
    }

    #[test]
    fn test_graph_simplify_mul_identity() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let one = graph.constant(1.0);
        graph.mul(x, one);

        let simplified = graph.simplify().unwrap();
        let value = simplified.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_graph_simplify_mul_left_identity() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let one = graph.constant(1.0);
        graph.mul(one, x);

        let simplified = graph.simplify().unwrap();
        let value = simplified.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_graph_simplify_add_right_zero() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let zero = graph.constant(0.0);
        graph.add(x, zero);

        let simplified = graph.simplify().unwrap();
        let value = simplified.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_graph_simplify_constant_fold() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let two = graph.constant(2.0);
        let three = graph.constant(3.0);
        let sum = graph.add(two, three);
        graph.mul(x, sum);

        let simplified = graph.simplify().unwrap();
        let value = simplified.compute(&[2.0]).unwrap();
        assert!(approx_eq(value, 10.0, 1e-10));
    }

    #[test]
    fn test_graph_set_input_name_errors() {
        let mut graph = Graph::new(2);
        assert!(graph.set_input_name(5, "z").is_err());
        assert!(graph.set_input_name(0, "x").is_ok());
        assert_eq!(graph.input_name(0), Some("x"));
        assert_eq!(graph.input_name(1), None);
    }

    #[test]
    fn test_graph_mark_parameter_errors() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        assert!(graph.mark_parameter(5).is_err());
        assert!(graph.mark_parameter(x).is_ok());
        // Duplicate mark should be ok
        assert!(graph.mark_parameter(x).is_ok());
        assert_eq!(graph.parameters().len(), 1);
    }

    #[test]
    fn test_graph_mark_parameter_empty_graph() {
        let mut graph = Graph::new(0);
        assert!(graph.mark_parameter(0).is_err());
    }

    #[test]
    fn test_graph_validate_bad_parameter() {
        let mut graph = Graph::new(1);
        graph.parameters = vec![5];
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_graph_set_output_name() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let y = graph.sin(x);
        graph.set_output_name(y, "sin_x").unwrap();
        assert!(graph.to_mermaid().contains("sin_x"));
        assert!(graph.to_dot().contains("sin_x"));
    }

    #[test]
    fn test_graph_input_name_out_of_bounds() {
        let graph = Graph::new(1);
        assert_eq!(graph.input_name(5), None);
    }

    #[test]
    fn test_graph_parameter_gradient_non_input_errors() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let sin_x = graph.sin(x);
        graph.mark_parameter(sin_x).unwrap();
        assert!(graph.parameter_gradient(&[1.0]).is_err());
    }

    #[test]
    fn test_graph_stats_constants_and_depth() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let c = graph.constant(2.0);
        graph.mul(x, c);

        let stats = graph.stats();
        assert_eq!(stats.num_inputs, 1);
        assert_eq!(stats.num_constants, 1);
        assert_eq!(stats.num_ops, 1);
        assert_eq!(stats.num_edges, 2);
        assert_eq!(stats.max_depth, 1);
        assert_eq!(stats.op_counts.len(), 1);
    }

    #[test]
    fn test_graph_stats_empty() {
        let graph = Graph::new(2);
        let stats = graph.stats();
        assert_eq!(stats.num_inputs, 2);
        assert_eq!(stats.num_constants, 0);
        assert_eq!(stats.num_ops, 0);
        assert_eq!(stats.max_depth, 0);
    }

    #[test]
    fn test_graph_compute_with_domain_policy_unchecked() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.ln(x);

        // Unchecked should not error even for ln(0)
        // (Actually ln(0) gives -inf which is a valid f64, but we use a small positive number)
        let value = graph
            .compute_with_domain_policy(&[1.0], DomainPolicy::Unchecked)
            .unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));
    }

    #[test]
    fn test_graph_compute_with_domain_policy_checked() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.ln(x);

        let value = graph
            .compute_with_domain_policy(&[1.0], DomainPolicy::Checked)
            .unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));

        assert!(graph
            .compute_with_domain_policy(&[0.0], DomainPolicy::Checked)
            .is_err());
    }

    #[test]
    fn test_graph_gradient_with_domain_policy() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);

        let (val, _grad) = graph
            .gradient_with_domain_policy(&[1.0], DomainPolicy::Unchecked)
            .unwrap();
        assert!(approx_eq(val, std::f64::consts::E, 1e-10));

        let (val2, _grad2) = graph
            .gradient_with_domain_policy(&[1.0], DomainPolicy::Checked)
            .unwrap();
        assert!(approx_eq(val2, std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_graph_check_gradient_pass() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let report = graph.check_gradient(&[3.0, 4.0], 1e-5).unwrap();
        assert!(report.passed);
        assert_eq!(report.entries.len(), 2);
        assert!(report.max_abs_error < 1e-5);
    }

    #[test]
    fn test_graph_compute_hessian() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let hessian = graph.compute_hessian(&[1.0, 1.0]).unwrap();
        assert_eq!(hessian.len(), 2);
        assert!(approx_eq(hessian[0][0], 0.0, 1e-4)); // d²/dx²
        assert!(approx_eq(hessian[0][1], 1.0, 1e-4)); // d²/dxdy
        assert!(approx_eq(hessian[1][0], 1.0, 1e-4)); // d²/dydx
        assert!(approx_eq(hessian[1][1], 0.0, 1e-4)); // d²/dy²
    }

    #[test]
    fn test_graph_exact_hessian_with_constants() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let two = graph.constant(2.0);
        graph.mul(x, two);

        let hessian = graph.exact_hessian_rr(&[3.0]).unwrap();
        assert!(approx_eq(hessian[0][0], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_errors_for_nonsmooth() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.relu(x);

        assert!(graph.exact_hessian_rr(&[1.0]).is_err());
    }

    #[test]
    fn test_graph_exact_hessian_errors_for_abs() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.abs(x);

        assert!(graph.exact_hessian_rr(&[1.0]).is_err());
    }

    #[test]
    fn test_graph_exact_hessian_fr_with_constants() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let two = graph.constant(2.0);
        graph.mul(x, two);

        let hessian = graph.exact_hessian_fr(&[3.0]).unwrap();
        assert!(approx_eq(hessian[0][0], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_rf_with_constants() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let two = graph.constant(2.0);
        graph.mul(x, two);

        let hessian = graph.exact_hessian_rf(&[3.0]).unwrap();
        assert!(approx_eq(hessian[0][0], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_fr_nonsmooth_errors() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.tanh(x);

        assert!(graph.exact_hessian_fr(&[0.0]).is_err());
    }

    #[test]
    fn test_graph_exact_hessian_rf_nonsmooth_errors() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.tanh(x);

        assert!(graph.exact_hessian_rf(&[0.0]).is_err());
    }

    #[test]
    fn test_graph_binary_cross_entropy_loss() {
        let mut graph = Graph::new(2);
        let pred = graph.input(0);
        let target = graph.input(1);

        // Create probability and target nodes
        let sigmoid = graph.sigmoid(pred);
        let bce = graph
            .binary_cross_entropy_loss(&[sigmoid], &[target])
            .unwrap();
        graph.set_output(bce).unwrap();

        let value = graph.compute(&[0.0, 1.0]).unwrap();
        // sigmoid(0) = 0.5, target=1 => -ln(0.5) = ln(2)
        assert!(approx_eq(value, std::f64::consts::LN_2, 1e-6));
    }

    #[test]
    fn test_graph_binary_cross_entropy_empty() {
        let mut _graph = Graph::new(1);
        assert!(Graph::new(1).binary_cross_entropy_loss(&[], &[]).is_err());
    }

    #[test]
    fn test_graph_binary_cross_entropy_mismatched() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        assert!(graph.binary_cross_entropy_loss(&[x], &[y, x]).is_err());
    }

    #[test]
    fn test_graph_mse_loss_errors() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        assert!(graph.mse_loss(&[], &[]).is_err());
        assert!(graph.mse_loss(&[x], &[]).is_err());
    }

    #[test]
    fn test_graph_dot_errors() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        assert!(graph.dot(&[], &[]).is_err());
        assert!(graph.dot(&[x], &[]).is_err());
    }

    #[test]
    fn test_graph_sum_mean_norm2_empty() {
        let mut graph = Graph::new(1);
        assert!(graph.sum(&[]).is_none());
        assert!(graph.mean(&[]).is_none());
        assert!(graph.norm2(&[]).is_none());
        assert!(graph.sum_squares(&[]).is_none());
    }

    #[test]
    fn test_graph_sum_mean_norm2() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);

        let sum = graph.sum(&[x, y]).unwrap();
        let value = graph.set_output(sum).unwrap().compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 7.0, 1e-10));

        let mean = graph.mean(&[x, y]).unwrap();
        graph.set_output(mean).unwrap();
        let value = graph.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 3.5, 1e-10));

        let norm2 = graph.norm2(&[x, y]).unwrap();
        graph.set_output(norm2).unwrap();
        let value = graph.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 5.0, 1e-10));
    }

    #[test]
    fn test_graph_logsumexp_approx_empty() {
        let mut graph = Graph::new(1);
        assert!(graph.logsumexp_approx(&[]).is_none());
    }

    #[test]
    fn test_graph_stable_softmax_approx_empty() {
        let mut graph = Graph::new(1);
        let result = graph.stable_softmax_approx(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_graph_sigmoid_stable() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let sigmoid = graph.sigmoid_stable(x);
        graph.set_output(sigmoid).unwrap();

        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.5, 1e-10));

        let val_high = graph.compute(&[100.0]).unwrap();
        assert!(val_high > 0.99);
    }

    #[test]
    fn test_graph_prune_preserves_parameters() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mark_parameter(x).unwrap();
        graph.set_parameter_name(y, "weight").unwrap();
        let used = graph.square(x);
        let _unused = graph.square(y);
        graph.set_output(used).unwrap();

        let pruned = graph.prune_to_outputs().unwrap();
        // The pruned graph should still evaluate correctly
        let value = pruned.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 9.0, 1e-10));
    }

    #[test]
    fn test_graph_compiled_metadata() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);

        let metadata = graph.compiled_metadata().unwrap();
        assert_eq!(metadata.num_inputs, 2);
        assert_eq!(metadata.num_outputs, 1);
    }

    #[test]
    fn test_graph_compiled_workspace() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);

        let _workspace = graph.compiled_workspace().unwrap();
    }

    #[test]
    fn test_graph_hessian_vector_product_errors() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        // Wrong vector length
        assert!(graph.hessian_vector_product(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_graph_compute_batch_and_gradient_batch() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();

        let values = graph.compute_batch(batch).unwrap();
        assert_eq!(values.data.len(), 2);
        assert!(approx_eq(values.data[0], 6.0, 1e-10));
        assert!(approx_eq(values.data[1], 20.0, 1e-10));

        let grads = graph.gradient_batch(batch).unwrap();
        assert_eq!(grads.batch_size, 2);
        assert!(approx_eq(grads.values[0], 6.0, 1e-10));
    }

    #[test]
    fn test_graph_compute_batch_auto_and_gradient_batch_auto() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();

        let (_backend, values) = graph.compute_batch_auto(batch).unwrap();
        assert_eq!(values.data.len(), 2);

        let (_backend, grads) = graph.gradient_batch_auto(batch).unwrap();
        assert_eq!(grads.values.len(), 2);
    }

    #[test]
    fn test_graph_compute_batch_into_and_gradient_batch_into() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let batch = BatchInputs::new(&[2.0, 3.0], 1, 2).unwrap();

        let mut values_buffer = BatchValuesBuffer::new();
        graph.compute_batch_into(batch, &mut values_buffer).unwrap();
        assert!(approx_eq(values_buffer.data[0], 6.0, 1e-10));

        let mut grad_buffer = BatchGradientsBuffer::new();
        graph.gradient_batch_into(batch, &mut grad_buffer).unwrap();
        assert!(approx_eq(grad_buffer.values[0], 6.0, 1e-10));
    }

    #[test]
    fn test_graph_compute_batch_auto_into_and_gradient_batch_auto_into() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let batch = BatchInputs::new(&[2.0, 3.0], 1, 2).unwrap();

        let mut values_buffer = BatchValuesBuffer::new();
        let backend = graph
            .compute_batch_auto_into(batch, &mut values_buffer)
            .unwrap();
        assert!(approx_eq(values_buffer.data[0], 6.0, 1e-10));

        let mut grad_buffer = BatchGradientsBuffer::new();
        let grad_backend = graph
            .gradient_batch_auto_into(batch, &mut grad_buffer)
            .unwrap();
        assert_eq!(backend, grad_backend);
    }

    #[test]
    fn test_graph_backend_support_reports() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let reports = graph.backend_support_reports().unwrap();
        assert_eq!(reports.len(), 5);

        let scalar_report = graph.backend_support_report(BackendKind::Scalar).unwrap();
        assert!(scalar_report.can_compute_batch());
    }

    #[test]
    fn test_graph_device_batch_plan() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let plan = graph.device_batch_plan(BackendKind::Scalar, 4).unwrap();
        assert_eq!(plan.batch_size, 4);
    }

    #[test]
    fn test_graph_allocate_mock_device_buffers() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let buffers = graph.allocate_mock_device_buffers(2).unwrap();
        assert_eq!(buffers.buffers().len(), 5);
    }

    #[test]
    fn test_graph_compute_batch_mock_device_into() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let mut buffers = graph.allocate_mock_device_buffers(2).unwrap();
        let mut output = BatchValuesBuffer::new();

        let trace = graph
            .compute_batch_mock_device_into(batch, &mut buffers, &mut output)
            .unwrap();
        assert_eq!(trace.mode, crate::DeviceExecutionMode::ComputeBatch);
    }

    #[test]
    fn test_graph_gradient_batch_mock_device_into() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let batch = BatchInputs::new(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let mut buffers = graph.allocate_mock_device_buffers(2).unwrap();
        let mut output = BatchGradientsBuffer::new();

        let trace = graph
            .gradient_batch_mock_device_into(batch, &mut buffers, &mut output)
            .unwrap();
        assert_eq!(trace.mode, crate::DeviceExecutionMode::GradientBatch);
    }

    #[test]
    fn test_graph_simd_support_report() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let report = graph.simd_support_report().unwrap();
        assert!(report.missing_opcodes.is_empty());
    }

    #[test]
    fn test_graph_recommended_batch_backends() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let compute = graph.recommended_batch_compute_backend().unwrap();
        let gradient = graph.recommended_batch_gradient_backend().unwrap();
        assert!(
            compute == BackendKind::Scalar
                || compute == BackendKind::SimdF64x4
                || compute == BackendKind::SimdF64x2
        );
        assert!(
            gradient == BackendKind::Scalar
                || gradient == BackendKind::SimdF64x4
                || gradient == BackendKind::SimdF64x2
        );
    }

    #[test]
    fn test_graph_to_operations_input_output() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let _y = graph.input(1);
        graph.set_output(x).unwrap();

        let ops = graph.to_operations().unwrap();
        // No operations, just the input marker
        assert_eq!(ops, vec![(MultiAD::Inp, vec![0])]);
    }

    #[test]
    fn test_graph_parse_expression() {
        let graph = Graph::parse_expression("x + y", &["x", "y"]).unwrap();
        let value = graph.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 7.0, 1e-10));
    }

    #[test]
    fn test_graph_parse_expression_complex() {
        let graph = Graph::parse_expression("sin(x) * y + 2", &["x", "y"]).unwrap();
        let value = graph.compute(&[0.5, 3.0]).unwrap();
        let expected = 0.5_f64.sin() * 3.0 + 2.0;
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_graph_to_operations_for_input_only() {
        // Empty graph with only inputs - output should be last input
        let graph = Graph::new(2);
        let ops = graph.to_operations().unwrap();
        assert_eq!(ops, vec![(MultiAD::Inp, vec![1])]); // Falls back to last input
    }

    #[test]
    fn test_graph_to_operations_no_output() {
        // Graph with 0 inputs and 0 nodes
        let graph = Graph::new(0);
        let ops = graph.to_operations().unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_expr_graph_neg_sub_div() {
        let expr = ExprGraph::new(2);
        let x = expr.input(0);
        let y = expr.input(1);
        let neg_x = -x.clone();
        let diff = y.clone() - neg_x;
        let quotient = diff / x;
        expr.set_output(&quotient).unwrap();
        let graph = expr.graph();
        let value = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.5, 1e-10)); // (3 - (-2)) / 2
    }

    #[test]
    fn test_expr_graph_with_constants() {
        let expr = ExprGraph::new(1);
        let x = expr.input(0);
        let result = x.clone() + 1.0;
        let result2 = result * 2.0;
        let result3 = result2.clone() - 0.5;
        let result4 = result3 / 1.0;
        expr.set_output(&result4).unwrap();
        let graph = expr.graph();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 7.5, 1e-10)); // ((3+1)*2 - 0.5)/1
    }

    #[test]
    fn test_expr_node_unary_methods() {
        let expr = ExprGraph::new(1);
        let x = expr.input(0);
        let sin_x = x.sin();
        let cos_x = sin_x.cos();
        let exp_x = cos_x.exp();
        let ln_x = exp_x.ln();
        let sqrt_x = ln_x.sqrt();
        expr.set_output(&sqrt_x).unwrap();
        let graph = expr.graph();
        let value = graph.compute(&[0.5]).unwrap();
        let expected = (0.5_f64.sin().cos().exp()).ln().sqrt();
        assert!(approx_eq(value, expected, 1e-8));
    }

    #[test]
    fn test_try_from_graph_to_operations() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);

        let ops: Vec<(MultiAD, Vec<usize>)> = (&graph).try_into().unwrap();
        let value = MultiAD::compute(&ops, &[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 7.0, 1e-10));
    }

    #[test]
    fn test_try_from_graph_ref_to_operations() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);

        let ops: Vec<(MultiAD, Vec<usize>)> = (&graph).try_into().unwrap();
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn test_graph_simplify_unary_chain() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let neg_x = graph.neg(x);
        let neg_neg = graph.neg(neg_x);
        graph.set_output(neg_neg).unwrap();

        // Simplification won't fold neg(neg(x)) yet, but it should still evaluate correctly
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_graph_simplify_add_left_zero() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let zero = graph.constant(0.0);
        graph.add(zero, x);

        let simplified = graph.simplify().unwrap();
        let value = simplified.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_graph_simplify_complex() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let zero = graph.constant(0.0);
        let one = graph.constant(1.0);
        let x_plus_zero = graph.add(x, zero);
        let times_one = graph.mul(x_plus_zero, one);
        graph.set_output(times_one).unwrap();

        let simplified = graph.simplify().unwrap();
        let value = simplified.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_fr_cos() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.cos(x);

        let hessian = graph.exact_hessian_fr(&[0.5]).unwrap();
        // d²/dx² cos(x) = -cos(x)
        assert!(approx_eq(hessian[0][0], -0.5_f64.cos(), 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_rf_cos() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.cos(x);

        let hessian = graph.exact_hessian_rf(&[0.5]).unwrap();
        assert!(approx_eq(hessian[0][0], -0.5_f64.cos(), 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_rr_sin() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sin(x);

        let hessian = graph.exact_hessian_rr(&[0.5]).unwrap();
        // d²/dx² sin(x) = -sin(x)
        assert!(approx_eq(hessian[0][0], -0.5_f64.sin(), 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_binary_ops() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let product = graph.mul(x, y);
        graph.set_output(product).unwrap();

        let hessian = graph.exact_hessian_rr(&[3.0, 4.0]).unwrap();
        // d²(xy)/dx² = 0, d²(xy)/dxdy = 1, d²(xy)/dy² = 0
        assert!(approx_eq(hessian[0][0], 0.0, 1e-10));
        assert!(approx_eq(hessian[0][1], 1.0, 1e-10));
        assert!(approx_eq(hessian[1][0], 1.0, 1e-10));
        assert!(approx_eq(hessian[1][1], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_exact_hessian_div() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.div(x, y);

        let hessian = graph.exact_hessian_rr(&[4.0, 2.0]).unwrap();
        // d²(x/y)/dx² = 0, d²(x/y)/dxdy = -1/y²
        assert!(approx_eq(hessian[0][0], 0.0, 1e-10));
        assert!(approx_eq(hessian[0][1], -0.25, 1e-10));
    }

    #[test]
    fn test_graph_compute_grad_closure() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let (value, grad_fn) = graph.compute_grad(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 12.0, 1e-10));
        let grad = grad_fn(1.0);
        assert!(approx_eq(grad[0], 4.0, 1e-10));
        assert!(approx_eq(grad[1], 3.0, 1e-10));
    }

    #[test]
    fn test_graph_compute_grad_multi_output() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.set_outputs(&[x, y]).unwrap();

        let (value, grad_fn) = graph.compute_grad(&[3.0, 4.0]).unwrap();
        // With multi-output, compute_grad returns the first output
        assert!(approx_eq(value, 3.0, 1e-10));
        let grad = grad_fn(1.0);
        assert!(approx_eq(grad[0], 1.0, 1e-10));
        assert!(approx_eq(grad[1], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_compute_hessian_empty() {
        let graph = Graph::new(0);
        let hessian = graph.compute_hessian(&[]).unwrap();
        assert_eq!(hessian.len(), 0);
    }

    #[test]
    fn test_graph_gradient_sparse_with_tolerance() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let sparse = graph
            .gradient_sparse_with_tolerance(&[0.0, 4.0], 0.0)
            .unwrap();
        // At x=0, dy/dx=4, dy/dy=0
        assert_eq!(sparse.len(), 1); // Only the y gradient (0) should be filtered? No - tolerance is 0 so both
        assert_eq!(sparse.len(), 1); // dy/dx=4 > 0, dy/dy=0 == 0
                                     // Actually at x=0,y=4: dy/dx = y = 4, dy/dy = x = 0
    }

    #[test]
    fn test_graph_hessian_sparse() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);

        let sparse = graph.hessian_sparse(&[1.0, 2.0]).unwrap();
        // Hessian of x+y is all zeros
        assert!(sparse.is_empty());
    }

    #[test]
    fn test_graph_hessian_sparse_with_tolerance() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);

        let sparse = graph
            .hessian_sparse_with_tolerance(&[1.0, 2.0], 0.0)
            .unwrap();
        // Hessian of xy: d²/dxdy = 1, d²/dydx = 1, rest are 0
        assert_eq!(sparse.len(), 2);
    }

    #[test]
    fn test_graph_compile_ir_with_invalid_ops() {
        let mut graph = Graph::new(1);
        // Push a ternary-like operation which should fail IR compilation
        graph.push_operation(MultiAD::Add, vec![0, 0, 0]); // 3 args for Add
        assert!(graph.compile_ir().is_err());
    }

    #[test]
    fn test_graph_validate_rejects_inp_marker() {
        let mut graph = Graph::new(1);
        graph.push_operation(MultiAD::Inp, vec![0]);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_graph_validate_rejects_wrong_arity() {
        let mut graph = Graph::new(1);
        graph.push_operation(MultiAD::Sin, vec![0, 1]); // Sin takes 1 arg
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_graph_try_push_operation_errors() {
        let mut graph = Graph::new(1);
        // Inp marker
        assert!(graph.try_push_operation(MultiAD::Inp, vec![0]).is_err());
        // Out of bounds
        assert!(graph.try_push_operation(MultiAD::Sin, vec![5]).is_err());
    }

    #[test]
    fn test_graph_try_unary_and_binary_errors() {
        let mut graph = Graph::new(1);
        assert!(graph.try_sin(5).is_err());
        assert!(graph.try_cos(5).is_err());
        assert!(graph.try_tan(5).is_err());
        assert!(graph.try_tanh(5).is_err());
        assert!(graph.try_relu(5).is_err());
        assert!(graph.try_log1p_exp(5).is_err());
        assert!(graph.try_neg(5).is_err());
        assert!(graph.try_exp(5).is_err());
        assert!(graph.try_ln(5).is_err());
        assert!(graph.try_sqrt(5).is_err());
        assert!(graph.try_abs(5).is_err());
        assert!(graph.try_add(5, 0).is_err());
        assert!(graph.try_sub(5, 0).is_err());
        assert!(graph.try_mul(5, 0).is_err());
        assert!(graph.try_div(5, 0).is_err());
        assert!(graph.try_pow(5, 0).is_err());
        assert!(graph.try_log_add_exp(5, 0).is_err());
    }

    #[test]
    fn test_graph_constant_and_square_cube_reciprocal() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let sq = graph.square(x);
        let cu = graph.cube(x);
        let recip = graph.reciprocal(x);
        graph.set_outputs(&[sq, cu, recip]).unwrap();

        let values = graph.compute_many(&[2.0]).unwrap();
        assert!(approx_eq(values[0], 4.0, 1e-10));
        assert!(approx_eq(values[1], 8.0, 1e-10));
        assert!(approx_eq(values[2], 0.5, 1e-10));
    }

    #[test]
    fn test_graph_pow_const() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.pow_const(x, 3.0);
        let value = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(value, 8.0, 1e-10));
    }

    #[test]
    fn test_graph_sub_const_and_div_const() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let subbed = graph.sub_const(x, 1.0);
        graph.div_const(subbed, 2.0);
        let value = graph.compute(&[5.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_graph_gelu() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.gelu(x);
        let value = graph.compute(&[1.0]).unwrap();
        // GELU(1) ≈ 0.8412
        assert!(value > 0.0 && value < 1.0);
    }

    #[test]
    fn test_graph_node_id_and_output_node() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        assert_eq!(x, 0);

        let y = graph.input(1);
        assert_eq!(y, 1);

        let sum = graph.add(x, y);
        assert_eq!(sum, 2);

        graph.set_output(sum).unwrap();
        assert_eq!(graph.output_node(), Some(2));
        assert_eq!(graph.output_nodes(), &[2]);
    }

    #[test]
    fn test_graph_to_mermaid_with_output_names() {
        let mut graph = Graph::new(1);
        graph.set_input_name(0, "x").unwrap();
        let x = graph.input(0);
        let sin_x = graph.sin(x);
        graph.set_output_name(sin_x, "result").unwrap();

        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("x: Input 0"));
        assert!(mermaid.contains("result"));
    }

    #[test]
    fn test_graph_to_dot_with_output_names() {
        let mut graph = Graph::new(1);
        graph.set_input_name(0, "x").unwrap();
        let x = graph.input(0);
        let sin_x = graph.sin(x);
        graph.set_output_name(sin_x, "result").unwrap();

        let dot = graph.to_dot();
        assert!(dot.contains("x: Input 0"));
        assert!(dot.contains("result"));
    }

    #[test]
    fn test_tape_compute_many_with_workspace_checked() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);
        let tape = graph.compile();
        let mut workspace = tape.workspace();

        let values = tape
            .compute_many_with_workspace_checked(&[1.0], &mut workspace)
            .unwrap();
        assert!(approx_eq(values[0], std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_tape_compute_with_workspace_checked() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.exp(x);
        let tape = graph.compile();
        let mut workspace = tape.workspace();

        let value = tape
            .compute_with_workspace_checked(&[1.0], &mut workspace)
            .unwrap();
        assert!(approx_eq(value, std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_tape_jacobian_with_workspace() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.set_outputs(&[x, y]).unwrap();
        let tape = graph.compile();
        let mut workspace = tape.workspace();

        let jacobian = tape
            .jacobian_with_workspace(&[1.0, 2.0], &mut workspace)
            .unwrap();
        assert_eq!(jacobian.len(), 2);
    }

    #[test]
    fn test_tape_jacobian_with_workspace_checked() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.set_outputs(&[x, y]).unwrap();
        let tape = graph.compile();
        let mut workspace = tape.workspace();

        let jacobian = tape
            .jacobian_with_workspace_checked(&[1.0, 2.0], &mut workspace)
            .unwrap();
        assert_eq!(jacobian.len(), 2);
    }

    #[test]
    fn test_tape_graph_access() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.add(x, y);
        let tape = graph.compile();
        assert_eq!(tape.graph().num_inputs(), 2);
    }

    #[test]
    fn test_tape_compute_hessian() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);
        let tape = graph.compile();

        let hessian = tape.compute_hessian(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(hessian[0][1], 1.0, 1e-4));
        assert!(approx_eq(hessian[1][0], 1.0, 1e-4));
    }

    #[test]
    fn test_tape_workspace_clear() {
        let mut ws = TapeWorkspace::new();
        ws.values.push(1.0);
        ws.cotangent_values.push(2.0);
        ws.gradients.push(3.0);
        ws.clear();
        assert!(ws.values.is_empty());
        assert!(ws.cotangent_values.is_empty());
        assert!(ws.gradients.is_empty());
    }

    #[test]
    fn test_graph_value_and_gradient_alias() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.square(x);

        let (v1, g1) = graph.gradient(&[3.0]).unwrap();
        let (v2, g2) = graph.value_and_gradient(&[3.0]).unwrap();
        assert!(approx_eq(v1, v2, 1e-10));
        assert_eq!(g1, g2);
    }

    #[test]
    fn test_graph_check_gradient_report() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.square(x);

        let report = graph.check_gradient(&[3.0], 1e-5).unwrap();
        assert!(report.passed);
        assert_eq!(report.entries.len(), 1);
        assert!(report.entries[0].abs_error < 1e-5);
    }

    // ====================
    // Coverage: op methods, checked, reductions, hessian, parameter metadata
    // ====================

    #[test]
    fn test_graph_pow_forward_and_gradient() {
        let mut graph = Graph::new(2);
        let base = graph.input(0);
        let exp = graph.input(1);
        graph.pow(base, exp);
        // 2^3 = 8
        let value = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 8.0, 1e-10));
        // dx: 3*2^(3-1)=12, dy: 8*ln(2)≈5.545
        let (_v, grad) = graph.gradient(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(grad[0], 12.0, 1e-8));
        assert!(approx_eq(grad[1], 8.0 * 2.0_f64.ln(), 1e-8));
    }

    #[test]
    fn test_graph_pow_const_compute() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.pow_const(x, 4.0);
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 81.0, 1e-10));
    }

    #[test]
    fn test_graph_softplus_compute_and_gradient() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.softplus(x);
        // softplus(0) = ln(2)
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::LN_2, 1e-12));
        // gradient at 0 = sigmoid(0) = 0.5
        let (_v, grad) = graph.gradient(&[0.0]).unwrap();
        assert!(approx_eq(grad[0], 0.5, 1e-12));
        // softplus large x ≈ x
        let large_v = graph.compute(&[1000.0]).unwrap();
        assert!((large_v - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_graph_sigmoid_compute_and_gradient() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sigmoid(x);
        // sigmoid(0) = 0.5
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.5, 1e-10));
        // d/dx sigmoid at 0 = sigmoid(0)*(1-sigmoid(0)) = 0.25
        let (_v, grad) = graph.gradient(&[0.0]).unwrap();
        assert!(approx_eq(grad[0], 0.25, 1e-10));
        // sigmoid large x ≈ 1
        let large_v = graph.compute(&[100.0]).unwrap();
        assert!(approx_eq(large_v, 1.0, 1e-10));
    }

    #[test]
    fn test_graph_gelu_compute_and_gradient() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.gelu(x);
        // GELU(0) = 0
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-12));
        // GELU(-large) ≈ 0
        let neg_v = graph.compute(&[-100.0]).unwrap();
        assert!((neg_v - 0.0).abs() < 1e-10);
        // GELU should be finite and between 0 and x for positive x
        let pos_v = graph.compute(&[2.0]).unwrap();
        assert!(pos_v > 0.0 && pos_v < 2.0);
        let (_v, grad) = graph.gradient(&[0.0]).unwrap();
        assert!(approx_eq(grad[0], 0.5, 1e-12));
    }

    #[test]
    fn test_graph_relu_at_zero() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.relu(x);
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));
        // relu(0) = 0, and standard convention: relu'(0) = 0
        let (_v, grad) = graph.gradient(&[0.0]).unwrap();
        assert!(approx_eq(grad[0], 0.0, 1e-10));
    }

    #[test]
    fn test_graph_gradient_checked_domain_error() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.ln(x);
        let error = graph.gradient_checked(&[0.0]).unwrap_err();
        assert_eq!(
            error,
            crate::AutodiffError::DomainError {
                operation: "Ln",
                reason: "input must be positive",
            }
        );
    }

    #[test]
    fn test_graph_compute_hessian_for_simple_function() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.square(x);
        // f(x)=x², f''(x)=2
        let hessian = graph.compute_hessian(&[3.0]).unwrap();
        assert_eq!(hessian.len(), 1);
        assert!(approx_eq(hessian[0][0], 2.0, 1e-4));
    }

    #[test]
    fn test_graph_compute_hessian_for_empty_graph() {
        let graph = Graph::new(0);
        let hessian = graph.compute_hessian(&[]).unwrap();
        assert_eq!(hessian.len(), 0);
    }

    #[test]
    fn test_graph_gradient_sparse_with_tolerance_all_nonzero() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);
        // At (x=3, y=4): dy/dx=4, dy/dy=3 — both > tolerance 1.0
        let sparse = graph
            .gradient_sparse_with_tolerance(&[3.0, 4.0], 1.0)
            .unwrap();
        assert_eq!(sparse.len(), 2);
    }

    #[test]
    fn test_graph_gradient_sparse_with_tolerance_filtered() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);
        // At (x=0, y=4): dy/dx=4, dy/dy=0 — only dy/dx > tolerance 0.5
        let sparse = graph
            .gradient_sparse_with_tolerance(&[0.0, 4.0], 0.5)
            .unwrap();
        assert_eq!(sparse.len(), 1);
        // dy/dx = y = 4 (input index 0), dy/dy = x = 0 (input index 1, filtered)
        assert_eq!(sparse[0].0, 0);
        assert!(approx_eq(sparse[0].1, 4.0, 1e-10));
    }

    #[test]
    fn test_graph_norm2_single_input() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let norm = graph.norm2(&[x]).unwrap();
        graph.set_output(norm).unwrap();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
        let value_neg = graph.compute(&[-4.0]).unwrap();
        assert!(approx_eq(value_neg, 4.0, 1e-10));
    }

    #[test]
    fn test_graph_sum_squares_single_input() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let ss = graph.sum_squares(&[x]).unwrap();
        graph.set_output(ss).unwrap();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 9.0, 1e-10));
    }

    #[test]
    fn test_graph_dot_with_valid_inputs() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let dot = graph.dot(&[x, y], &[x, y]).unwrap();
        graph.set_output(dot).unwrap();
        // dot([3,4], [3,4]) = 9+16 = 25
        let value = graph.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 25.0, 1e-10));
        // dot([a,b], [c,d]) = ac+bd
        let dot2 = graph.dot(&[x, y], &[y, x]).unwrap();
        graph.set_output(dot2).unwrap();
        let value2 = graph.compute(&[2.0, 3.0]).unwrap();
        // 2*3 + 3*2 = 12
        assert!(approx_eq(value2, 12.0, 1e-10));
    }

    #[test]
    fn test_graph_dot_with_empty_and_mismatched() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        assert!(graph.dot(&[], &[]).is_err());
        assert!(graph.dot(&[x], &[]).is_err());
        assert!(graph.dot(&[x], &[x, y]).is_err());
    }

    #[test]
    fn test_graph_sum_with_multiple_inputs() {
        let mut graph = Graph::new(3);
        let a = graph.input(0);
        let b = graph.input(1);
        let c = graph.input(2);
        let sum = graph.sum(&[a, b, c]).unwrap();
        graph.set_output(sum).unwrap();
        let value = graph.compute(&[1.0, 2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 6.0, 1e-10));
    }

    #[test]
    fn test_graph_mean_with_multiple_inputs() {
        let mut graph = Graph::new(3);
        let a = graph.input(0);
        let b = graph.input(1);
        let c = graph.input(2);
        let mean = graph.mean(&[a, b, c]).unwrap();
        graph.set_output(mean).unwrap();
        let value = graph.compute(&[1.0, 2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_graph_to_operations_input_only() {
        let graph = Graph::new(2);
        let ops = graph.to_operations().unwrap();
        assert_eq!(ops, vec![(MultiAD::Inp, vec![1])]);
    }

    #[test]
    fn test_graph_from_operations_roundtrip() {
        let ops = vec![
            (MultiAD::Inp, vec![0]),
            (MultiAD::Inp, vec![1]),
            (MultiAD::Mul, vec![0, 1]),
        ];
        let graph = Graph::from_operations(2, &ops);
        let value = graph.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, 12.0, 1e-10));
        // Round trip via try_from_operations
        let converted = graph.to_operations().unwrap();
        let graph2 = Graph::try_from_operations(2, &converted).unwrap();
        let value2 = graph2.compute(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(value, value2, 1e-10));
    }

    #[test]
    fn test_graph_parameter_names_and_metadata() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.set_parameter_name(x, "weight").unwrap();
        graph.set_parameter_name(y, "bias").unwrap();

        assert_eq!(graph.parameter_name(x), Some("weight"));
        assert_eq!(graph.parameter_name(y), Some("bias"));
        assert_eq!(graph.parameter_names().len(), 2);
        assert!(graph
            .parameter_names()
            .iter()
            .any(|(id, name)| *id == x && name == "weight"));
        assert!(graph
            .parameter_names()
            .iter()
            .any(|(id, name)| *id == y && name == "bias"));
        // parameter_name for non-parameter returns None
        assert_eq!(graph.parameter_name(99), None);
    }

    #[test]
    fn test_graph_compile_accelerated_full() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        graph.sin(x);
        let compiled = graph.compile_accelerated().unwrap();
        let value = compiled.compute(&[0.5]).unwrap();
        assert!(approx_eq(value, 0.5_f64.sin(), 1e-10));
        let (_v, grad) = compiled.gradient(&[0.5]).unwrap();
        assert!(approx_eq(grad[0], 0.5_f64.cos(), 1e-8));
    }

    #[test]
    fn test_graph_is_empty_and_len_both() {
        let mut graph = Graph::new(2);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);

        let x = graph.input(0);
        graph.sin(x);
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);

        graph.cos(x);
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_graph_sum_squares_multiple_inputs() {
        let mut graph = Graph::new(3);
        let a = graph.input(0);
        let b = graph.input(1);
        let c = graph.input(2);
        let ss = graph.sum_squares(&[a, b, c]).unwrap();
        graph.set_output(ss).unwrap();
        let value = graph.compute(&[1.0, 2.0, 3.0]).unwrap();
        // 1 + 4 + 9 = 14
        assert!(approx_eq(value, 14.0, 1e-10));
    }

    #[test]
    fn test_graph_norm2_multiple_inputs() {
        let mut graph = Graph::new(3);
        let a = graph.input(0);
        let b = graph.input(1);
        let c = graph.input(2);
        let norm = graph.norm2(&[a, b, c]).unwrap();
        graph.set_output(norm).unwrap();
        let value = graph.compute(&[3.0, 4.0, 0.0]).unwrap();
        assert!(approx_eq(value, 5.0, 1e-10));
    }

    #[test]
    fn test_graph_hessian_sparse_full() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);
        let sparse = graph.hessian_sparse(&[2.0, 3.0]).unwrap();
        // Hessian of xy: d²/dxdy=1, d²/dydx=1 — 2 nonzeros
        assert_eq!(sparse.len(), 2);
        assert!(approx_eq(sparse[0].2, 1.0, 1e-10));
        assert!(approx_eq(sparse[1].2, 1.0, 1e-10));
    }

    #[test]
    fn test_graph_hessian_sparse_with_tolerance_full() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        graph.mul(x, y);
        // At tolerance 0.0: all nonzero entries returned
        let sparse = graph
            .hessian_sparse_with_tolerance(&[2.0, 3.0], 0.0)
            .unwrap();
        assert_eq!(sparse.len(), 2);
        // At tolerance 2.0: no entries survive
        let sparse_filtered = graph
            .hessian_sparse_with_tolerance(&[2.0, 3.0], 2.0)
            .unwrap();
        assert_eq!(sparse_filtered.len(), 0);
    }

    #[test]
    fn test_graph_try_from_operations_with_inp_error() {
        let ops = [(MultiAD::Inp, vec![5])];
        assert!(Graph::try_from_operations(2, &ops).is_err());
    }

    #[test]
    fn test_graph_try_from_operations_multi_inp() {
        let ops = [(MultiAD::Inp, vec![0, 1])];
        assert!(Graph::try_from_operations(2, &ops).is_err());
    }

    #[test]
    fn test_try_sin_and_try_cos() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let s = graph.try_sin(x).unwrap();
        let c = graph.try_cos(x).unwrap();
        graph.set_outputs(&[s, c]).unwrap();
        let v = graph.compute_many(&[1.0]).unwrap();
        assert!(approx_eq(v[0], 1.0_f64.sin(), 1e-10));
        assert!(approx_eq(v[1], 1.0_f64.cos(), 1e-10));
    }

    #[test]
    fn test_try_tan_and_try_exp() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let t = graph.try_tan(x).unwrap();
        let e = graph.try_exp(x).unwrap();
        graph.set_outputs(&[t, e]).unwrap();
        let v = graph.compute_many(&[0.5]).unwrap();
        assert!(approx_eq(v[0], 0.5_f64.tan(), 1e-10));
        assert!(approx_eq(v[1], 0.5_f64.exp(), 1e-10));
    }

    #[test]
    fn test_try_ln_and_try_sqrt_try_abs() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let l = graph.try_ln(x).unwrap();
        let s = graph.try_sqrt(x).unwrap();
        let a = graph.try_abs(x).unwrap();
        graph.set_outputs(&[l, s, a]).unwrap();
        let v = graph.compute_many(&[4.0]).unwrap();
        assert!(approx_eq(v[0], 4.0_f64.ln(), 1e-10));
        assert!(approx_eq(v[1], 4.0_f64.sqrt(), 1e-10));
        assert!(approx_eq(v[2], 4.0_f64.abs(), 1e-10));
    }

    #[test]
    fn test_try_neg_and_try_tanh_try_relu() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let n = graph.try_neg(x).unwrap();
        let t = graph.try_tanh(x).unwrap();
        let r = graph.try_relu(x).unwrap();
        graph.set_outputs(&[n, t, r]).unwrap();
        let v = graph.compute_many(&[2.0]).unwrap();
        assert!(approx_eq(v[0], -2.0, 1e-10));
        assert!(approx_eq(v[1], 2.0_f64.tanh(), 1e-10));
        assert!(approx_eq(v[2], 2.0, 1e-10));
    }

    #[test]
    fn test_try_reductions() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let s = graph.sum(&[x, y]).unwrap();
        let m = graph.mean(&[x, y]).unwrap();
        let ss = graph.sum_squares(&[x, y]).unwrap();
        let n = graph.norm2(&[x]).unwrap();
        graph.set_outputs(&[s, m, ss, n]).unwrap();
        let v = graph.compute_many(&[3.0, 4.0]).unwrap();
        assert!(approx_eq(v[0], 7.0, 1e-10));
        assert!(approx_eq(v[1], 3.5, 1e-10));
        assert!(approx_eq(v[2], 25.0, 1e-10));
        assert!(approx_eq(v[3], 3.0, 1e-10));
    }

    #[test]
    fn test_try_binary_ops() {
        let mut graph = Graph::new(2);
        let x = graph.input(0);
        let y = graph.input(1);
        let a = graph.try_add(x, y).unwrap();
        let s = graph.try_sub(x, y).unwrap();
        let m = graph.try_mul(x, y).unwrap();
        let d = graph.try_div(x, y).unwrap();
        let p = graph.try_pow(x, y).unwrap();
        graph.set_outputs(&[a, s, m, d, p]).unwrap();
        let v = graph.compute_many(&[4.0, 2.0]).unwrap();
        assert!(approx_eq(v[0], 6.0, 1e-10));
        assert!(approx_eq(v[1], 2.0, 1e-10));
        assert!(approx_eq(v[2], 8.0, 1e-10));
        assert!(approx_eq(v[3], 2.0, 1e-10));
        assert!(approx_eq(v[4], 16.0, 1e-10));
    }

    #[test]
    fn test_graph_effective_output_nodes_with_set_outputs() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let s = graph.sin(x);
        let c = graph.cos(x);
        graph.set_outputs(&[s, c]).unwrap();
        let nodes = graph.effective_output_nodes();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&s));
        assert!(nodes.contains(&c));
    }

    #[test]
    fn test_graph_clear_output_keeps_last_node() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let s = graph.sin(x);
        graph.set_output(s).unwrap();
        assert_eq!(graph.effective_output_node(), Some(s));
        graph.clear_output();
        // After clearing, output goes back to last node
        assert_eq!(graph.effective_output_node(), Some(s));
    }

    #[test]
    fn test_graph_from_operations_checked() {
        let ops = vec![(MultiAD::Inp, vec![0]), (MultiAD::Sin, vec![0])];
        let graph = Graph::try_from_operations(1, &ops).unwrap();
        let v = graph.compute(&[1.0]).unwrap();
        assert!(approx_eq(v, 1.0_f64.sin(), 1e-10));
    }

    #[test]
    fn test_graph_from_operations_checked_with_bad_input() {
        // Input index out of bounds
        let ops = vec![(MultiAD::Inp, vec![99])];
        let result = Graph::try_from_operations(1, &ops);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_from_operations_with_constants() {
        let mut graph = Graph::new(1);
        let _c = graph.constant(5.0);
        let ops = graph.to_operations();
        // to_operations should fail when constants are present
        assert!(ops.is_err());
    }

    #[test]
    fn test_graph_add_const() {
        let mut graph = Graph::new(1);
        let x = graph.input(0);
        let a = graph.add_const(x, 3.0);
        graph.set_output(a).unwrap();
        let v = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(v, 5.0, 1e-10));
        let _ = a;
    }

    #[test]
    fn test_graph_dot_product() {
        let mut graph = Graph::new(3);
        let a = graph.input(0);
        let b = graph.input(1);
        let c_ = graph.input(2);
        let dot = graph.dot(&[a, b, c_], &[a, b, c_]).unwrap();
        graph.set_output(dot).unwrap();
        let v = graph.compute(&[1.0, 2.0, 3.0]).unwrap();
        assert!(approx_eq(v, 14.0, 1e-10));
        let _ = dot;
    }
}
