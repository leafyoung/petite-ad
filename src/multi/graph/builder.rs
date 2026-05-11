//! Builder API for constructing multi-variable computation graphs.
//!
//! This module provides a fluent, type-safe interface for building computational
//! graphs without manually managing indices and vectors.

use crate::multi::first_order::MultiAD;
use crate::multi::graph::core::{Graph, GraphNode, NodeId};

/// Builder for constructing multi-variable computation graphs.
///
/// Provides a fluent API for building computational graphs without manually
/// tracking indices. The builder automatically manages node indexing and
/// produces the final graph structure.
///
/// # Examples
///
/// ```rust
/// use petite_ad::GraphBuilder;
///
/// // Legacy tuple graph API.
/// let graph = GraphBuilder::new(2)
///     .add(0, 1)
///     .sin(0)
///     .mul(2, 3)
///     .build();
///
/// let inputs = &[0.6, 1.4];
/// let (value, grad_fn) = petite_ad::MultiAD::compute_grad(&graph, inputs).unwrap();
///
/// // Node-handle API for reusable graphs.
/// let mut builder = GraphBuilder::new(2);
/// let x = builder.input_node(0);
/// let y = builder.input_node(1);
/// let sum = builder.add_node(x, y);
/// let sin_x = builder.sin_node(x);
/// let reusable = builder.mul_node(sum, sin_x);
/// assert_eq!(reusable, 4);
/// let graph = builder.build_graph();
/// let value_from_graph = graph.compute(inputs).unwrap();
/// assert!((value - value_from_graph).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    /// Number of input variables
    num_inputs: usize,
    /// Operations in the legacy tuple computation graph
    operations: Vec<(MultiAD, Vec<usize>)>,
    /// Nodes for the reusable graph representation
    graph_nodes: Vec<GraphNode>,
    /// Next available index for new operations
    next_index: usize,
}

impl GraphBuilder {
    #[inline]
    fn push_node(&mut self, op: MultiAD, indices: Vec<usize>) -> NodeId {
        let node_id = self.next_index;
        self.operations.push((op, indices.clone()));
        if op != MultiAD::Inp {
            self.graph_nodes.push(GraphNode::Operation {
                op,
                inputs: indices,
            });
            self.next_index += 1;
        }
        node_id
    }

    /// Creates a new graph builder with the specified number of inputs.
    ///
    /// # Arguments
    ///
    /// * `num_inputs` - Number of input variables (indices 0 to num_inputs-1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use petite_ad::GraphBuilder;
    ///
    /// let builder = GraphBuilder::new(3);  // 3 inputs: x0, x1, x2
    /// ```
    pub fn new(num_inputs: usize) -> Self {
        Self {
            num_inputs,
            operations: Vec::new(),
            graph_nodes: Vec::new(),
            next_index: num_inputs,
        }
    }

    /// Adds an input placeholder operation.
    ///
    /// Inputs are already available at indices `0..num_inputs`, so this marker
    /// does not allocate a new graph value and does not change `next_index()`.
    /// It is mainly useful for keeping builder output visually aligned with
    /// graphs built using the `multi_ops!` macro.
    ///
    /// # Arguments
    ///
    /// * `input_index` - Which input variable to reference (0 to num_inputs-1)
    pub fn input(&mut self, input_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Inp, vec![input_index]));
        self
    }

    /// Returns the node id for an input.
    #[must_use]
    pub fn input_node(&self, input_index: usize) -> NodeId {
        input_index
    }

    /// Adds a graph-only constant node and returns its output node id.
    ///
    /// Constants cannot be represented in the legacy tuple graph returned by
    /// [`GraphBuilder::build`]. Use [`GraphBuilder::build_graph`] after calling
    /// this method.
    pub fn constant_node(&mut self, value: f64) -> NodeId {
        let node_id = self.next_index;
        self.graph_nodes.push(GraphNode::Constant(value));
        self.next_index += 1;
        node_id
    }

    /// Adds a sine operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    ///
    /// # Returns
    ///
    /// The index where this operation's result will be stored
    pub fn sin(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.sin_node(arg_index);
        self
    }

    /// Adds a sine operation and returns its output node id.
    pub fn sin_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Sin, vec![arg_index])
    }

    /// Adds a cosine operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn cos(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.cos_node(arg_index);
        self
    }

    /// Adds a cosine operation and returns its output node id.
    pub fn cos_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Cos, vec![arg_index])
    }

    /// Adds a tangent operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn tan(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.tan_node(arg_index);
        self
    }

    /// Adds a tangent operation and returns its output node id.
    pub fn tan_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Tan, vec![arg_index])
    }

    /// Adds a negation operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn neg(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.neg_node(arg_index);
        self
    }

    /// Adds a negation operation and returns its output node id.
    pub fn neg_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Neg, vec![arg_index])
    }

    /// Adds an exponential operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn exp(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.exp_node(arg_index);
        self
    }

    /// Adds an exponential operation and returns its output node id.
    pub fn exp_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Exp, vec![arg_index])
    }

    /// Adds a natural logarithm operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn ln(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.ln_node(arg_index);
        self
    }

    /// Adds a natural logarithm operation and returns its output node id.
    pub fn ln_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Ln, vec![arg_index])
    }

    /// Adds a square root operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn sqrt(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.sqrt_node(arg_index);
        self
    }

    /// Adds a square root operation and returns its output node id.
    pub fn sqrt_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Sqrt, vec![arg_index])
    }

    /// Adds an absolute value operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn abs(&mut self, arg_index: usize) -> &mut Self {
        let _ = self.abs_node(arg_index);
        self
    }

    /// Adds an absolute-value operation and returns its output node id.
    pub fn abs_node(&mut self, arg_index: usize) -> NodeId {
        self.push_node(MultiAD::Abs, vec![arg_index])
    }

    /// Adds an addition operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the left operand
    /// * `right_index` - Index of the right operand
    pub fn add(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        let _ = self.add_node(left_index, right_index);
        self
    }

    /// Adds an addition operation and returns its output node id.
    pub fn add_node(&mut self, left_index: usize, right_index: usize) -> NodeId {
        self.push_node(MultiAD::Add, vec![left_index, right_index])
    }

    /// Adds a subtraction operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the left operand
    /// * `right_index` - Index of the right operand
    pub fn sub(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        let _ = self.sub_node(left_index, right_index);
        self
    }

    /// Adds a subtraction operation and returns its output node id.
    pub fn sub_node(&mut self, left_index: usize, right_index: usize) -> NodeId {
        self.push_node(MultiAD::Sub, vec![left_index, right_index])
    }

    /// Adds a multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the left operand
    /// * `right_index` - Index of the right operand
    pub fn mul(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        let _ = self.mul_node(left_index, right_index);
        self
    }

    /// Adds a multiplication operation and returns its output node id.
    pub fn mul_node(&mut self, left_index: usize, right_index: usize) -> NodeId {
        self.push_node(MultiAD::Mul, vec![left_index, right_index])
    }

    /// Adds a division operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the numerator
    /// * `right_index` - Index of the denominator
    pub fn div(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        let _ = self.div_node(left_index, right_index);
        self
    }

    /// Adds a division operation and returns its output node id.
    pub fn div_node(&mut self, left_index: usize, right_index: usize) -> NodeId {
        self.push_node(MultiAD::Div, vec![left_index, right_index])
    }

    /// Adds a power operation.
    ///
    /// # Arguments
    ///
    /// * `base_index` - Index of the base
    /// * `exp_index` - Index of the exponent
    pub fn pow(&mut self, base_index: usize, exp_index: usize) -> &mut Self {
        let _ = self.pow_node(base_index, exp_index);
        self
    }

    /// Adds a power operation and returns its output node id.
    pub fn pow_node(&mut self, base_index: usize, exp_index: usize) -> NodeId {
        self.push_node(MultiAD::Pow, vec![base_index, exp_index])
    }

    /// Builds the final legacy computation graph.
    ///
    /// Returns a vector of `(operation, indices)` pairs that can be used
    /// with `MultiAD::compute()` and `MultiAD::compute_grad()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use petite_ad::{GraphBuilder, MultiAD};
    ///
    /// let graph = GraphBuilder::new(2)
    ///     .add(0, 1)
    ///     .sin(0)
    ///     .mul(2, 3)
    ///     .build();
    ///
    /// let inputs = &[0.6, 1.4];
    /// let (value, grad_fn) = MultiAD::compute_grad(&graph, inputs).unwrap();
    /// ```
    pub fn build(&self) -> Vec<(MultiAD, Vec<usize>)> {
        debug_assert!(
            !self
                .graph_nodes
                .iter()
                .any(|n| matches!(n, GraphNode::Constant(_))),
            "build() does not include constant nodes; use build_graph() instead"
        );
        self.operations.clone()
    }

    /// Builds a reusable [`Graph`] with node-handle semantics.
    #[must_use]
    pub fn build_graph(&self) -> Graph {
        let mut graph = Graph::new(self.num_inputs);
        for node in &self.graph_nodes {
            match node {
                GraphNode::Constant(value) => {
                    graph.constant(*value);
                }
                GraphNode::Operation { op, inputs } => {
                    graph.push_operation(*op, inputs.clone());
                }
            }
        }
        graph
    }

    /// Builds a reusable graph and selects an explicit output node.
    pub fn build_graph_with_output(&self, output: NodeId) -> crate::Result<Graph> {
        let mut graph = self.build_graph();
        graph.set_output(output)?;
        Ok(graph)
    }

    /// Builds a reusable graph and selects multiple explicit outputs.
    pub fn build_graph_with_outputs(&self, outputs: &[NodeId]) -> crate::Result<Graph> {
        let mut graph = self.build_graph();
        graph.set_outputs(outputs)?;
        Ok(graph)
    }

    /// Returns the current number of operations in the graph.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns true if the graph has no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Returns the next index that will be assigned to an operation.
    ///
    /// This is useful for chaining operations when you need to know
    /// what index the next operation will have.
    pub fn next_index(&self) -> usize {
        self.next_index
    }

    /// Adds a custom operation to the graph.
    ///
    /// This allows extending the builder with operations not directly
    /// supported by the fluent API.
    ///
    /// Input marker operations do not allocate a new graph value, matching
    /// [`GraphBuilder::input`]. All other operations increment `next_index()`.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to add
    /// * `indices` - Argument indices for the operation
    pub fn custom(&mut self, op: MultiAD, indices: Vec<usize>) -> &mut Self {
        let _ = self.custom_node(op, indices);
        self
    }

    /// Adds a custom operation and returns its output node id.
    ///
    /// For `Inp`, the referenced input index is returned because input markers
    /// do not allocate new graph values.
    pub fn custom_node(&mut self, op: MultiAD, indices: Vec<usize>) -> NodeId {
        if op == MultiAD::Inp {
            assert_eq!(
                indices.len(),
                1,
                "Inp requires exactly one index, got {}",
                indices.len()
            );
            let input_index = indices[0];
            self.operations.push((op, indices));
            input_index
        } else {
            self.push_node(op, indices)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi::first_order::MultiAD;
    use crate::test_utils::approx_eq_eps as approx_eq;

    #[test]
    fn test_builder_basic() {
        // Build: f(x, y) = x + y
        let graph = GraphBuilder::new(2).add(0, 1).build();

        let inputs = &[2.0, 3.0];
        let result = MultiAD::compute(&graph, inputs).unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_complex() {
        // Build: f(x, y) = sin(x) * (x + y)
        let graph = GraphBuilder::new(2)
            .add(0, 1) // x + y at index 2
            .sin(0) // sin(x) at index 3
            .mul(2, 3) // sin(x) * (x + y) at index 4
            .build();

        let inputs = &[0.6, 1.4];
        let (value, grad_fn) = MultiAD::compute_grad(&graph, inputs).unwrap();
        let grads = grad_fn(1.0);

        // Verify we get 2 gradients back
        assert_eq!(grads.len(), 2);

        // Value should be computed correctly
        let expected = 0.6_f64.sin() * (0.6 + 1.4);
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_builder_chaining() {
        // Build: f(x) = sin(cos(exp(x)))
        let graph = GraphBuilder::new(1)
            .exp(0) // exp(x) at index 1
            .cos(1) // cos(exp(x)) at index 2
            .sin(2) // sin(cos(exp(x))) at index 3
            .build();

        let inputs = &[0.5];
        let result = MultiAD::compute(&graph, inputs).unwrap();
        let expected = 0.5_f64.exp().cos().sin();
        assert!(approx_eq(result, expected, 1e-10));
    }

    #[test]
    fn test_builder_with_pow() {
        // Build: f(x, y, z) = x^y + z
        let graph = GraphBuilder::new(3)
            .pow(0, 1) // x^y at index 3
            .add(3, 2) // x^y + z at index 4
            .build();

        let inputs = &[2.0, 3.0, 1.0];
        let result = MultiAD::compute(&graph, inputs).unwrap();
        // 2^3 + 1 = 9
        assert!(approx_eq(result, 9.0, 1e-10));
    }

    #[test]
    fn test_builder_next_index() {
        let mut builder = GraphBuilder::new(2);
        assert_eq!(builder.next_index(), 2); // Start at 2 (after inputs)

        builder.input(0);
        assert_eq!(builder.next_index(), 2); // Input markers do not allocate values

        builder.add(0, 1);
        assert_eq!(builder.next_index(), 3); // After add operation

        builder.sin(0);
        assert_eq!(builder.next_index(), 4); // After sin operation

        builder.mul(2, 3);
        assert_eq!(builder.next_index(), 5); // After mul operation
    }

    #[test]
    fn test_builder_len_and_is_empty() {
        let mut builder = GraphBuilder::new(2);

        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());

        builder.add(0, 1);
        assert_eq!(builder.len(), 1);
        assert!(!builder.is_empty());

        builder.sin(0);
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_builder_custom_operation() {
        // Use custom to add an operation not in the fluent API
        let graph = GraphBuilder::new(2)
            .custom(MultiAD::Add, vec![0, 1])
            .build();

        let inputs = &[2.0, 3.0];
        let result = MultiAD::compute(&graph, inputs).unwrap();
        assert!(approx_eq(result, 5.0, 1e-10));
    }

    #[test]
    fn test_builder_custom_input_does_not_advance_index() {
        let mut builder = GraphBuilder::new(2);
        builder.custom(MultiAD::Inp, vec![0]);

        assert_eq!(builder.next_index(), 2);
    }

    #[test]
    fn test_builder_node_handle_api() {
        let mut builder = GraphBuilder::new(2);
        let x = builder.input_node(0);
        let y = builder.input_node(1);
        let sum = builder.add_node(x, y);
        let sin_x = builder.sin_node(x);
        let product = builder.mul_node(sum, sin_x);

        assert_eq!(sum, 2);
        assert_eq!(sin_x, 3);
        assert_eq!(product, 4);

        let graph = builder.build_graph();
        let value = graph.compute(&[0.6, 1.4]).unwrap();
        let expected = (0.6_f64 + 1.4_f64) * 0.6_f64.sin();
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_builder_build_graph_with_output() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let x_sq = builder.mul_node(x, x);
        builder.mul_node(x_sq, x);

        let graph = builder.build_graph_with_output(x_sq).unwrap();
        let value = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(value, 4.0, 1e-10));
    }

    #[test]
    fn test_builder_build_graph_with_outputs() {
        let mut builder = GraphBuilder::new(2);
        let x = builder.input_node(0);
        let y = builder.input_node(1);
        let sum = builder.add_node(x, y);
        let product = builder.mul_node(x, y);

        let graph = builder.build_graph_with_outputs(&[sum, product]).unwrap();
        let values = graph.compute_many(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(values[0], 5.0, 1e-10));
        assert!(approx_eq(values[1], 6.0, 1e-10));
    }

    // ---- Cover all builder operation methods ----

    #[test]
    fn test_builder_cos() {
        let graph = GraphBuilder::new(1).cos(0).build();
        let result = MultiAD::compute(&graph, &[0.0]).unwrap();
        assert!(approx_eq(result, 1.0, 1e-10));
    }

    #[test]
    fn test_builder_cos_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let cos_x = builder.cos_node(x);
        assert_eq!(cos_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(val, 1.0, 1e-10));
    }

    #[test]
    fn test_builder_tan() {
        let graph = GraphBuilder::new(1).tan(0).build();
        let result = MultiAD::compute(&graph, &[0.0]).unwrap();
        assert!(approx_eq(result, 0.0, 1e-10));
    }

    #[test]
    fn test_builder_tan_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let tan_x = builder.tan_node(x);
        assert_eq!(tan_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(val, 0.0, 1e-10));
    }

    #[test]
    fn test_builder_neg() {
        let graph = GraphBuilder::new(1).neg(0).build();
        let result = MultiAD::compute(&graph, &[3.0]).unwrap();
        assert!(approx_eq(result, -3.0, 1e-10));
    }

    #[test]
    fn test_builder_neg_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let neg_x = builder.neg_node(x);
        assert_eq!(neg_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(val, -3.0, 1e-10));
    }

    #[test]
    fn test_builder_exp() {
        let graph = GraphBuilder::new(1).exp(0).build();
        let result = MultiAD::compute(&graph, &[1.0]).unwrap();
        assert!(approx_eq(result, std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_builder_exp_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let exp_x = builder.exp_node(x);
        assert_eq!(exp_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[1.0]).unwrap();
        assert!(approx_eq(val, std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_builder_ln() {
        let graph = GraphBuilder::new(1).ln(0).build();
        let result = MultiAD::compute(&graph, &[std::f64::consts::E]).unwrap();
        assert!(approx_eq(result, 1.0, 1e-10));
    }

    #[test]
    fn test_builder_ln_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let ln_x = builder.ln_node(x);
        assert_eq!(ln_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[std::f64::consts::E]).unwrap();
        assert!(approx_eq(val, 1.0, 1e-10));
    }

    #[test]
    fn test_builder_sqrt() {
        let graph = GraphBuilder::new(1).sqrt(0).build();
        let result = MultiAD::compute(&graph, &[4.0]).unwrap();
        assert!(approx_eq(result, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_sqrt_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let sqrt_x = builder.sqrt_node(x);
        assert_eq!(sqrt_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[4.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_abs() {
        let graph = GraphBuilder::new(1).abs(0).build();
        let result = MultiAD::compute(&graph, &[-5.0]).unwrap();
        assert!(approx_eq(result, 5.0, 1e-10));
    }

    #[test]
    fn test_builder_abs_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let abs_x = builder.abs_node(x);
        assert_eq!(abs_x, 1);
        let graph = builder.build_graph();
        let val = graph.compute(&[-5.0]).unwrap();
        assert!(approx_eq(val, 5.0, 1e-10));
    }

    #[test]
    fn test_builder_sub() {
        let graph = GraphBuilder::new(2).sub(0, 1).build();
        let result = MultiAD::compute(&graph, &[5.0, 3.0]).unwrap();
        assert!(approx_eq(result, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_sub_node() {
        let mut builder = GraphBuilder::new(2);
        let x = builder.input_node(0);
        let y = builder.input_node(1);
        let diff = builder.sub_node(x, y);
        assert_eq!(diff, 2);
        let graph = builder.build_graph();
        let val = graph.compute(&[5.0, 3.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_div() {
        let graph = GraphBuilder::new(2).div(0, 1).build();
        let result = MultiAD::compute(&graph, &[6.0, 3.0]).unwrap();
        assert!(approx_eq(result, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_div_node() {
        let mut builder = GraphBuilder::new(2);
        let x = builder.input_node(0);
        let y = builder.input_node(1);
        let quotient = builder.div_node(x, y);
        assert_eq!(quotient, 2);
        let graph = builder.build_graph();
        let val = graph.compute(&[6.0, 3.0]).unwrap();
        assert!(approx_eq(val, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_pow_node() {
        let mut builder = GraphBuilder::new(2);
        let x = builder.input_node(0);
        let y = builder.input_node(1);
        let result = builder.pow_node(x, y);
        assert_eq!(result, 2);
        let graph = builder.build_graph();
        let val = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(val, 8.0, 1e-10));
    }

    #[test]
    fn test_builder_constant_node() {
        let mut builder = GraphBuilder::new(1);
        let x = builder.input_node(0);
        let c = builder.constant_node(3.0);
        let _sum = builder.add_node(x, c);

        let graph = builder.build_graph();
        let val = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(val, 5.0, 1e-10));
    }

    #[test]
    fn test_builder_custom_node_non_inp() {
        let mut builder = GraphBuilder::new(2);
        let x = builder.input_node(0);
        let y = builder.input_node(1);
        let node = builder.custom_node(MultiAD::Mul, vec![x, y]);
        assert_eq!(node, 2);
        let graph = builder.build_graph();
        let val = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(val, 6.0, 1e-10));
    }

    #[test]
    #[should_panic(expected = "Inp requires exactly one index")]
    fn test_builder_custom_node_inp_wrong_arity() {
        let mut builder = GraphBuilder::new(2);
        builder.custom_node(MultiAD::Inp, vec![0, 1]);
    }

    #[test]
    fn test_builder_fluent_all_unary_ops() {
        // Test sin via fluent API already tested above; test chaining multiple ops
        let graph = GraphBuilder::new(1)
            .exp(0) // exp(x) at index 1
            .ln(1) // ln(exp(x)) = x at index 2
            .build();
        let result = MultiAD::compute(&graph, &[2.0]).unwrap();
        assert!(approx_eq(result, 2.0, 1e-10));
    }

    #[test]
    fn test_builder_input_marker() {
        let graph = GraphBuilder::new(2).input(0).input(1).add(0, 1).build();
        let result = MultiAD::compute(&graph, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(result, 5.0, 1e-10));
    }

    #[test]
    fn test_builder_chain_multiple_binary_ops() {
        // f(x, y) = (x + y) - (x * y) => 5 - 6 = -1
        let graph = GraphBuilder::new(2)
            .add(0, 1) // index 2: x+y
            .mul(0, 1) // index 3: x*y
            .sub(2, 3) // index 4: (x+y)-(x*y)
            .build();
        let result = MultiAD::compute(&graph, &[2.0, 3.0]).unwrap();
        assert!(approx_eq(result, -1.0, 1e-10));
    }
}
