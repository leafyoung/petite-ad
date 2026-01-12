//! Builder API for constructing multi-variable computation graphs.
//!
//! This module provides a fluent, type-safe interface for building computational
//! graphs without manually managing indices and vectors.

use super::multi_ad::MultiAD;

/// Builder for constructing multi-variable computation graphs.
///
/// Provides a fluent API for building computational graphs without manually
/// tracking indices. The builder automatically manages node indexing and
/// produces the final graph structure.
///
/// # Examples
///
/// ```rust
/// use petite_ad::{GraphBuilder, MultiAD};
///
/// // Build: f(x, y) = sin(x) * (x + y)
/// let graph = GraphBuilder::new(2)  // 2 inputs
///     .add(0, 1)      // x + y at index 2
///     .sin(0)         // sin(x) at index 3
///     .mul(2, 3)      // sin(x) * (x + y) at index 4
///     .build();
///
/// let inputs = &[0.6, 1.4];
/// let (value, grad_fn) = MultiAD::compute_grad(&graph, inputs).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    /// Number of input variables
    #[allow(dead_code)]
    num_inputs: usize,
    /// Operations in the computation graph
    operations: Vec<(MultiAD, Vec<usize>)>,
    /// Next available index for new operations
    next_index: usize,
}

impl GraphBuilder {
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
            next_index: num_inputs,
        }
    }

    /// Adds an input placeholder operation.
    ///
    /// This is rarely needed directly as inputs are automatically available,
    /// but can be useful for explicit graph construction.
    ///
    /// # Arguments
    ///
    /// * `input_index` - Which input variable to reference (0 to num_inputs-1)
    pub fn input(&mut self, input_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Inp, vec![input_index]));
        self.next_index += 1;
        self
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
        self.operations.push((MultiAD::Sin, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds a cosine operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn cos(&mut self, arg_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Cos, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds a tangent operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn tan(&mut self, arg_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Tan, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds an exponential operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn exp(&mut self, arg_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Exp, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds a natural logarithm operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn ln(&mut self, arg_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Ln, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds a square root operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn sqrt(&mut self, arg_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Sqrt, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds an absolute value operation.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - Index of the input value
    pub fn abs(&mut self, arg_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Abs, vec![arg_index]));
        self.next_index += 1;
        self
    }

    /// Adds an addition operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the left operand
    /// * `right_index` - Index of the right operand
    pub fn add(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Add, vec![left_index, right_index]));
        self.next_index += 1;
        self
    }

    /// Adds a subtraction operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the left operand
    /// * `right_index` - Index of the right operand
    pub fn sub(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Sub, vec![left_index, right_index]));
        self.next_index += 1;
        self
    }

    /// Adds a multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the left operand
    /// * `right_index` - Index of the right operand
    pub fn mul(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Mul, vec![left_index, right_index]));
        self.next_index += 1;
        self
    }

    /// Adds a division operation.
    ///
    /// # Arguments
    ///
    /// * `left_index` - Index of the numerator
    /// * `right_index` - Index of the denominator
    pub fn div(&mut self, left_index: usize, right_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Div, vec![left_index, right_index]));
        self.next_index += 1;
        self
    }

    /// Adds a power operation.
    ///
    /// # Arguments
    ///
    /// * `base_index` - Index of the base
    /// * `exp_index` - Index of the exponent
    pub fn pow(&mut self, base_index: usize, exp_index: usize) -> &mut Self {
        self.operations.push((MultiAD::Pow, vec![base_index, exp_index]));
        self.next_index += 1;
        self
    }

    /// Builds the final computation graph.
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
        self.operations.clone()
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
    /// # Arguments
    ///
    /// * `op` - The operation to add
    /// * `indices` - Argument indices for the operation
    pub fn custom(&mut self, op: MultiAD, indices: Vec<usize>) -> &mut Self {
        self.operations.push((op, indices));
        self.next_index += 1;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi::multi_ad::MultiAD;
    use crate::test_utils::approx_eq_eps as approx_eq;

    #[test]
    fn test_builder_basic() {
        // Build: f(x, y) = x + y
        let graph = GraphBuilder::new(2)
            .add(0, 1)
            .build();

        let inputs = &[2.0, 3.0];
        let result = MultiAD::compute(&graph, inputs).unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_complex() {
        // Build: f(x, y) = sin(x) * (x + y)
        let graph = GraphBuilder::new(2)
            .add(0, 1)      // x + y at index 2
            .sin(0)         // sin(x) at index 3
            .mul(2, 3)      // sin(x) * (x + y) at index 4
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
            .exp(0)        // exp(x) at index 1
            .cos(1)        // cos(exp(x)) at index 2
            .sin(2)        // sin(cos(exp(x))) at index 3
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
            .pow(0, 1)     // x^y at index 3
            .add(3, 2)     // x^y + z at index 4
            .build();

        let inputs = &[2.0, 3.0, 1.0];
        let result = MultiAD::compute(&graph, inputs).unwrap();
        // 2^3 + 1 = 9
        assert!(approx_eq(result, 9.0, 1e-10));
    }

    #[test]
    fn test_builder_next_index() {
        let mut builder = GraphBuilder::new(2);
        assert_eq!(builder.next_index(), 2);  // Start at 2 (after inputs)

        builder.add(0, 1);
        assert_eq!(builder.next_index(), 3);  // After add operation

        builder.sin(0);
        assert_eq!(builder.next_index(), 4);  // After sin operation

        builder.mul(2, 3);
        assert_eq!(builder.next_index(), 5);  // After mul operation
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
}
