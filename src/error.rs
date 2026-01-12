//! Error types for automatic differentiation operations.

use std::fmt;

/// Errors that can occur during automatic differentiation computations.
#[derive(Debug, Clone, PartialEq)]
pub enum AutodiffError {
    /// An operation received an incorrect number of arguments.
    InvalidArgumentCount {
        /// Name of the operation
        operation: &'static str,
        /// Expected number of arguments
        expected: usize,
        /// Actual number of arguments received
        actual: usize,
    },
    /// The computation graph is empty or invalid.
    EmptyGraph,
    /// An index references a non-existent value in the computation.
    IndexOutOfBounds {
        /// The invalid index
        index: usize,
        /// The maximum valid index
        max_index: usize,
    },
}

impl fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutodiffError::InvalidArgumentCount {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{} expects {} argument(s), but received {}",
                operation, expected, actual
            ),
            AutodiffError::EmptyGraph => write!(f, "Computation graph is empty"),
            AutodiffError::IndexOutOfBounds { index, max_index } => write!(
                f,
                "Index {} is out of bounds (max: {})",
                index, max_index
            ),
        }
    }
}

impl std::error::Error for AutodiffError {}

/// Result type for automatic differentiation operations.
pub type Result<T> = std::result::Result<T, AutodiffError>;
