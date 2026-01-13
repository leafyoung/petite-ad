//! Error types for automatic differentiation operations.

use std::fmt;

/// Errors that can occur during automatic differentiation computations.
#[derive(Debug, Clone, PartialEq)]
pub enum AutodiffError {
    /// An operation received an incorrect number of arguments (specific arity error).
    ArityError {
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
            AutodiffError::ArityError {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "Arity error in {}: expected {}, got {}",
                operation, expected, actual
            ),
            AutodiffError::EmptyGraph => write!(f, "Computation graph is empty"),
            AutodiffError::IndexOutOfBounds { index, max_index } => {
                write!(f, "Index {} is out of bounds (max: {})", index, max_index)
            }
        }
    }
}

impl std::error::Error for AutodiffError {}

impl AutodiffError {
    /// Create an ArityError for an operation with incorrect argument count.
    pub fn arity(operation: &'static str, expected: usize, actual: usize) -> Self {
        AutodiffError::ArityError {
            operation,
            expected,
            actual,
        }
    }

    /// Validate that an operation received the correct number of arguments.
    pub fn check_arity(
        operation: &'static str,
        expected: usize,
        actual: usize,
    ) -> std::result::Result<(), AutodiffError> {
        if actual == expected {
            Ok(())
        } else {
            Err(AutodiffError::arity(operation, expected, actual))
        }
    }
}

/// Result type for automatic differentiation operations.
pub type Result<T> = std::result::Result<T, AutodiffError>;
