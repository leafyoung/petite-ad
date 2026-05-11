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
    /// The computation graph is malformed.
    InvalidGraph {
        /// Human-readable reason for the graph validation failure.
        reason: &'static str,
    },
    /// An index references a non-existent value in the computation.
    IndexOutOfBounds {
        /// The invalid index
        index: usize,
        /// The maximum valid index
        max_index: usize,
    },
    /// A real-domain restriction was violated in checked evaluation mode.
    DomainError {
        /// Name of the operation.
        operation: &'static str,
        /// Human-readable reason for the domain failure.
        reason: &'static str,
    },
    /// Invalid arguments were passed to a function.
    InvalidArguments {
        /// Human-readable reason for the argument validation failure.
        reason: &'static str,
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
            AutodiffError::InvalidGraph { reason } => {
                write!(f, "Computation graph is invalid: {}", reason)
            }
            AutodiffError::IndexOutOfBounds { index, max_index } => {
                write!(f, "Index {} is out of bounds (max: {})", index, max_index)
            }
            AutodiffError::DomainError { operation, reason } => {
                write!(f, "Domain error in {}: {}", operation, reason)
            }
            AutodiffError::InvalidArguments { reason } => {
                write!(f, "Invalid arguments: {}", reason)
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

    /// Create a domain error for checked evaluation mode.
    pub fn domain(operation: &'static str, reason: &'static str) -> Self {
        AutodiffError::DomainError { operation, reason }
    }

    /// Create an invalid-arguments error.
    pub fn invalid_arguments(reason: &'static str) -> Self {
        AutodiffError::InvalidArguments { reason }
    }
}

/// Result type for automatic differentiation operations.
pub type Result<T> = std::result::Result<T, AutodiffError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_empty_graph() {
        let err = AutodiffError::EmptyGraph;
        assert_eq!(format!("{}", err), "Computation graph is empty");
    }

    #[test]
    fn test_display_index_out_of_bounds() {
        let err = AutodiffError::IndexOutOfBounds {
            index: 5,
            max_index: 3,
        };
        assert_eq!(format!("{}", err), "Index 5 is out of bounds (max: 3)");
    }

    #[test]
    fn test_display_domain_error() {
        let err = AutodiffError::DomainError {
            operation: "Sqrt",
            reason: "input must be non-negative",
        };
        assert_eq!(
            format!("{}", err),
            "Domain error in Sqrt: input must be non-negative"
        );
    }

    #[test]
    fn test_display_arity_error() {
        let err = AutodiffError::ArityError {
            operation: "Mul",
            expected: 2,
            actual: 1,
        };
        assert_eq!(format!("{}", err), "Arity error in Mul: expected 2, got 1");
    }

    #[test]
    fn test_display_invalid_graph() {
        let err = AutodiffError::InvalidGraph { reason: "bad node" };
        assert_eq!(format!("{}", err), "Computation graph is invalid: bad node");
    }

    #[test]
    fn test_arity_constructor() {
        let err = AutodiffError::arity("Mul", 2, 1);
        assert_eq!(
            err,
            AutodiffError::ArityError {
                operation: "Mul",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn test_check_arity_success() {
        assert!(AutodiffError::check_arity("Sin", 1, 1).is_ok());
    }

    #[test]
    fn test_check_arity_failure() {
        let err = AutodiffError::check_arity("Sin", 1, 2).unwrap_err();
        assert_eq!(
            err,
            AutodiffError::ArityError {
                operation: "Sin",
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn test_domain_constructor() {
        let err = AutodiffError::domain("Ln", "input must be positive");
        assert_eq!(
            err,
            AutodiffError::DomainError {
                operation: "Ln",
                reason: "input must be positive",
            }
        );
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AutodiffError>();
    }
}
