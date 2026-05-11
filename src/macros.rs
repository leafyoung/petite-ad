//! Convenience macros for building single-variable and multi-variable
//! computational graphs concisely.
//!
//! - [`mono_ops!`] — Build a `Vec<MonoAD>` expression with sinusoidal, exp, ln, sqrt, etc.
//! - [`multi_ops!`] — Build a `Vec<(MultiAD, Vec<usize>)>` tuple graph with input markers.

/// Macro to convert function names to MonoAD enum at compile time.
/// This avoids the function pointer comparison issue across library boundaries.
///
/// # Example
/// ```
/// use petite_ad::{mono_ops, MonoAD};
///
/// let (value, backprop) = MonoAD::compute_grad(&mono_ops![sin, tan, exp], 2.0);
/// println!("backprop: {} {}", value, backprop(1.0));
/// ```
///
#[macro_export]
macro_rules! mono_ops {
    (@one sin) => { $crate::MonoAD::Sin };
    (@one cos) => { $crate::MonoAD::Cos };
    (@one tan) => { $crate::MonoAD::Tan };
    (@one exp) => { $crate::MonoAD::Exp };
    (@one neg) => { $crate::MonoAD::Neg };
    (@one ln) => { $crate::MonoAD::Ln };
    (@one sqrt) => { $crate::MonoAD::Sqrt };
    (@one abs) => { $crate::MonoAD::Abs };
    (@one $x:ident) => {
        compile_error!(concat!("Unsupported math operation: ", stringify!($x), ". Use: sin, cos, tan, exp, neg, ln, sqrt, or abs"))
    };
    ($($x:ident),* $(,)?) => {
        [$($crate::mono_ops!(@one $x)),*]
    };
}

/// Macro for MonoAD2RR (Reverse-over-Reverse) operations.
#[macro_export]
macro_rules! mono_ops_rr {
    (@one sin) => { $crate::MonoAD2RR::Sin };
    (@one cos) => { $crate::MonoAD2RR::Cos };
    (@one tan) => { $crate::MonoAD2RR::Tan };
    (@one exp) => { $crate::MonoAD2RR::Exp };
    (@one neg) => { $crate::MonoAD2RR::Neg };
    (@one ln) => { $crate::MonoAD2RR::Ln };
    (@one sqrt) => { $crate::MonoAD2RR::Sqrt };
    (@one abs) => { $crate::MonoAD2RR::Abs };
    (@one $x:ident) => {
        compile_error!(concat!("Unsupported math operation: ", stringify!($x), ". Use: sin, cos, tan, exp, neg, ln, sqrt, or abs"))
    };
    ($($x:ident),* $(,)?) => {
        [$($crate::mono_ops_rr!(@one $x)),*]
    };
}

/// Macro for MonoAD2FR (Forward-over-Reverse) operations.
#[macro_export]
macro_rules! mono_ops_fr {
    (@one sin) => { $crate::MonoAD2FR::Sin };
    (@one cos) => { $crate::MonoAD2FR::Cos };
    (@one tan) => { $crate::MonoAD2FR::Tan };
    (@one exp) => { $crate::MonoAD2FR::Exp };
    (@one neg) => { $crate::MonoAD2FR::Neg };
    (@one ln) => { $crate::MonoAD2FR::Ln };
    (@one sqrt) => { $crate::MonoAD2FR::Sqrt };
    (@one abs) => { $crate::MonoAD2FR::Abs };
    (@one $x:ident) => {
        compile_error!(concat!("Unsupported math operation: ", stringify!($x), ". Use: sin, cos, tan, exp, neg, ln, sqrt, or abs"))
    };
    ($($x:ident),* $(,)?) => {
        [$($crate::mono_ops_fr!(@one $x)),*]
    };
}

/// Macro for MonoAD2RF (Reverse-over-Forward) operations.
#[macro_export]
macro_rules! mono_ops_rf {
    (@one sin) => { $crate::MonoAD2RF::Sin };
    (@one cos) => { $crate::MonoAD2RF::Cos };
    (@one tan) => { $crate::MonoAD2RF::Tan };
    (@one exp) => { $crate::MonoAD2RF::Exp };
    (@one neg) => { $crate::MonoAD2RF::Neg };
    (@one ln) => { $crate::MonoAD2RF::Ln };
    (@one sqrt) => { $crate::MonoAD2RF::Sqrt };
    (@one abs) => { $crate::MonoAD2RF::Abs };
    (@one $x:ident) => {
        compile_error!(concat!("Unsupported math operation: ", stringify!($x), ". Use: sin, cos, tan, exp, neg, ln, sqrt, or abs"))
    };
    ($($x:ident),* $(,)?) => {
        [$($crate::mono_ops_rf!(@one $x)),*]
    };
}

/// Macro to build multi-variable computation graphs with lowercase operation names.
/// Converts lowercase identifiers to MultiAD enum variants.
///
/// # Syntax
/// Each operation is written as `(op, indices...)` where:
/// - `op` is the operation name (lowercase)
/// - `indices...` are comma-separated argument indices
///
/// # Supported Operations
/// - `inp` - Input placeholder (takes single index: the input number)
/// - `add`, `sub`, `mul`, `div` - Binary operations (takes two indices)
/// - `pow` - Power operation (takes two indices: base, exponent)
/// - `sin`, `cos`, `tan`, `exp`, `ln` - Unary operations (takes single index)
/// - `sqrt`, `abs`, `log1p_exp` - Unary operations (takes single index)
/// - `log_add_exp` - Binary stable log-sum-exp (takes two indices)
///
/// # Example
/// ```
/// use petite_ad::{multi_ops, MultiAD};
///
/// // Build: f(x, y) = sin(x) * (x + y)
/// let exprs = multi_ops![
///     (inp, 0),      // x at index 0
///     (inp, 1),      // y at index 1
///     (add, 0, 1),   // x + y at index 2
///     (sin, 0),      // sin(x) at index 3
///     (mul, 2, 3),   // sin(x) * (x + y) at index 4
/// ];
///
/// let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]).unwrap();
/// ```
///
#[macro_export]
macro_rules! multi_ops {
    // Unary operations
    (@op sin) => { $crate::MultiAD::Sin };
    (@op cos) => { $crate::MultiAD::Cos };
    (@op tan) => { $crate::MultiAD::Tan };
    (@op tanh) => { $crate::MultiAD::Tanh };
    (@op relu) => { $crate::MultiAD::Relu };
    (@op log1p_exp) => { $crate::MultiAD::Log1pExp };
    (@op neg) => { $crate::MultiAD::Neg };
    (@op exp) => { $crate::MultiAD::Exp };
    (@op ln) => { $crate::MultiAD::Ln };
    (@op sqrt) => { $crate::MultiAD::Sqrt };
    (@op abs) => { $crate::MultiAD::Abs };
    // Binary operations
    (@op add) => { $crate::MultiAD::Add };
    (@op sub) => { $crate::MultiAD::Sub };
    (@op mul) => { $crate::MultiAD::Mul };
    (@op div) => { $crate::MultiAD::Div };
    (@op pow) => { $crate::MultiAD::Pow };
    (@op log_add_exp) => { $crate::MultiAD::LogAddExp };
    // Input
    (@op inp) => { $crate::MultiAD::Inp };
    // Error for unknown operations
    (@op $x:ident) => {
        compile_error!(
            concat!(
                "Unsupported operation: ",
                stringify!($x),
                ". Use: inp, add, sub, mul, div, pow, log_add_exp, sin, cos, tan, tanh, relu, log1p_exp, neg, exp, ln, sqrt, or abs"
            )
        )
    };
    // Main parsing rule: (op, indices...)
    (@one ($op:ident, $($idx:expr),+)) => {
        ($crate::multi_ops!(@op $op), vec![$($idx),+])
    };
    // Entry point: parse all tuples
    ($(($op:ident, $($idx:expr),+)),* $(,)?) => {
        [$($crate::multi_ops!(@one ($op, $($idx),+))),*]
    };
}
