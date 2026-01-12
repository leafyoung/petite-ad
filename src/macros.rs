/// Macro to convert function names to MonoAD enum at compile time.
/// This avoids the function pointer comparison issue across library boundaries.
///
/// # Example
/// ```
/// use autodiff::{mono_ops, MonoAD};
///
/// let (value, backprop) = MonoAD::compute_grad(&mono_ops![sin, sin, exp], 2.0);
/// println!("backprop: {} {}", value, backprop(1.0));
/// ```
///
#[macro_export]
macro_rules! mono_ops {
    (@one sin) => { $crate::MonoAD::Sin };
    (@one cos) => { $crate::MonoAD::Cos };
    (@one exp) => { $crate::MonoAD::Exp };
    (@one $x:ident) => {
        compile_error!(concat!("Unsupported math operation: ", stringify!($x), ". Use: sin, cos, or exp"))
    };
    ($($x:ident),* $(,)?) => {
        [$($crate::mono_ops!(@one $x)),*]
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
/// - `sin`, `cos`, `tan`, `exp`, `ln` - Unary operations (takes single index)
///
/// # Example
/// ```
/// use autodiff::{multi_ops, MultiAD};
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
/// let (value, grad_fn) = MultiAD::compute_grad(&exprs, &[0.6, 1.4]);
/// ```
///
#[macro_export]
macro_rules! multi_ops {
    // Unary operations
    (@op sin) => { $crate::MultiAD::Sin };
    (@op cos) => { $crate::MultiAD::Cos };
    (@op tan) => { $crate::MultiAD::Tan };
    (@op exp) => { $crate::MultiAD::Exp };
    (@op ln) => { $crate::MultiAD::Ln };
    // Binary operations
    (@op add) => { $crate::MultiAD::Add };
    (@op sub) => { $crate::MultiAD::Sub };
    (@op mul) => { $crate::MultiAD::Mul };
    (@op div) => { $crate::MultiAD::Div };
    // Input
    (@op inp) => { $crate::MultiAD::Inp };
    // Error for unknown operations
    (@op $x:ident) => {
        compile_error!(
            concat!(
                "Unsupported operation: ",
                stringify!($x),
                ". Use: inp, add, sub, mul, div, sin, cos, tan, exp, or ln"
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
