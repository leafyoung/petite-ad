/// Macro to convert function names to MathOp enum at compile time.
/// This avoids the function pointer comparison issue across library boundaries.
///
/// # Example
/// ```
/// use autodiff::{compute, math_ops};
///
/// let (value, backprop) = compute(math_ops![sin, sin, exp], 2.0);
/// ```
///
#[macro_export]
macro_rules! math_ops {
    (@one sin) => { $crate::MathOp::Sin };
    (@one cos) => { $crate::MathOp::Cos };
    (@one exp) => { $crate::MathOp::Exp };
    (@one $x:ident) => {
        compile_error!(concat!("Unsupported math operation: ", stringify!($x), ". Use: sin, cos, or exp"))
    };
    ($($x:ident),* $(,)?) => {
        [$($crate::math_ops!(@one $x)),*]
    };
}
