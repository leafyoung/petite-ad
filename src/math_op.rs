use std::sync::Arc;

pub type MathFn = fn(f64) -> f64;
pub type DynMathFn = dyn Fn(f64) -> f64;
pub type BackwardResult = (f64, Box<DynMathFn>);

pub type BackwardResultArc = (f64, Arc<DynMathFn>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathOp {
    Sin,
    Cos,
    Exp,
}

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

impl MathOp {
    pub fn backward(self, x: f64) -> (f64, Box<DynMathFn>) {
        match self {
            MathOp::Sin => {
                let y = x.sin();
                let grad = move |dy: f64| -> f64 { dy * x.cos() };
                (y, Box::new(grad))
            }
            MathOp::Cos => {
                let y = x.cos();
                let grad = move |dy: f64| -> f64 { dy * -x.sin() };
                (y, Box::new(grad))
            }
            MathOp::Exp => {
                let y = x.exp();
                let grad = move |dy: f64| -> f64 { dy * y };
                (y, Box::new(grad))
            }
        }
    }

    pub fn backward_arc(self, x: f64) -> (f64, Arc<DynMathFn>) {
        match self {
            MathOp::Sin => {
                let y = x.sin();
                let grad = move |dy: f64| -> f64 { dy * x.cos() };
                (y, Arc::new(grad))
            }
            MathOp::Cos => {
                let y = x.cos();
                let grad = move |dy: f64| -> f64 { dy * -x.sin() };
                (y, Arc::new(grad))
            }
            MathOp::Exp => {
                let y = x.exp();
                let grad = move |dy: f64| -> f64 { dy * y };
                (y, Arc::new(grad))
            }
        }
    }
}
