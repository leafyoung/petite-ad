mod math_op;
mod single_backprop;
mod single_backprop_arc;

pub use math_op::{BackwardResult, BackwardResultArc, MathFn, MathOp};
pub use single_backprop::compute;
pub use single_backprop_arc::compute as compute_arc;
