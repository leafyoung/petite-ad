use std::sync::Arc;

use super::types::*;

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonoAD {
    Sin,
    Cos,
    Exp,
}

impl MonoAD {
    pub fn compute(exprs: &[MonoAD], x: f64) -> f64 {
        let mut value = x;
        for expr in exprs {
            value = expr.backward_generic::<Box<DynMathFn>>(value).0;
        }
        value
    }

    // Generic helper that works with any wrapper type
    // Box<dyn Fn> is the common type that all arms return
    fn backward_generic<W>(self, x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>>,
    {
        let (y, grad_fn): (f64, Box<dyn Fn(f64) -> f64>) = match self {
            MonoAD::Sin => {
                let y = x.sin();
                let grad = Box::new(move |dy: f64| -> f64 { dy * x.cos() });
                (y, grad)
            }
            MonoAD::Cos => {
                let y = x.cos();
                let grad = Box::new(move |dy: f64| -> f64 { dy * -x.sin() });
                (y, grad)
            }
            MonoAD::Exp => {
                let y = x.exp();
                let grad = Box::new(move |dy: f64| -> f64 { dy * y });
                (y, grad)
            }
        };
        // For backward(): Box::from(boxed_closure) → returns the Box as-is (identity)
        // For backward_arc(): Arc::from(boxed_closure) → converts Box to Arc
        (y, W::from(grad_fn))
    }

    // Generic helper for compute operations
    // MathOp shall be reversed outside before calling this function
    fn compute_grad_generic<W>(exprs: &[MonoAD], x: f64) -> (f64, W)
    where
        W: From<Box<DynMathFn>> + std::ops::Deref<Target = DynMathFn> + 'static,
    {
        let mut value = x;
        let mut backprops: Vec<W> = Vec::new();

        // Compute backward pass for each operation
        for &op in exprs {
            let (new_value, backprop) = op.backward_generic(value);
            value = new_value;
            backprops.push(backprop);
        }

        // Chain all the backward functions
        let backward_fn = Box::new(move |cotangent: f64| -> f64 {
            let mut grad = cotangent;
            for backprop in backprops.iter().rev() {
                grad = backprop(grad);
            }
            grad
        });

        (value, W::from(backward_fn))
    }

    pub fn compute_grad(exprs: &[MonoAD], x: f64) -> BackwardResultBox {
        Self::compute_grad_generic::<Box<DynMathFn>>(exprs, x)
    }

    pub fn compute_grad_arc(exprs: &[MonoAD], x: f64) -> BackwardResultArc {
        Self::compute_grad_generic::<Arc<DynMathFn>>(exprs, x)
    }
}
