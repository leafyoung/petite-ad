use std::sync::Arc;

#[allow(unused)]
pub type MathFn = fn(f64) -> f64;
pub type DynMathFn = dyn Fn(f64) -> f64;
pub type BackwardResult = (f64, Box<DynMathFn>);

pub type BackwardResultArc = (f64, Arc<DynMathFn>);

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathOp {
    Sin,
    Cos,
    Exp,
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

    pub fn compute(exprs: &[MathOp], x: f64) -> BackwardResult {
        let mut value = x;
        let mut backprops = Vec::new();

        // Compute backward pass for each operation
        for &op in exprs {
            let (new_value, backprop) = op.backward(value);
            value = new_value;
            backprops.push(backprop);
        }

        // Chain all the backward functions
        let backward_fn = move |cotangent: f64| -> f64 {
            let mut grad = cotangent;
            for backprop in backprops.iter().rev() {
                grad = backprop(grad);
            }
            grad
        };

        (value, Box::new(backward_fn))
    }

    pub fn compute_arc(exprs: &[MathOp], x: f64) -> BackwardResultArc {
        let mut value = x;
        let mut backprops = Vec::new();

        // Compute backward pass for each operation
        for &op in exprs {
            let (new_value, backprop) = op.backward_arc(value);
            value = new_value;
            backprops.push(backprop);
        }

        // Chain all the backward functions
        let backward_fn = move |cotangent: f64| -> f64 {
            let mut grad = cotangent;
            for backprop in backprops.iter().rev() {
                grad = backprop(grad);
            }
            grad
        };

        (value, Arc::new(backward_fn))
    }
}
