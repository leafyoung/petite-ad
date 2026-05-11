//! Expression graph with operator-overloaded construction.

use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use super::graph::{Graph, NodeId};
use super::multi_ad::MultiAD;
use crate::Result;

/// Shared expression graph used for operator-overloaded graph construction.
#[derive(Debug, Clone)]
pub struct ExprGraph {
    graph: Rc<RefCell<Graph>>,
}

/// A node handle tied to an [`ExprGraph`].
#[derive(Debug, Clone)]
pub struct ExprNode {
    graph: Rc<RefCell<Graph>>,
    node: NodeId,
}

impl ExprGraph {
    /// Create an expression graph with `num_inputs` input variables.
    #[must_use]
    pub fn new(num_inputs: usize) -> Self {
        Self {
            graph: Rc::new(RefCell::new(Graph::new(num_inputs))),
        }
    }

    /// Return an expression node for an input.
    #[must_use]
    pub fn input(&self, input_index: usize) -> ExprNode {
        let node = self.graph.borrow().input(input_index);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    /// Return an expression node for a literal constant.
    pub fn constant(&self, value: f64) -> ExprNode {
        let node = self.graph.borrow_mut().constant(value);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    /// Select an expression node as the graph output.
    pub fn set_output(&self, expr: &ExprNode) -> Result<()> {
        self.graph.borrow_mut().set_output(expr.node)?;
        Ok(())
    }

    /// Clone out the underlying reusable graph.
    #[must_use]
    pub fn graph(&self) -> Graph {
        self.graph.borrow().clone()
    }
}

impl ExprNode {
    /// Return the underlying node id.
    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.node
    }

    fn same_graph(&self, other: &ExprNode) {
        assert!(
            Rc::ptr_eq(&self.graph, &other.graph),
            "ExprNode graph mismatch"
        );
    }

    fn unary(&self, op: MultiAD) -> ExprNode {
        let node = self.graph.borrow_mut().push_operation(op, vec![self.node]);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    fn binary(&self, op: MultiAD, other: &ExprNode) -> ExprNode {
        self.same_graph(other);
        let node = self
            .graph
            .borrow_mut()
            .push_operation(op, vec![self.node, other.node]);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    fn binary_const(&self, op: MultiAD, value: f64) -> ExprNode {
        let mut graph = self.graph.borrow_mut();
        let constant = graph.constant(value);
        let node = graph.push_operation(op, vec![self.node, constant]);
        ExprNode {
            graph: Rc::clone(&self.graph),
            node,
        }
    }

    /// Append `sin(self)`.
    pub fn sin(&self) -> ExprNode {
        self.unary(MultiAD::Sin)
    }

    /// Append `cos(self)`.
    pub fn cos(&self) -> ExprNode {
        self.unary(MultiAD::Cos)
    }

    /// Append `exp(self)`.
    pub fn exp(&self) -> ExprNode {
        self.unary(MultiAD::Exp)
    }

    /// Append `ln(self)`.
    pub fn ln(&self) -> ExprNode {
        self.unary(MultiAD::Ln)
    }

    /// Append `sqrt(self)`.
    pub fn sqrt(&self) -> ExprNode {
        self.unary(MultiAD::Sqrt)
    }
}

impl Add for ExprNode {
    type Output = ExprNode;

    fn add(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Add, &rhs)
    }
}

impl Add<f64> for ExprNode {
    type Output = ExprNode;

    fn add(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Add, rhs)
    }
}

impl Sub for ExprNode {
    type Output = ExprNode;

    fn sub(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Sub, &rhs)
    }
}

impl Sub<f64> for ExprNode {
    type Output = ExprNode;

    fn sub(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Sub, rhs)
    }
}

impl Mul for ExprNode {
    type Output = ExprNode;

    fn mul(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Mul, &rhs)
    }
}

impl Mul<f64> for ExprNode {
    type Output = ExprNode;

    fn mul(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Mul, rhs)
    }
}

impl Div for ExprNode {
    type Output = ExprNode;

    fn div(self, rhs: ExprNode) -> Self::Output {
        self.binary(MultiAD::Div, &rhs)
    }
}

impl Div<f64> for ExprNode {
    type Output = ExprNode;

    fn div(self, rhs: f64) -> Self::Output {
        self.binary_const(MultiAD::Div, rhs)
    }
}

impl Neg for ExprNode {
    type Output = ExprNode;

    fn neg(self) -> Self::Output {
        self.unary(MultiAD::Neg)
    }
}
