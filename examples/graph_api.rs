use petite_ad::{ExprGraph, Graph};

fn main() -> petite_ad::Result<()> {
    let mut graph = Graph::new(2);
    graph.set_input_name(0, "x")?;
    graph.set_input_name(1, "y")?;
    let x = graph.input(0);
    let y = graph.input(1);
    let product = graph.mul(x, y);
    let output = graph.add_const(product, 2.0);
    graph.set_output_name(output, "xy_plus_two")?;

    let (value, gradient) = graph.value_and_gradient(&[3.0, 4.0])?;
    println!("value = {value}");
    println!("gradient = {gradient:?}");
    println!("stats = {:?}", graph.stats());

    let expr_graph = ExprGraph::new(2);
    let x = expr_graph.input(0);
    let y = expr_graph.input(1);
    let expr_output = x.clone().sin() * (x + y) + 2.0;
    expr_graph.set_output(&expr_output)?;
    let expr_value = expr_graph.graph().compute(&[0.6, 1.4])?;
    println!("expression graph value = {expr_value}");

    Ok(())
}
