use petite_ad::Graph;

fn main() -> petite_ad::Result<()> {
    // Minimize f(x) = (x - 3)^2 using Newton's method.
    let mut graph = Graph::new(1);
    let x = graph.input(0);
    let shifted = graph.sub_const(x, 3.0);
    graph.square(shifted);

    let mut point = vec![0.0];
    for _step in 0..5 {
        let (_value, gradient) = graph.value_and_gradient(&point)?;
        let hessian = graph.exact_hessian_rr(&point)?;
        point[0] -= gradient[0] / hessian[0][0];
    }

    let value = graph.compute(&point)?;
    println!("x = {:.6}, f(x) = {:.6}", point[0], value);
    Ok(())
}
