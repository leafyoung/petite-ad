use petite_ad::{Adam, Graph};

fn main() -> petite_ad::Result<()> {
    let mut graph = Graph::new(1);
    let x = graph.input(0);
    let shifted = graph.sub_const(x, 3.0);
    graph.square(shifted);

    let mut params = vec![0.0];
    let mut optimizer = Adam::new(1, 0.1);
    for _ in 0..100 {
        let (_value, gradient) = graph.value_and_gradient(&params)?;
        optimizer.step(&mut params, &gradient)?;
    }

    println!(
        "x = {:.6}, f(x) = {:.6}",
        params[0],
        graph.compute(&params)?
    );
    Ok(())
}
