mod macros;
mod mono;
mod multi;

use mono::{MF1, MonoAD, MonoFn};
use multi::{F1, F2, F3, MultiAD, MultiFn};

fn main() {
    println!("--- Mono-variable automatic differentiation ---");
    let mf1 = MF1(2.0);
    mf1.demonstrate();

    println!("\n--- Multi-variable automatic differentiation ---");

    // Build computation graph using F1's graph method: f(x₁, x₂) = sin(x₁) * (x₁ + x₂)
    let x1 = 0.4;
    let x2 = 1.6;
    let formula = F1(x1, x2); // Values don't matter for graph construction
    formula.demonstrate();

    let formula = F2(x1, x2); // Values don't matter for graph construction
    formula.demonstrate();

    let formula = F3(x1, x2); // Values don't matter for graph construction
    formula.demonstrate();
}
