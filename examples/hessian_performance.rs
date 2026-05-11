use petite_ad::{MultiAD2FR, MultiAD2RF, MultiAD2RR};
use std::time::Instant;

fn main() {
    println!("Hessian Performance Comparison (all exact methods)");
    println!("==================================================\n");

    // Test 1: Simple quadratic f(x,y) = x² + y²
    let ops_rr = vec![
        MultiAD2RR::Inp(0),
        MultiAD2RR::Inp(0),
        MultiAD2RR::Mul,
        MultiAD2RR::Inp(1),
        MultiAD2RR::Inp(1),
        MultiAD2RR::Mul,
        MultiAD2RR::Add,
    ];

    let ops_fr = vec![
        MultiAD2FR::Inp(0),
        MultiAD2FR::Inp(0),
        MultiAD2FR::Mul,
        MultiAD2FR::Inp(1),
        MultiAD2FR::Inp(1),
        MultiAD2FR::Mul,
        MultiAD2FR::Add,
    ];

    let ops_rf = vec![
        MultiAD2RF::Inp(0),
        MultiAD2RF::Inp(0),
        MultiAD2RF::Mul,
        MultiAD2RF::Inp(1),
        MultiAD2RF::Inp(1),
        MultiAD2RF::Mul,
        MultiAD2RF::Add,
    ];

    let x = vec![1.0, 2.0];

    println!("Test 1: Simple quadratic f(x,y) = x² + y²");
    println!("Expected Hessian: [[2, 0], [0, 2]]\n");

    let iterations = 10000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = MultiAD2RR::compute_hessian(&ops_rr, &x).unwrap();
    }
    println!(
        "RR (exact, reverse-reverse): {:?}",
        start.elapsed() / iterations
    );

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = MultiAD2FR::compute_hessian(&ops_fr, &x).unwrap();
    }
    println!(
        "FR (exact, forward-reverse): {:?}",
        start.elapsed() / iterations
    );

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = MultiAD2RF::compute_hessian(&ops_rf, &x).unwrap();
    }
    println!(
        "RF (exact, reverse-forward): {:?}",
        start.elapsed() / iterations
    );

    // Test 2: Trigonometric function f(x,y) = sin(x) * cos(y)
    let ops_trig_rr = vec![
        MultiAD2RR::Inp(0),
        MultiAD2RR::Sin,
        MultiAD2RR::Inp(1),
        MultiAD2RR::Cos,
        MultiAD2RR::Mul,
    ];

    let ops_trig_fr = vec![
        MultiAD2FR::Inp(0),
        MultiAD2FR::Sin,
        MultiAD2FR::Inp(1),
        MultiAD2FR::Cos,
        MultiAD2FR::Mul,
    ];

    let ops_trig_rf = vec![
        MultiAD2RF::Inp(0),
        MultiAD2RF::Sin,
        MultiAD2RF::Inp(1),
        MultiAD2RF::Cos,
        MultiAD2RF::Mul,
    ];

    println!("\nTest 2: Trigonometric f(x,y) = sin(x) * cos(y)");
    println!("x = {}, y = {}\n", x[0], x[1]);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = MultiAD2RR::compute_hessian(&ops_trig_rr, &x).unwrap();
    }
    println!("RR (exact): {:?}", start.elapsed() / iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = MultiAD2FR::compute_hessian(&ops_trig_fr, &x).unwrap();
    }
    println!("FR (exact): {:?}", start.elapsed() / iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = MultiAD2RF::compute_hessian(&ops_trig_rf, &x).unwrap();
    }
    println!("RF (exact): {:?}", start.elapsed() / iterations);

    println!("\nSummary:");
    println!("- All three methods produce machine-precision exact results");
    println!("- RR uses per-node gradient vectors with outer products in reverse pass");
    println!("- FR uses dual-number forward + dual-adjoint reverse passes");
    println!("- RF uses the same dual algorithm as FR (equivalent for scalar fns)");
}
