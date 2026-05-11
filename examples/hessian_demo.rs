use petite_ad::{
    mono_ops, mono_ops_fr, mono_ops_rf, mono_ops_rr, MonoAD, MonoAD2FR, MonoAD2RF, MonoAD2RR,
};

/// Demonstrates exact vs. approximate Hessian computation.
///
/// This example shows:
/// 1. How the existing MonoAD.compute_hessian (finite differences) works
/// 2. The new exact Hessian methods (RR, FR, RF)
/// 3. Accuracy comparison between finite differences and exact methods
fn main() {
    println!("=== Exact vs Approximate Hessian Demonstration ===\n");

    // Example 1: f(x) = sin(x)
    // f'(x) = cos(x)
    // f''(x) = -sin(x)
    println!("Example 1: f(x) = sin(x)");
    let ops1 = mono_ops![sin];
    let ops1_rr = mono_ops_rr![sin];
    let ops1_fr = mono_ops_fr![sin];
    let ops1_rf = mono_ops_rf![sin];
    let x1: f64 = 0.5;
    let expected1 = -x1.sin();

    let (value1, _grad_fn1) = MonoAD::compute_grad(&ops1, x1);
    let hess_fd1 = MonoAD::compute_hessian(&ops1, x1);
    let hess_rr1 = MonoAD2RR::compute_hessian(&ops1_rr, x1);
    let hess_fr1 = MonoAD2FR::compute_hessian(&ops1_fr, x1);
    let hess_rf1 = MonoAD2RF::compute_hessian(&ops1_rf, x1);

    println!("  f({}) = {}", x1, value1);
    println!("  Analytical f''({}) = {}", x1, expected1);
    println!(
        "  Finite Difference: {:.10} (error: {:.2e})",
        hess_fd1,
        (hess_fd1 - expected1).abs()
    );
    println!(
        "  RR (Exact):        {:.10} (error: {:.2e})",
        hess_rr1,
        (hess_rr1 - expected1).abs()
    );
    println!(
        "  FR (Exact):        {:.10} (error: {:.2e})",
        hess_fr1,
        (hess_fr1 - expected1).abs()
    );
    println!(
        "  RF (Exact):        {:.10} (error: {:.2e})",
        hess_rf1,
        (hess_rf1 - expected1).abs()
    );

    // Example 2: f(x) = exp(x)
    // f'(x) = exp(x)
    // f''(x) = exp(x)
    println!("\nExample 2: f(x) = exp(x)");
    let ops2 = mono_ops![exp];
    let ops2_rr = mono_ops_rr![exp];
    let ops2_fr = mono_ops_fr![exp];
    let ops2_rf = mono_ops_rf![exp];
    let x2: f64 = 1.5;
    let expected2 = x2.exp();

    let (value2, _grad_fn2) = MonoAD::compute_grad(&ops2, x2);
    let hess_fd2 = MonoAD::compute_hessian(&ops2, x2);
    let hess_rr2 = MonoAD2RR::compute_hessian(&ops2_rr, x2);
    let hess_fr2 = MonoAD2FR::compute_hessian(&ops2_fr, x2);
    let hess_rf2 = MonoAD2RF::compute_hessian(&ops2_rf, x2);

    println!("  f({}) = {}", x2, value2);
    println!("  Analytical f''({}) = {}", x2, expected2);
    println!(
        "  Finite Difference: {:.10} (error: {:.2e})",
        hess_fd2,
        (hess_fd2 - expected2).abs()
    );
    println!(
        "  RR (Exact):        {:.10} (error: {:.2e})",
        hess_rr2,
        (hess_rr2 - expected2).abs()
    );
    println!(
        "  FR (Exact):        {:.10} (error: {:.2e})",
        hess_fr2,
        (hess_fr2 - expected2).abs()
    );
    println!(
        "  RF (Exact):        {:.10} (error: {:.2e})",
        hess_rf2,
        (hess_rf2 - expected2).abs()
    );

    // Example 3: f(x) = exp(sin(sin(x))) - composed function
    // Note: operations are applied right-to-left, so [sin, sin, exp] means exp(sin(sin(x)))
    // This tests the chain rule handling
    println!("\nExample 3: f(x) = exp(sin(sin(x))) [composed]");
    let ops3 = mono_ops![sin, sin, exp];
    let ops3_rr = mono_ops_rr![sin, sin, exp];
    let ops3_fr = mono_ops_fr![sin, sin, exp];
    let ops3_rf = mono_ops_rf![sin, sin, exp];
    let x3: f64 = 2.0;

    let (value3, _grad_fn3) = MonoAD::compute_grad(&ops3, x3);
    let hess_fd3 = MonoAD::compute_hessian(&ops3, x3);
    let hess_rr3 = MonoAD2RR::compute_hessian(&ops3_rr, x3);
    let hess_fr3 = MonoAD2FR::compute_hessian(&ops3_fr, x3);
    let hess_rf3 = MonoAD2RF::compute_hessian(&ops3_rf, x3);

    // Compute exact value manually for verification: f(x) = exp(sin(sin(x)))
    let t1 = x3.sin();
    let t2 = t1.sin();
    let dt1 = x3.cos();
    let dt2 = t1.cos() * dt1;
    let ddt1 = -x3.sin();
    let ddt2 = -t1.sin() * dt1 * dt1 + t1.cos() * ddt1;
    let expected3 = t2.exp() * dt2 * dt2 + t2.exp() * ddt2;

    println!("  f({}) = {}", x3, value3);
    println!("  Analytical f''({}) = {:.10}", x3, expected3);
    println!(
        "  Finite Difference: {:.10} (error: {:.2e})",
        hess_fd3,
        (hess_fd3 - expected3).abs()
    );
    println!(
        "  RR (Exact):        {:.10} (error: {:.2e})",
        hess_rr3,
        (hess_rr3 - expected3).abs()
    );
    println!(
        "  FR (Exact):        {:.10} (error: {:.2e})",
        hess_fr3,
        (hess_fr3 - expected3).abs()
    );
    println!(
        "  RF (Exact):        {:.10} (error: {:.2e})",
        hess_rf3,
        (hess_rf3 - expected3).abs()
    );

    println!("\n=== Summary ===\n");

    println!("1. Finite Difference Method (MonoAD::compute_hessian):");
    println!("   - Uses ε = 1e-5");
    println!("   - Accuracy: ~1e-4 to 1e-6");
    println!("   - Pros: Simple, works for all operations");
    println!("   - Cons: Numerical approximation, not machine precision");
    println!();

    println!("2. Exact Methods (MonoAD2RR/FR/RF::compute_hessian):");
    println!("   - RR: Reverse-over-Reverse (tracks 2nd derivatives during backward pass)");
    println!("   - FR: Forward-over-Reverse (forward-mode on gradient function)");
    println!("   - RF: Reverse-over-Forward (reverse-mode on forward-mode)");
    println!("   - Accuracy: Machine precision (~1e-15)");
    println!("   - Pros: Exact differentiation, no approximation error");
    println!("   - Cons: More complex implementation, slightly more computation");
    println!();

    println!("3. When to Use Which:");
    println!("   - Finite differences: Quick prototyping, most practical applications");
    println!("   - Exact methods: When you need machine precision for:");
    println!("     * Numerical optimization requiring exact Hessians");
    println!("     * Scientific computing with strict accuracy requirements");
    println!("     * Research applications in automatic differentiation");
    println!();

    println!("4. Performance Note:");
    println!("   - All three exact methods have similar performance");
    println!("   - For univariate functions, FR/RF delegate to RR for composed operations");
    println!("   - Choice between FR/RF/RR is more relevant for multivariate functions");
}
