// Criterion benchmarks for Hessian computation performance
// Compares RR, FR, and RF methods on different problem sizes

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use petite_ad::{Graph, MultiAD2FR, MultiAD2RF, MultiAD2RR};

fn bench_hessian_rr_vs_fr_vs_rf(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian_methods");

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

    group.bench_function("rr_quadratic_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2RR::compute_hessian(
                std::hint::black_box(&ops_rr),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.bench_function("fr_quadratic_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2FR::compute_hessian(
                std::hint::black_box(&ops_fr),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.bench_function("rf_quadratic_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2RF::compute_hessian(
                std::hint::black_box(&ops_rf),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    let mut graph = Graph::new(2);
    let x0 = graph.input(0);
    let x1 = graph.input(1);
    let x0_sq = graph.square(x0);
    let x1_sq = graph.square(x1);
    graph.add(x0_sq, x1_sq);

    group.bench_function("graph_exact_rr_quadratic_2vars", |b| {
        b.iter(|| {
            let hessian = graph.exact_hessian_rr(std::hint::black_box(&x)).unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.finish();
}

fn bench_hessian_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian_scalability");

    // Test different numbers of variables
    // For n variables: f(x) = sum(x_i²)
    for n in [2, 3, 4, 5].iter() {
        // Build operations for f(x) = x_0² + x_1² + ... + x_{n-1}²
        let mut ops_rr = vec![];
        let mut ops_fr = vec![];
        let mut ops_rf = vec![];

        for i in 0..*n {
            ops_rr.push(MultiAD2RR::Inp(i));
            ops_rr.push(MultiAD2RR::Inp(i));
            ops_rr.push(MultiAD2RR::Mul);

            ops_fr.push(MultiAD2FR::Inp(i));
            ops_fr.push(MultiAD2FR::Inp(i));
            ops_fr.push(MultiAD2FR::Mul);

            ops_rf.push(MultiAD2RF::Inp(i));
            ops_rf.push(MultiAD2RF::Inp(i));
            ops_rf.push(MultiAD2RF::Mul);

            if i > 0 {
                ops_rr.push(MultiAD2RR::Add);
                ops_fr.push(MultiAD2FR::Add);
                ops_rf.push(MultiAD2RF::Add);
            }
        }

        let x: Vec<f64> = (0..*n).map(|i| (i + 1) as f64).collect();

        group.bench_with_input(BenchmarkId::new("rr", n), &n, |b, _| {
            b.iter(|| {
                let hessian = MultiAD2RR::compute_hessian(
                    std::hint::black_box(&ops_rr),
                    std::hint::black_box(&x),
                )
                .unwrap();
                std::hint::black_box(hessian);
            })
        });

        group.bench_with_input(BenchmarkId::new("fr", n), &n, |b, _| {
            b.iter(|| {
                let hessian = MultiAD2FR::compute_hessian(
                    std::hint::black_box(&ops_fr),
                    std::hint::black_box(&x),
                )
                .unwrap();
                std::hint::black_box(hessian);
            })
        });

        group.bench_with_input(BenchmarkId::new("rf", n), &n, |b, _| {
            b.iter(|| {
                let hessian = MultiAD2RF::compute_hessian(
                    std::hint::black_box(&ops_rf),
                    std::hint::black_box(&x),
                )
                .unwrap();
                std::hint::black_box(hessian);
            })
        });
    }

    group.finish();
}

fn bench_hessian_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian_complex");

    // Test 1: Trigonometric function f(x,y) = sin(x) * cos(y)
    let ops_rr = vec![
        MultiAD2RR::Inp(0),
        MultiAD2RR::Sin,
        MultiAD2RR::Inp(1),
        MultiAD2RR::Cos,
        MultiAD2RR::Mul,
    ];

    let ops_fr = vec![
        MultiAD2FR::Inp(0),
        MultiAD2FR::Sin,
        MultiAD2FR::Inp(1),
        MultiAD2FR::Cos,
        MultiAD2FR::Mul,
    ];

    let ops_rf = vec![
        MultiAD2RF::Inp(0),
        MultiAD2RF::Sin,
        MultiAD2RF::Inp(1),
        MultiAD2RF::Cos,
        MultiAD2RF::Mul,
    ];

    let x = vec![1.0, 2.0];

    group.bench_function("rr_sin_cos_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2RR::compute_hessian(
                std::hint::black_box(&ops_rr),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.bench_function("fr_sin_cos_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2FR::compute_hessian(
                std::hint::black_box(&ops_fr),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.bench_function("rf_sin_cos_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2RF::compute_hessian(
                std::hint::black_box(&ops_rf),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    // Test 2: Exponential function f(x,y) = exp(x) + exp(y)
    let ops_rr_exp = vec![
        MultiAD2RR::Inp(0),
        MultiAD2RR::Exp,
        MultiAD2RR::Inp(1),
        MultiAD2RR::Exp,
        MultiAD2RR::Add,
    ];

    let ops_fr_exp = vec![
        MultiAD2FR::Inp(0),
        MultiAD2FR::Exp,
        MultiAD2FR::Inp(1),
        MultiAD2FR::Exp,
        MultiAD2FR::Add,
    ];

    let ops_rf_exp = vec![
        MultiAD2RF::Inp(0),
        MultiAD2RF::Exp,
        MultiAD2RF::Inp(1),
        MultiAD2RF::Exp,
        MultiAD2RF::Add,
    ];

    group.bench_function("rr_exp_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2RR::compute_hessian(
                std::hint::black_box(&ops_rr_exp),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.bench_function("fr_exp_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2FR::compute_hessian(
                std::hint::black_box(&ops_fr_exp),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.bench_function("rf_exp_2vars", |b| {
        b.iter(|| {
            let hessian = MultiAD2RF::compute_hessian(
                std::hint::black_box(&ops_rf_exp),
                std::hint::black_box(&x),
            )
            .unwrap();
            std::hint::black_box(hessian);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hessian_rr_vs_fr_vs_rf,
    bench_hessian_scalability,
    bench_hessian_complex_operations
);
criterion_main!(benches);
