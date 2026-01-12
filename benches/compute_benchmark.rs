// Criterion benchmarks for autodiff compute performance
use autodiff::{MonoAD, MultiAD, mono_ops};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn bench_single_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_operation");

    for op in [MonoAD::Sin, MonoAD::Cos, MonoAD::Exp] {
        let ops = vec![op];

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{:?}", op)),
            &ops,
            |b, ops| {
                b.iter(|| {
                    let (value, backprop) =
                        MonoAD::compute_grad(std::hint::black_box(ops), std::hint::black_box(2.0));
                    // Consume the results to prevent optimization
                    std::hint::black_box(value);
                    std::hint::black_box(backprop(1.0));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compute_arc", format!("{:?}", op)),
            &ops,
            |b, ops| {
                b.iter(|| {
                    let (value, backprop) = MonoAD::compute_grad(
                        std::hint::black_box(ops),
                        std::hint::black_box(2.0),
                    );
                    std::hint::black_box(value);
                    std::hint::black_box(backprop(1.0));
                })
            },
        );
    }

    group.finish();
}

fn bench_chained_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_operations");

    // Test different chain lengths
    for chain_length in [2, 3, 5, 10, 20].iter() {
        let exprs: Vec<MonoAD> = (0..*chain_length)
            .map(|i| match i % 3 {
                0 => MonoAD::Sin,
                1 => MonoAD::Cos,
                _ => MonoAD::Exp,
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("compute", chain_length),
            &exprs,
            |b, exprs| {
                b.iter(|| {
                    let (value, backprop) = MonoAD::compute_grad(
                        std::hint::black_box(exprs),
                        std::hint::black_box(2.0),
                    );
                    std::hint::black_box(value);
                    std::hint::black_box(backprop(1.0));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compute_arc", chain_length),
            &exprs,
            |b, exprs| {
                b.iter(|| {
                    let (value, backprop) = MonoAD::compute_grad(
                        std::hint::black_box(exprs),
                        std::hint::black_box(2.0),
                    );
                    std::hint::black_box(value);
                    std::hint::black_box(backprop(1.0));
                })
            },
        );
    }

    group.finish();
}

fn bench_macro_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("macro_usage");

    // Benchmark the example from main.rs
    let exprs = mono_ops![sin, sin, exp];

    group.bench_function("compute_with_macro", |b| {
        b.iter(|| {
            let (value, backprop) =
                MonoAD::compute_grad(std::hint::black_box(&exprs), std::hint::black_box(2.0));
            std::hint::black_box(value);
            std::hint::black_box(backprop(1.0));
        })
    });

    group.bench_function("compute_arc_with_macro", |b| {
        b.iter(|| {
            let (value, backprop) =
                MonoAD::compute_grad(std::hint::black_box(&exprs), std::hint::black_box(2.0));
            std::hint::black_box(value);
            std::hint::black_box(backprop(1.0));
        })
    });

    group.finish();
}

fn bench_backprop_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("backprop_only");

    let exprs = mono_ops![sin, sin, exp];

    // Benchmark just the backward pass
    group.bench_function("compute_backprop", |b| {
        let (_value, backprop) = MonoAD::compute_grad(&exprs, 2.0);
        b.iter(|| {
            std::hint::black_box(backprop(std::hint::black_box(1.0)));
        })
    });

    group.bench_function("compute_arc_backprop", |b| {
        let (_value, backprop) = MonoAD::compute_grad(&exprs, 2.0);
        b.iter(|| {
            std::hint::black_box(backprop(std::hint::black_box(1.0)));
        })
    });

    group.finish();
}

// ===== MultiAD Benchmarks =====

fn bench_multi_forward_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_forward");

    // Build computation graph: f(x₁, x₂) = sin(x₁) * (x₁ + x₂)
    let exprs = &[
        (MultiAD::Inp, vec![0]),    // x₁ at index 0
        (MultiAD::Inp, vec![1]),    // x₂ at index 1
        (MultiAD::Add, vec![0, 1]), // x₁ + x₂ at index 2
        (MultiAD::Sin, vec![0]),    // sin(x₁) at index 3
        (MultiAD::Mul, vec![2, 3]), // sin(x₁) * (x₁ + x₂) at index 4
    ];

    group.bench_function("forward_only", |b| {
        b.iter(|| {
            let result = MultiAD::compute(
                std::hint::black_box(exprs),
                std::hint::black_box(&[0.6, 1.4]),
            ).unwrap();
            std::hint::black_box(result);
        })
    });

    group.finish();
}

fn bench_multi_forward_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_forward_backward");

    let exprs = &[
        (MultiAD::Inp, vec![0]),
        (MultiAD::Inp, vec![1]),
        (MultiAD::Add, vec![0, 1]),
        (MultiAD::Sin, vec![0]),
        (MultiAD::Mul, vec![2, 3]),
    ];

    group.bench_function("compute_grad", |b| {
        b.iter(|| {
            let (value, backprop_fn) = MultiAD::compute_grad(
                std::hint::black_box(exprs),
                std::hint::black_box(&[0.6, 1.4]),
            ).unwrap();
            std::hint::black_box(value);
            let grads = backprop_fn(1.0);
            std::hint::black_box(grads);
        })
    });

    group.bench_function("compute_grad_box", |b| {
        b.iter(|| {
            let (value, backprop_fn) = MultiAD::compute_grad(
                std::hint::black_box(exprs),
                std::hint::black_box(&[0.6, 1.4]),
            ).unwrap();
            std::hint::black_box(value);
            let grads = backprop_fn(1.0);
            std::hint::black_box(grads);
        })
    });

    group.finish();
}

fn bench_multi_backward_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_backward_only");

    let exprs = &[
        (MultiAD::Inp, vec![0]),
        (MultiAD::Inp, vec![1]),
        (MultiAD::Add, vec![0, 1]),
        (MultiAD::Sin, vec![0]),
        (MultiAD::Mul, vec![2, 3]),
    ];

    // Benchmark just the backward pass for Box version
    group.bench_function("compute_grad_backprop", |b| {
        let (_value, backprop_fn) = MultiAD::compute_grad(exprs, &[0.6, 1.4]).unwrap();
        b.iter(|| {
            let grads = backprop_fn(std::hint::black_box(1.0));
            std::hint::black_box(grads);
        })
    });

    // Benchmark just the backward pass for Arc version
    group.bench_function("compute_grad_backprop_arc", |b| {
        let (_value, backprop_fn) = MultiAD::compute_grad(exprs, &[0.6, 1.4]).unwrap();
        b.iter(|| {
            let grads = backprop_fn(std::hint::black_box(1.0));
            std::hint::black_box(grads);
        })
    });

    group.finish();
}

fn bench_multi_graph_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_graph_complexity");

    // Benchmark graphs with different numbers of operations
    for num_ops in [3, 5, 10, 15].iter() {
        // Build a computation graph with increasing complexity
        let mut exprs: Vec<(MultiAD, Vec<usize>)> =
            vec![(MultiAD::Inp, vec![0]), (MultiAD::Inp, vec![1])];

        // Add operations dynamically based on num_ops
        for i in 0..(*num_ops - 2) {
            let op = match i % 4 {
                0 => MultiAD::Sin,
                1 => MultiAD::Cos,
                2 => MultiAD::Exp,
                _ => MultiAD::Add,
            };

            // Determine which indices to use as arguments
            let arg_indices = if matches!(
                op,
                MultiAD::Sin | MultiAD::Cos | MultiAD::Exp | MultiAD::Inp
            ) {
                vec![i % 2] // Use single input
            } else {
                vec![i % 2, (i + 1) % 2] // Use two inputs
            };

            exprs.push((op, arg_indices));
        }

        group.bench_with_input(
            BenchmarkId::new("compute_grad", num_ops),
            &exprs,
            |b, exprs| {
                b.iter(|| {
                    let (value, backprop_fn) = MultiAD::compute_grad(
                        std::hint::black_box(exprs),
                        std::hint::black_box(&[0.5, 1.5]),
                    ).unwrap();
                    std::hint::black_box(value);
                    let grads = backprop_fn(1.0);
                    std::hint::black_box(grads);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compute_grad_box", num_ops),
            &exprs,
            |b, exprs| {
                b.iter(|| {
                    let (value, backprop_fn) = MultiAD::compute_grad(
                        std::hint::black_box(exprs),
                        std::hint::black_box(&[0.5, 1.5]),
                    ).unwrap();
                    std::hint::black_box(value);
                    let grads = backprop_fn(1.0);
                    std::hint::black_box(grads);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_backprop_execution,
    bench_single_operations,
    bench_chained_operations,
    bench_macro_usage,
    bench_multi_forward_only,
    bench_multi_forward_backward,
    bench_multi_backward_only,
    bench_multi_graph_complexity,
);
criterion_main!(benches);
