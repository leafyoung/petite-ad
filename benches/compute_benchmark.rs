// Criterion benchmarks for autodiff compute performance
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use petite_ad::{
    mono_ops, types::MonoGradientFn, types::MultiGradientFn, BackendKind, BatchGradientsBuffer,
    BatchInputs, BatchValuesBuffer, ExecutionBackend, ForwardAD, Graph, MonoAD, MultiAD,
    SimdBackend,
};

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
                    let (value, backprop) = MonoAD::compute_grad_generic::<Arc<MonoGradientFn>>(
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
                    let (value, backprop) = MonoAD::compute_grad_generic::<Arc<MonoGradientFn>>(
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
            let (value, backprop) = MonoAD::compute_grad_generic::<Arc<MonoGradientFn>>(
                std::hint::black_box(&exprs),
                std::hint::black_box(2.0),
            );
            std::hint::black_box(value);
            std::hint::black_box(backprop(1.0));
        })
    });

    group.finish();
}

fn bench_mono_checked(c: &mut Criterion) {
    let mut group = c.benchmark_group("mono_checked");
    let exprs = mono_ops![sqrt, ln, exp];
    let x = 4.0;

    group.bench_function("compute_checked", |b| {
        b.iter(|| {
            let value =
                MonoAD::compute_checked(std::hint::black_box(&exprs), std::hint::black_box(x))
                    .unwrap();
            std::hint::black_box(value);
        })
    });

    group.bench_function("compute_grad_checked", |b| {
        b.iter(|| {
            let (value, backprop) =
                MonoAD::compute_grad_checked(std::hint::black_box(&exprs), std::hint::black_box(x))
                    .unwrap();
            std::hint::black_box(value);
            std::hint::black_box(backprop(1.0));
        })
    });

    group.bench_function("compute_hessian_checked", |b| {
        b.iter(|| {
            let hessian = MonoAD::compute_hessian_checked(
                std::hint::black_box(&exprs),
                std::hint::black_box(x),
            )
            .unwrap();
            std::hint::black_box(hessian);
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
        let (_value, backprop) = MonoAD::compute_grad_generic::<Arc<MonoGradientFn>>(&exprs, 2.0);
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
            )
            .unwrap();
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
            )
            .unwrap();
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
            )
            .unwrap();
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
        let (_value, backprop_fn) =
            MultiAD::compute_grad_generic::<Arc<MultiGradientFn>>(exprs, &[0.6, 1.4]).unwrap();
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
                    )
                    .unwrap();
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
                    )
                    .unwrap();
                    std::hint::black_box(value);
                    let grads = backprop_fn(1.0);
                    std::hint::black_box(grads);
                })
            },
        );
    }

    group.finish();
}

fn make_reusable_graph() -> Graph {
    let mut graph = Graph::new(2);
    let x = graph.input(0);
    let y = graph.input(1);
    let sum = graph.add(x, y);
    let sin_x = graph.sin(x);
    graph.mul(sum, sin_x);
    graph
}

fn make_simd_basic_graph() -> Graph {
    let mut graph = Graph::new(2);
    let x = graph.input(0);
    let y = graph.input(1);
    let product = graph.mul(x, y);
    let ratio = graph.div(x, y);
    let sum = graph.add(product, ratio);
    let shifted = graph.add_const(sum, 0.25);
    let root = graph.sqrt(shifted);
    let exponent = graph.log1p_exp(y);
    let powered = graph.pow(root, exponent);
    let mixed = graph.log_add_exp(powered, y);
    graph.tanh(mixed);
    graph
}

fn make_multi_output_graph() -> Graph {
    let mut graph = Graph::new(2);
    let x = graph.input(0);
    let y = graph.input(1);
    let sum = graph.add(x, y);
    let product = graph.mul(x, y);
    let sin_x = graph.sin(x);
    let out = graph.mul(sum, sin_x);
    graph.set_outputs(&[sum, product, out]).unwrap();
    graph
}

fn make_checked_graph() -> Graph {
    let mut graph = Graph::new(2);
    let x = graph.input(0);
    let y = graph.input(1);
    let sqrt_x = graph.sqrt(x);
    let ln_y = graph.ln(y);
    let ratio = graph.div(sqrt_x, ln_y);
    graph.set_outputs(&[ratio, sqrt_x, ln_y]).unwrap();
    graph
}

fn bench_graph_tape_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_tape_compute");
    let inputs = [0.6, 1.4];
    let legacy_exprs = &[
        (MultiAD::Inp, vec![0]),
        (MultiAD::Inp, vec![1]),
        (MultiAD::Add, vec![0, 1]),
        (MultiAD::Sin, vec![0]),
        (MultiAD::Mul, vec![2, 3]),
    ];
    let seed = [1.0, 0.0];
    let graph = make_reusable_graph();
    let tape = graph.compile();
    let compiled = graph.compile_ir().unwrap();
    let mut workspace = tape.workspace();
    let mut compiled_workspace = compiled.workspace();

    group.bench_function("legacy_tuple_compute", |b| {
        b.iter(|| {
            let value = MultiAD::compute(
                std::hint::black_box(legacy_exprs),
                std::hint::black_box(&inputs),
            )
            .unwrap();
            std::hint::black_box(value);
        })
    });

    group.bench_function("forward_directional_derivative", |b| {
        b.iter(|| {
            let value = ForwardAD::directional_derivative(
                std::hint::black_box(legacy_exprs),
                std::hint::black_box(&inputs),
                std::hint::black_box(&seed),
            )
            .unwrap();
            std::hint::black_box(value);
        })
    });

    group.bench_function("graph_compute", |b| {
        b.iter(|| {
            let value = graph.compute(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(value);
        })
    });

    group.bench_function("tape_compute", |b| {
        b.iter(|| {
            let value = tape.compute(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(value);
        })
    });

    group.bench_function("tape_compute_workspace", |b| {
        b.iter(|| {
            let value = tape
                .compute_with_workspace(std::hint::black_box(&inputs), &mut workspace)
                .unwrap();
            std::hint::black_box(value);
        })
    });

    group.bench_function("compiled_compute_workspace", |b| {
        b.iter(|| {
            let value = compiled
                .compute_with_workspace(std::hint::black_box(&inputs), &mut compiled_workspace)
                .unwrap();
            std::hint::black_box(value);
        })
    });

    let batch_data = [0.6, 1.4, 0.7, 1.3, 0.8, 1.2, 0.9, 1.1];
    let batch = BatchInputs::new(&batch_data, 4, 2).unwrap();
    group.bench_function("compiled_compute_batch", |b| {
        b.iter(|| {
            let value = compiled.compute_batch(std::hint::black_box(batch)).unwrap();
            std::hint::black_box(value);
        })
    });

    let mut batch_values_buffer = BatchValuesBuffer::new();
    group.bench_function("compiled_compute_batch_into", |b| {
        b.iter(|| {
            compiled
                .compute_batch_into(std::hint::black_box(batch), &mut batch_values_buffer)
                .unwrap();
            std::hint::black_box(&batch_values_buffer);
        })
    });

    group.finish();
}

fn make_batch_data(batch_size: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(batch_size * 2);
    for row in 0..batch_size {
        let row_f64 = row as f64;
        data.push(2.0 + row_f64 * 0.01);
        data.push(1.0 + row_f64 * 0.005);
    }
    data
}

fn bench_simd_batch_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_compute");
    let graph = make_simd_basic_graph();
    let compiled = graph.compile_ir().unwrap();
    let simd = SimdBackend;

    for batch_size in [3_usize, 4, 8, 31, 64] {
        let batch_data = make_batch_data(batch_size);
        let batch = BatchInputs::new(&batch_data, batch_size, 2).unwrap();

        let mut scalar_buffer = BatchValuesBuffer::new();
        group.bench_with_input(
            BenchmarkId::new("scalar_compute_into", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    compiled
                        .compute_batch_into(std::hint::black_box(*batch), &mut scalar_buffer)
                        .unwrap();
                    std::hint::black_box(&scalar_buffer);
                })
            },
        );

        let mut scalar_gradient_buffer = BatchGradientsBuffer::new();
        group.bench_with_input(
            BenchmarkId::new("scalar_gradient_into", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    compiled
                        .gradient_batch_into(
                            std::hint::black_box(*batch),
                            &mut scalar_gradient_buffer,
                        )
                        .unwrap();
                    std::hint::black_box(&scalar_gradient_buffer);
                })
            },
        );

        let mut auto_buffer = BatchValuesBuffer::new();
        group.bench_with_input(
            BenchmarkId::new("auto_compute_into", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    compiled
                        .compute_batch_auto_into(std::hint::black_box(*batch), &mut auto_buffer)
                        .unwrap();
                    std::hint::black_box(&auto_buffer);
                })
            },
        );

        let mut auto_gradient_buffer = BatchGradientsBuffer::new();
        group.bench_with_input(
            BenchmarkId::new("auto_gradient_into", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    compiled
                        .gradient_batch_auto_into(
                            std::hint::black_box(*batch),
                            &mut auto_gradient_buffer,
                        )
                        .unwrap();
                    std::hint::black_box(&auto_gradient_buffer);
                })
            },
        );

        for backend in [BackendKind::SimdF64x2, BackendKind::SimdF64x4] {
            let report = compiled.backend_support_report(backend).unwrap();
            if report.can_compute_batch() {
                let mut values_buffer = BatchValuesBuffer::new();
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_compute_into", backend.name()), batch_size),
                    &batch,
                    |b, batch| {
                        b.iter(|| {
                            backend
                                .compute_batch(
                                    std::hint::black_box(&compiled),
                                    std::hint::black_box(*batch),
                                    &mut values_buffer,
                                )
                                .unwrap();
                            std::hint::black_box(&values_buffer);
                        })
                    },
                );
            }
            if report.can_gradient_batch() {
                let mut gradients_buffer = BatchGradientsBuffer::new();
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_gradient_into", backend.name()), batch_size),
                    &batch,
                    |b, batch| {
                        b.iter(|| {
                            backend
                                .gradient_batch(
                                    std::hint::black_box(&compiled),
                                    std::hint::black_box(*batch),
                                    &mut gradients_buffer,
                                )
                                .unwrap();
                            std::hint::black_box(&gradients_buffer);
                        })
                    },
                );
            }
        }
    }

    if simd.capabilities().supports_batch_compute {
        let batch_data = make_batch_data(64);
        let batch = BatchInputs::new(&batch_data, 64, 2).unwrap();
        let mut simd_buffer = BatchValuesBuffer::new();
        group.bench_function("simd_trait_compute_into/64", |b| {
            b.iter(|| {
                simd.compute_batch(std::hint::black_box(&compiled), batch, &mut simd_buffer)
                    .unwrap();
                std::hint::black_box(&simd_buffer);
            })
        });
    }

    group.finish();
}

fn bench_graph_tape_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_tape_gradient");
    let inputs = [0.6, 1.4];
    let graph = make_reusable_graph();
    let tape = graph.compile();
    let compiled = graph.compile_ir().unwrap();
    let mut workspace = tape.workspace();
    let mut compiled_workspace = compiled.workspace();

    group.bench_function("graph_compute_grad", |b| {
        b.iter(|| {
            let (value, backprop_fn) = graph.compute_grad(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(value);
            std::hint::black_box(backprop_fn(1.0));
        })
    });

    group.bench_function("tape_compute_grad", |b| {
        b.iter(|| {
            let (value, backprop_fn) = tape.compute_grad(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(value);
            std::hint::black_box(backprop_fn(1.0));
        })
    });

    group.bench_function("tape_gradient_workspace", |b| {
        b.iter(|| {
            let (value, grad) = tape
                .gradient_with_workspace(std::hint::black_box(&inputs), &mut workspace)
                .unwrap();
            std::hint::black_box(value);
            std::hint::black_box(grad);
        })
    });

    group.bench_function("compiled_gradient_workspace", |b| {
        b.iter(|| {
            let (value, grad) = compiled
                .gradient_with_workspace(std::hint::black_box(&inputs), &mut compiled_workspace)
                .unwrap();
            std::hint::black_box(value);
            std::hint::black_box(grad);
        })
    });

    let batch_data = [0.6, 1.4, 0.7, 1.3, 0.8, 1.2, 0.9, 1.1];
    let batch = BatchInputs::new(&batch_data, 4, 2).unwrap();
    group.bench_function("compiled_gradient_batch", |b| {
        b.iter(|| {
            let value = compiled
                .gradient_batch(std::hint::black_box(batch))
                .unwrap();
            std::hint::black_box(value);
        })
    });

    let mut batch_gradients_buffer = BatchGradientsBuffer::new();
    group.bench_function("compiled_gradient_batch_into", |b| {
        b.iter(|| {
            compiled
                .gradient_batch_into(std::hint::black_box(batch), &mut batch_gradients_buffer)
                .unwrap();
            std::hint::black_box(&batch_gradients_buffer);
        })
    });

    group.finish();
}

fn bench_graph_tape_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_tape_jacobian");
    let inputs = [0.6, 1.4];
    let graph = make_multi_output_graph();
    let tape = graph.compile();
    let mut workspace = tape.workspace();

    group.bench_function("graph_jacobian", |b| {
        b.iter(|| {
            let jacobian = graph.jacobian(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(jacobian);
        })
    });

    group.bench_function("tape_jacobian", |b| {
        b.iter(|| {
            let jacobian = tape.jacobian(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(jacobian);
        })
    });

    group.bench_function("tape_jacobian_workspace", |b| {
        b.iter(|| {
            let jacobian = tape
                .jacobian_with_workspace(std::hint::black_box(&inputs), &mut workspace)
                .unwrap();
            std::hint::black_box(jacobian);
        })
    });

    group.finish();
}

fn bench_graph_tape_checked(c: &mut Criterion) {
    let mut compute_group = c.benchmark_group("graph_tape_checked_compute");
    let inputs = [4.0, 2.5];
    let graph = make_checked_graph();
    let tape = graph.compile();
    let mut compute_workspace = tape.workspace();

    compute_group.bench_function("graph_compute_checked", |b| {
        b.iter(|| {
            let value = graph
                .compute_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(value);
        })
    });

    compute_group.bench_function("tape_compute_checked", |b| {
        b.iter(|| {
            let value = tape.compute_checked(std::hint::black_box(&inputs)).unwrap();
            std::hint::black_box(value);
        })
    });

    compute_group.bench_function("tape_compute_workspace_checked", |b| {
        b.iter(|| {
            let value = tape
                .compute_with_workspace_checked(
                    std::hint::black_box(&inputs),
                    &mut compute_workspace,
                )
                .unwrap();
            std::hint::black_box(value);
        })
    });

    compute_group.bench_function("graph_compute_many_checked", |b| {
        b.iter(|| {
            let values = graph
                .compute_many_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(values);
        })
    });

    compute_group.bench_function("tape_compute_many_checked", |b| {
        b.iter(|| {
            let values = tape
                .compute_many_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(values);
        })
    });

    compute_group.bench_function("tape_compute_many_workspace_checked", |b| {
        b.iter(|| {
            let values = tape
                .compute_many_with_workspace_checked(
                    std::hint::black_box(&inputs),
                    &mut compute_workspace,
                )
                .unwrap();
            std::hint::black_box(values);
        })
    });

    compute_group.finish();

    let mut gradient_group = c.benchmark_group("graph_tape_checked_gradient");
    let mut gradient_workspace = tape.workspace();

    gradient_group.bench_function("graph_gradient_checked", |b| {
        b.iter(|| {
            let (value, grad) = graph
                .gradient_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(value);
            std::hint::black_box(grad);
        })
    });

    gradient_group.bench_function("tape_gradient_checked", |b| {
        b.iter(|| {
            let (value, grad) = tape
                .gradient_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(value);
            std::hint::black_box(grad);
        })
    });

    gradient_group.bench_function("tape_gradient_workspace_checked", |b| {
        b.iter(|| {
            let (value, grad) = tape
                .gradient_with_workspace_checked(
                    std::hint::black_box(&inputs),
                    &mut gradient_workspace,
                )
                .unwrap();
            std::hint::black_box(value);
            std::hint::black_box(grad);
        })
    });

    gradient_group.finish();

    let mut jacobian_group = c.benchmark_group("graph_tape_checked_jacobian");
    let mut jacobian_workspace = tape.workspace();

    jacobian_group.bench_function("graph_jacobian_checked", |b| {
        b.iter(|| {
            let jacobian = graph
                .jacobian_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(jacobian);
        })
    });

    jacobian_group.bench_function("tape_jacobian_checked", |b| {
        b.iter(|| {
            let jacobian = tape
                .jacobian_checked(std::hint::black_box(&inputs))
                .unwrap();
            std::hint::black_box(jacobian);
        })
    });

    jacobian_group.bench_function("tape_jacobian_workspace_checked", |b| {
        b.iter(|| {
            let jacobian = tape
                .jacobian_with_workspace_checked(
                    std::hint::black_box(&inputs),
                    &mut jacobian_workspace,
                )
                .unwrap();
            std::hint::black_box(jacobian);
        })
    });

    jacobian_group.finish();
}

criterion_group!(
    benches,
    bench_backprop_execution,
    bench_single_operations,
    bench_chained_operations,
    bench_macro_usage,
    bench_mono_checked,
    bench_multi_forward_only,
    bench_multi_forward_backward,
    bench_multi_backward_only,
    bench_multi_graph_complexity,
    bench_graph_tape_compute,
    bench_simd_batch_compute,
    bench_graph_tape_gradient,
    bench_graph_tape_jacobian,
    bench_graph_tape_checked,
);
criterion_main!(benches);
