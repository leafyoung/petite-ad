# AGENTS.md

Guidance for agentic coding agents working with petite-ad.

## Quick Start

**Build directory**: `~/.cache/rust-build/` (cached, never mention in docs)
**Edition**: Rust 2021 | **License**: MIT OR Apache-2.0

### Essential Commands

```bash
# Build & check
cargo build --release                    # Build optimized release
cargo check                              # Quick syntax check
cargo fmt                                # Auto-format code
cargo clippy --all-targets --all-features -- -D warnings  # Lint with strict warnings

# Testing
cargo test                               # Run all tests
cargo test test_name -- --nocapture     # Run specific test with output
cargo test --lib                         # Library tests only
cargo tarpaulin --lib --out Stdout       # Generate coverage report

# Benchmarking
cargo bench                              # Run benchmarks
cargo bench --no-run                     # Check benchmarks compile without running
```

## Project Structure

**Core Library** (`src/lib.rs`): `MonoAD`, `MultiAD`, `MonoAD2RR`, `MonoAD2FR`, `MonoAD2RF`, `GraphBuilder`

**Modules**:
- `src/mono/` - Single-variable automatic differentiation
- `src/multi/` - Multi-variable automatic differentiation
- `src/error.rs` - Error types (`AutodiffError`)
- `src/macros.rs` - Macros (`mono_ops!`, `multi_ops!`)

**Tests**: Unit tests in `src/*/tests.rs`, comprehensive tests in `src/tests_comprehensive.rs`

**Benchmarks**: `benches/compute_benchmark.rs`, `benches/hessian_benchmark.rs`

## Code Style & Conventions

### Imports & Organization

- **Standard library first**, then external crates, then internal modules (separated by blank lines)
- Use explicit imports for clarity
- Re-export public types in `lib.rs` for API simplicity
- Use `crate::Result` for error-returning functions

### Naming & Types

- **Functions**: `snake_case`, methods starting with action verbs (`compute_`, `forward_`, `backward_`)
- **Types**: `PascalCase` for structs/enums/traits
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Private helpers**: Prefix internal functions with underscore if needed
- **Type annotations**: Always explicit for public functions and struct fields

### Error Handling

- Use `crate::Result<T>` (alias for `std::result::Result<T, AutodiffError>`)
- Use `?` operator for propagation
- Document error conditions in doc comments

### Performance

- Use `#[inline]` and `#[inline(always)]` for hot functions
- Pre-allocate vectors with `with_capacity()` when size is known
- Capture computed values in closures to avoid recomputation
- Use mold linker for faster builds (configured in `.cargo/config.toml`)

### Comments & Documentation

- **Public items**: Full doc comments `///` with examples
- **Complex logic**: Explain the "why" not the "what"
- **Inline comments**: Use `//` for clarification, keep minimal
- **Tests**: Include usage examples in doc comments for public APIs

### Formatting

- 4-space indentation (enforced by `cargo fmt`)
- Max line length: ~100 characters (soft limit)
- One statement per line; use temporary variables for readability

## Pre-commit Hooks

All commits run: `cargo fmt --check` → `cargo clippy -- -D warnings` → `cargo test --all-features`

### ⚠️ CRITICAL: Never use `--no-verify`

**❌ NEVER use `git commit --no-verify`** - This bypasses all pre-commit checks and will likely cause CI failures.

If you used `--no-verify` and CI fails:
1. Fix the issues (usually formatting: run `cargo fmt`)
2. Re-stage files: `git add .`
3. Commit **without** `--no-verify`: `git commit -m "fix: ..."`
4. Push and verify CI passes

### Common Pre-commit Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| `cargo fmt` | Code formatting | Run `cargo fmt` then re-stage |
| `cargo clippy` | Lint warnings | Fix warnings or run `cargo clippy --fix` |
| `cargo test` | Test failures | Fix failing tests |

## GitHub CI

The CI workflow (`.github/workflows/ci.yml`) runs:
- **Test matrix**: Rust stable and nightly
- **Coverage**: Code coverage with cargo-tarpaulin
- **Security audit**: cargo-audit for vulnerability checking

### Post-Commit CI Verification Checklist

After pushing any commit, agents MUST verify CI passes:

- [ ] Push commit and note the run ID
- [ ] Run `gh run watch --repo leafyoung/petite-ad --exit-status` until completion
- [ ] If CI fails, run `gh run view --log-failed --repo leafyoung/petite-ad` to diagnose
- [ ] Fix any issues and re-push until all jobs pass
- [ ] Only proceed to next task after CI is green

### GitHub CLI Commands

```bash
# View CI run status (get run ID from push output or list)
gh run list --repo leafyoung/petite-ad --limit 3

# Watch CI run progress with live updates
gh run watch <run-id> --repo leafyoung/petite-ad --exit-status

# If CI fails, view detailed failure logs
gh run view <run-id> --repo leafyoung/petite-ad --log-failed

# View specific job logs
gh run view <run-id> --repo leafyoung/petite-ad --job <job-id>
```

## Key Patterns in Codebase

### Forward Pass

```rust
// ✅ Good: Pre-allocate with capacity, inline hot functions
#[inline]
pub fn compute(exprs: &[MonoAD], x: f64) -> f64 {
    let mut value = x;
    for expr in exprs {
        value = expr.forward(value);
    }
    value
}
```

### Backward Pass

```rust
// ✅ Good: Capture computed values in closures
MonoAD::Sin => {
    let y = x.sin();
    let x_cos = x.cos(); // Capture to avoid recomputation
    let grad = Box::new(move |dy: f64| -> f64 { dy * x_cos });
    (y, grad)
}
```

### Error Handling

```rust
// ✅ Good: Use Result type alias and ? operator
pub fn compute(exprs: &[(MultiAD, Vec<usize>)], inputs: &[f64]) -> Result<f64> {
    let mut values: Vec<f64> = inputs.to_vec();
    // ... operations with ? for error propagation
    Ok(values.last().copied().unwrap_or(0.0))
}
```

## Important Constraints

- ❌ Never: `cargo clean` (breaks build cache)
- ❌ Never: Use `git commit --no-verify` (bypasses pre-commit checks)
- ❌ Never: Commit `Cargo.lock` if this is a library (only for binaries)
- ✅ Always: Use `cargo test` before committing
- ✅ Always: Run `cargo fmt` on modified files
- ✅ Always: Verify CI passes after pushing

## Testing Strategy

**Unit tests** (`#[test]`): Logic validation in `src/*/tests.rs`
**Comprehensive tests** (`src/tests_comprehensive.rs`): Edge cases and coverage
**Benchmarks** (`benches/`): Performance regression testing
**Doc tests**: Examples in `///` doc comments

Example: `cargo test test_mono_neg_forward -- --nocapture`

## Coverage

Target: >85% code coverage (currently ~87%)

```bash
# Generate coverage report
cargo tarpaulin --lib --out Stdout

# Generate HTML report
cargo tarpaulin --lib --out Html
```
