# Long-term Priority Fixes - API Simplification

## Summary

Simplified the API by removing Box/Arc duplication. The library now defaults to `Box<dyn Fn>` for better performance and simplicity, with easy conversion to `Arc` when needed.

## Changes Made

### 1. Removed Arc Variants
- **Removed methods**: `compute_grad_arc()` from both `MonoAD` and `MultiAD`
- **Removed methods**: `auto_grad_arc()` from `MonoFn` and `MultiFn` traits
- **Simplified API**: Single method `compute_grad()` returns `Box`, users convert to `Arc` when needed

### 2. Arc Conversion Pattern
Users can now convert Box to Arc explicitly:
```rust
let (value, grad_fn) = MonoAD::compute_grad(&ops, x);
let arc_grad_fn: Arc<dyn Fn(f64) -> f64> = Arc::from(grad_fn);
```

### 3. Performance Optimizations
- **Pre-allocated buffers**: Added `Vec::with_capacity()` in MultiAD for better performance
- **Simplified backward pass**: Removed generic wrapper complexity
- **Cleaner implementation**: Reduced code duplication

### 4. Updated Documentation
- Updated all doctests to show Arc conversion pattern
- Fixed README to reflect simplified API
- Updated examples to demonstrate Arc usage
- Removed confusing Box/Arc comparison sections

### 5. Test Updates
- Fixed all tests to use single `compute_grad()` method
- Removed Box/Arc comparison tests (no longer needed)
- Updated benchmarks to use simplified API

## Benefits

1. **Simpler API**: One method instead of two for gradient computation
2. **Better defaults**: Box is faster and sufficient for most use cases
3. **Explicit conversion**: Users only pay Arc overhead when they need it
4. **Clearer documentation**: Less confusion about which method to use
5. **Easier maintenance**: Less code duplication

## Migration Guide

### Before (Old API)
```rust
// Had to choose between Box and Arc
let (v1, g1) = MonoAD::compute_grad(&ops, x);      // Box
let (v2, g2) = MonoAD::compute_grad_arc(&ops, x);  // Arc
```

### After (New API)
```rust
// Default to Box, convert to Arc when needed
let (value, grad_fn) = MonoAD::compute_grad(&ops, x);
let arc_fn: Arc<dyn Fn(f64) -> f64> = Arc::from(grad_fn); // if needed
```

## Test Results

- ✅ 24 unit tests passing
- ✅ 10 doc tests passing  
- ✅ All examples building and running
- ✅ Clean release build (zero warnings)
- ✅ Benchmarks compiling successfully

## Files Modified

- `src/mono/mono_ad.rs` - Removed Arc variant, added capacity pre-allocation
- `src/multi/multi_ad.rs` - Removed Arc variant, optimized with pre-allocated buffers
- `src/mono/mono_fn.rs` - Removed `auto_grad_arc()`, updated demonstrate()
- `src/multi/multi_fn.rs` - Removed `auto_grad_arc()`, updated demonstrate()
- `src/mono/tests.rs` - Updated tests to use single API
- `src/multi/tests.rs` - Updated tests to use single API
- `examples/using_new_api.rs` - Added Arc conversion example
- `examples/multi_macro_demo.rs` - Updated to new API
- `benches/compute_benchmark.rs` - Updated all benchmarks
- `README.md` - Updated documentation to reflect API changes
