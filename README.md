# hpc-algorithms

Rust implementations of algorithms from [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/) with benchmarks comparing different variants.

## Implemented

- **Argmin**: Scalar, SIMD-backed (AVX2/NEON), min+find, and blocked variants
- **GCD**: Euclidean vs Binary (Stein's algorithm)
- **Matrix Multiplication**: Baseline, transposed, loop-reordered, register-blocked, cache-blocked variants
- **Prefix Sum**: Scalar vs NEON SIMD (aarch64)

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific algorithm
cargo bench -- gcd
cargo bench -- argmin
cargo bench -- matmul
cargo bench -- prefix_sum
```

## Perf Analysis

```bash
# Hardware counter profiling
./scripts/perf_stat.sh --bench argmin_simd_filtered --len 1000000 --iters 10
```

## License

MIT
