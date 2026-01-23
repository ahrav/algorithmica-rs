# hpc-algorithms

Rust implementations of algorithms from [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/) with benchmarks comparing different variants.

## Implemented

- **Argmin**: Scalar, SIMD-backed (AVX2/AVX-512/NEON), min+find, and blocked variants
- **Binary Search**: Stdlib, branchless, branchless+prefetch, Eytzinger layout, Eytzinger+prefetch
- **S-tree / S+ tree**: Implicit B-tree and B+ tree layouts (B=16, B=32) with scalar and NEON search
- **GCD**: Euclidean vs Binary (Stein's algorithm)
- **Matrix Multiplication**: Baseline, transposed, loop-reordered, register-blocked, cache-blocked, NEON-blocked (aarch64)
- **Prefix Sum**: Scalar vs NEON SIMD (aarch64)

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific algorithm
cargo bench -- gcd
cargo bench -- argmin
cargo bench -- binary_search
cargo bench -- s_tree
cargo bench -- matmul
cargo bench -- prefix_sum

# Opt-in AVX-512 path for argmin on x86_64
ARGMIN_AVX512=1 cargo bench -- argmin
```

## Perf Analysis

```bash
# Hardware counter profiling
./scripts/perf_stat.sh --bench argmin_simd_filtered --len 1000000 --iters 10
```

## License

MIT
