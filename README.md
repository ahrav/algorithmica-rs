# hpc-algorithms

Rust implementations of algorithms from [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/) with benchmarks comparing different variants.

## Implemented

- **GCD**: Euclidean vs Binary (Stein's algorithm)
- **Prefix Sum**: Scalar vs NEON SIMD (aarch64)

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific algorithm
cargo bench -- gcd
cargo bench -- prefix_sum
```

## Perf Analysis

```bash
# Hardware counter profiling
./scripts/perf_stat.sh gcd
```

## License

MIT
