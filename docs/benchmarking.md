# Benchmarking

This repo supports two workflows:
- A lightweight perf harness for hardware counters and microarchitectural analysis.
- Criterion benchmarks for statistical timing and regression tracking.

## Perf harness
The perf harness is a standalone binary you can drive with `perf stat`.

Build and list benches:
```sh
cargo build --release --bin perf_harness
target/release/perf_harness --list
```

Run a bench directly:
```sh
target/release/perf_harness --bench gcd_scalar --len 100000 --iters 100
```

Helper script (builds the harness and runs perf with a default event set):
```sh
scripts/perf_stat.sh --bench gcd_scalar --len 100000 --iters 100
```

Use perf counters (repeat `-e` to avoid shell line-break issues):
```sh
perf stat -r 5 \
  -e cycles -e instructions -e branches -e branch-misses \
  -e cache-references -e cache-misses \
  -e L1-dcache-loads -e L1-dcache-load-misses \
  -e dTLB-loads -e dTLB-load-misses \
  -e stalled-cycles-frontend -e stalled-cycles-backend \
  -- target/release/perf_harness --bench gcd_scalar --len 100000 --iters 100
```

Notes:
- Use `perf list` to see CPU-specific events (L2/LLC names vary).
- `--no-reset` skips per-iteration resets for in-place algorithms.
- `--verify` runs a quick correctness check before timing.
- `--report` prints throughput and time per item (pairs/s for gcd, elem/s and B/s for prefix_sum).
- GFLOP is not meaningful for integer-only kernels; use elements or bytes per second.

To add new algorithms, extend `src/bin/perf_harness.rs` with a new bench enum
variant, parser entry, and bench function.

## Criterion benches
Criterion runs under `benches/` and provides statistical timing with warmup,
outlier detection, and change reports.

Examples:
```sh
cargo bench --bench gcd
cargo bench --bench prefix_sum
```

Results are stored under `target/criterion/`.

## When to use which
- Use the perf harness when you need hardware counters (IPC, cache/TLB misses,
  stalls) or want minimal framework overhead.
- Use Criterion when you want stable wall-clock comparisons and regression
  tracking across changes.
