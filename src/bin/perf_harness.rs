use std::env;
use std::hint::black_box;
use std::process;
use std::time::Instant;

use hpc_algorithms::{
    BLOCK_SIZE, EytzingerLayout, SPlusTreeLayout, STreeLayout, argmin_blocked, argmin_branchless,
    argmin_min_then_find, argmin_scalar, argmin_simd_filtered, argmin_std, argmin_vector_indices,
    binary_search_branchless, binary_search_branchless_prefetch, binary_search_eytzinger,
    binary_search_eytzinger_prefetch, binary_search_std, gcd_binary, gcd_scalar, matmul_baseline,
    matmul_blocked, matmul_ikj, matmul_register_blocked_2x2, matmul_transposed, prefix_sum_scalar,
    prefix_sum_scalar_in_place, s_plus_tree_search_neon, s_plus_tree_search_scalar,
    s_tree_search_neon, s_tree_search_scalar,
};

#[cfg(target_arch = "aarch64")]
use hpc_algorithms::{matmul_neon_blocked, prefix_sum, prefix_sum_blocked, prefix_sum_interleaved};

const DEFAULT_SEED: u64 = 0x1234_5678_9ABC_DEF0;

#[derive(Clone, Copy)]
enum Bench {
    ArgminStd,
    ArgminScalar,
    ArgminBranchless,
    ArgminMinThenFind,
    ArgminBlocked,
    ArgminSimdIndices,
    ArgminSimdFiltered,
    BinarySearchStd,
    BinarySearchBranchless,
    BinarySearchBranchlessPrefetch,
    BinarySearchEytzinger,
    BinarySearchEytzingerPrefetch,
    STreeScalar,
    STreeNeon,
    SPlusTreeScalar,
    SPlusTreeNeon,
    MatmulBaseline,
    MatmulTransposed,
    MatmulIkj,
    MatmulRegister2x2,
    MatmulBlocked,
    #[cfg(target_arch = "aarch64")]
    MatmulNeonBlocked,
    GcdScalar,
    GcdBinary,
    PrefixSumScalar,
    PrefixSumScalarInPlace,
    #[cfg(target_arch = "aarch64")]
    PrefixSumNeon,
    #[cfg(target_arch = "aarch64")]
    PrefixSumNeonBlocked,
    #[cfg(target_arch = "aarch64")]
    PrefixSumNeonInterleaved,
}

#[derive(Clone, Copy)]
struct Config {
    bench: Bench,
    len: usize,
    iters: usize,
    seed: u64,
    reset: bool,
    verify: bool,
    report: bool,
}

type BinarySearchFn = fn(&[i32], i32) -> Option<usize>;
type BinarySearchEytzingerFn = fn(&EytzingerLayout, i32) -> Option<usize>;
type STreeFn = fn(&STreeLayout, i32) -> Option<usize>;
type SPlusTreeFn = fn(&SPlusTreeLayout, i32) -> Option<usize>;
type MatmulFn = fn(&[f32], &[f32], &mut [f32], usize);

fn main() {
    let config = match parse_args() {
        Ok(result) => result,
        Err(err) => {
            eprintln!("error: {err}");
            print_usage(&program_name());
            process::exit(2);
        }
    };

    if config.verify {
        verify_bench(config.bench);
    }

    run_bench(config);
}

fn parse_args() -> Result<Config, String> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "perf_harness".to_string());

    let mut bench = None;
    let mut len = None;
    let mut iters = None;
    let mut seed = DEFAULT_SEED;
    let mut reset = true;
    let mut verify = false;
    let mut report = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {
                let name = args.next().ok_or("--bench requires a value")?;
                let parsed = parse_bench(&name).ok_or_else(|| format!("unknown bench: {name}"))?;
                bench = Some(parsed);
            }
            "--len" => {
                let value = args.next().ok_or("--len requires a value")?;
                len = Some(parse_usize(&value, "--len")?);
            }
            "--iters" => {
                let value = args.next().ok_or("--iters requires a value")?;
                iters = Some(parse_usize(&value, "--iters")?);
            }
            "--seed" => {
                let value = args.next().ok_or("--seed requires a value")?;
                seed = parse_u64(&value, "--seed")?;
            }
            "--reset" => reset = true,
            "--no-reset" => reset = false,
            "--verify" => verify = true,
            "--report" => report = true,
            "--no-report" => report = false,
            "--list" => {
                list_benches();
                process::exit(0);
            }
            "-h" | "--help" => {
                print_usage(&program);
                process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    let bench = bench.ok_or("missing --bench")?;
    let len = len.unwrap_or_else(|| bench.default_len());
    let iters = iters.unwrap_or_else(|| bench.default_iters());

    Ok(Config {
        bench,
        len,
        iters,
        seed,
        reset,
        verify,
        report,
    })
}

fn program_name() -> String {
    env::args()
        .next()
        .unwrap_or_else(|| "perf_harness".to_string())
}

fn print_usage(program: &str) {
    eprintln!(
        "\
Usage:
  {program} --bench <name> [--len N] [--iters N] [--seed N] [--no-reset] [--verify]
  {program} --list

Options:
  --bench <name>   Benchmark to run (see --list)
  --len N          Input length (elements or pairs, bench-specific default)
  --iters N        Iterations (bench-specific default)
  --seed N         RNG seed (default: 0x123456789ABCDEF0)
  --no-reset       Skip resetting in-place inputs each iteration
  --verify         Run a quick correctness check before benchmarking
  --report         Print throughput summary after the run
  --no-report      Disable throughput summary
  --list           Show available benches
"
    );
}

fn list_benches() {
    println!("argmin_std");
    println!("argmin_scalar");
    println!("argmin_branchless");
    println!("argmin_min_then_find");
    println!("argmin_blocked");
    println!("argmin_simd_indices");
    println!("argmin_simd_filtered");
    println!("binary_search_std");
    println!("binary_search_branchless");
    println!("binary_search_branchless_prefetch");
    println!("binary_search_eytzinger");
    println!("binary_search_eytzinger_prefetch");
    println!("s_tree_scalar");
    println!("s_tree_neon");
    println!("s_plus_tree_scalar");
    println!("s_plus_tree_neon");
    println!("matmul_baseline");
    println!("matmul_transposed");
    println!("matmul_ikj");
    println!("matmul_register_2x2");
    println!("matmul_blocked");
    #[cfg(target_arch = "aarch64")]
    println!("matmul_neon_blocked");
    println!("gcd_scalar");
    println!("gcd_binary");
    println!("prefix_sum_scalar");
    println!("prefix_sum_scalar_in_place");
    #[cfg(target_arch = "aarch64")]
    {
        println!("prefix_sum_neon");
        println!("prefix_sum_neon_blocked");
        println!("prefix_sum_neon_interleaved");
    }
}

fn parse_bench(name: &str) -> Option<Bench> {
    match name {
        "argmin_std" => Some(Bench::ArgminStd),
        "argmin_scalar" => Some(Bench::ArgminScalar),
        "argmin_branchless" => Some(Bench::ArgminBranchless),
        "argmin_min_then_find" => Some(Bench::ArgminMinThenFind),
        "argmin_blocked" => Some(Bench::ArgminBlocked),
        "argmin_simd_indices" => Some(Bench::ArgminSimdIndices),
        "argmin_simd_filtered" => Some(Bench::ArgminSimdFiltered),
        "binary_search_std" => Some(Bench::BinarySearchStd),
        "binary_search_branchless" => Some(Bench::BinarySearchBranchless),
        "binary_search_branchless_prefetch" => Some(Bench::BinarySearchBranchlessPrefetch),
        "binary_search_eytzinger" => Some(Bench::BinarySearchEytzinger),
        "binary_search_eytzinger_prefetch" => Some(Bench::BinarySearchEytzingerPrefetch),
        "s_tree_scalar" => Some(Bench::STreeScalar),
        "s_tree_neon" => Some(Bench::STreeNeon),
        "s_plus_tree_scalar" => Some(Bench::SPlusTreeScalar),
        "s_plus_tree_neon" => Some(Bench::SPlusTreeNeon),
        "matmul_baseline" => Some(Bench::MatmulBaseline),
        "matmul_transposed" => Some(Bench::MatmulTransposed),
        "matmul_ikj" => Some(Bench::MatmulIkj),
        "matmul_register_2x2" => Some(Bench::MatmulRegister2x2),
        "matmul_blocked" => Some(Bench::MatmulBlocked),
        #[cfg(target_arch = "aarch64")]
        "matmul_neon_blocked" => Some(Bench::MatmulNeonBlocked),
        "gcd_scalar" => Some(Bench::GcdScalar),
        "gcd_binary" => Some(Bench::GcdBinary),
        "prefix_sum_scalar" => Some(Bench::PrefixSumScalar),
        "prefix_sum_scalar_in_place" => Some(Bench::PrefixSumScalarInPlace),
        #[cfg(target_arch = "aarch64")]
        "prefix_sum_neon" => Some(Bench::PrefixSumNeon),
        #[cfg(target_arch = "aarch64")]
        "prefix_sum_neon_blocked" => Some(Bench::PrefixSumNeonBlocked),
        #[cfg(target_arch = "aarch64")]
        "prefix_sum_neon_interleaved" => Some(Bench::PrefixSumNeonInterleaved),
        _ => None,
    }
}

impl Bench {
    fn default_len(self) -> usize {
        match self {
            Bench::ArgminStd
            | Bench::ArgminScalar
            | Bench::ArgminBranchless
            | Bench::ArgminMinThenFind
            | Bench::ArgminBlocked
            | Bench::ArgminSimdIndices
            | Bench::ArgminSimdFiltered
            | Bench::BinarySearchStd
            | Bench::BinarySearchBranchless
            | Bench::BinarySearchBranchlessPrefetch
            | Bench::BinarySearchEytzinger
            | Bench::BinarySearchEytzingerPrefetch
            | Bench::STreeScalar
            | Bench::STreeNeon
            | Bench::SPlusTreeScalar
            | Bench::SPlusTreeNeon => 1_000_000,
            Bench::MatmulBaseline
            | Bench::MatmulTransposed
            | Bench::MatmulIkj
            | Bench::MatmulRegister2x2
            | Bench::MatmulBlocked => 256,
            #[cfg(target_arch = "aarch64")]
            Bench::MatmulNeonBlocked => 256,
            Bench::GcdScalar | Bench::GcdBinary => 100_000,
            Bench::PrefixSumScalar | Bench::PrefixSumScalarInPlace => 1_000_000,
            #[cfg(target_arch = "aarch64")]
            Bench::PrefixSumNeon
            | Bench::PrefixSumNeonBlocked
            | Bench::PrefixSumNeonInterleaved => 1_000_000,
        }
    }

    fn default_iters(self) -> usize {
        match self {
            Bench::ArgminStd
            | Bench::ArgminScalar
            | Bench::ArgminBranchless
            | Bench::ArgminMinThenFind
            | Bench::ArgminBlocked
            | Bench::ArgminSimdIndices
            | Bench::ArgminSimdFiltered => 10,
            Bench::BinarySearchStd
            | Bench::BinarySearchBranchless
            | Bench::BinarySearchBranchlessPrefetch
            | Bench::BinarySearchEytzinger
            | Bench::BinarySearchEytzingerPrefetch
            | Bench::STreeScalar
            | Bench::STreeNeon
            | Bench::SPlusTreeScalar
            | Bench::SPlusTreeNeon => 5,
            Bench::MatmulBaseline
            | Bench::MatmulTransposed
            | Bench::MatmulIkj
            | Bench::MatmulRegister2x2
            | Bench::MatmulBlocked => 3,
            #[cfg(target_arch = "aarch64")]
            Bench::MatmulNeonBlocked => 3,
            Bench::GcdScalar | Bench::GcdBinary => 100,
            Bench::PrefixSumScalar | Bench::PrefixSumScalarInPlace => 10,
            #[cfg(target_arch = "aarch64")]
            Bench::PrefixSumNeon
            | Bench::PrefixSumNeonBlocked
            | Bench::PrefixSumNeonInterleaved => 10,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Bench::ArgminStd => "argmin_std",
            Bench::ArgminScalar => "argmin_scalar",
            Bench::ArgminBranchless => "argmin_branchless",
            Bench::ArgminMinThenFind => "argmin_min_then_find",
            Bench::ArgminBlocked => "argmin_blocked",
            Bench::ArgminSimdIndices => "argmin_simd_indices",
            Bench::ArgminSimdFiltered => "argmin_simd_filtered",
            Bench::BinarySearchStd => "binary_search_std",
            Bench::BinarySearchBranchless => "binary_search_branchless",
            Bench::BinarySearchBranchlessPrefetch => "binary_search_branchless_prefetch",
            Bench::BinarySearchEytzinger => "binary_search_eytzinger",
            Bench::BinarySearchEytzingerPrefetch => "binary_search_eytzinger_prefetch",
            Bench::STreeScalar => "s_tree_scalar",
            Bench::STreeNeon => "s_tree_neon",
            Bench::SPlusTreeScalar => "s_plus_tree_scalar",
            Bench::SPlusTreeNeon => "s_plus_tree_neon",
            Bench::MatmulBaseline => "matmul_baseline",
            Bench::MatmulTransposed => "matmul_transposed",
            Bench::MatmulIkj => "matmul_ikj",
            Bench::MatmulRegister2x2 => "matmul_register_2x2",
            Bench::MatmulBlocked => "matmul_blocked",
            #[cfg(target_arch = "aarch64")]
            Bench::MatmulNeonBlocked => "matmul_neon_blocked",
            Bench::GcdScalar => "gcd_scalar",
            Bench::GcdBinary => "gcd_binary",
            Bench::PrefixSumScalar => "prefix_sum_scalar",
            Bench::PrefixSumScalarInPlace => "prefix_sum_scalar_in_place",
            #[cfg(target_arch = "aarch64")]
            Bench::PrefixSumNeon => "prefix_sum_neon",
            #[cfg(target_arch = "aarch64")]
            Bench::PrefixSumNeonBlocked => "prefix_sum_neon_blocked",
            #[cfg(target_arch = "aarch64")]
            Bench::PrefixSumNeonInterleaved => "prefix_sum_neon_interleaved",
        }
    }
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} expects a non-negative integer"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{flag} expects a non-negative integer"))
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn make_pairs(len: usize, seed: u64) -> Vec<(i64, i64)> {
    let mut state = seed;
    let mut pairs = Vec::with_capacity(len);
    for _ in 0..len {
        let a = ((next_u64(&mut state) & 0x7FFF_FFFF) as i64) + 1;
        let b = ((next_u64(&mut state) & 0x7FFF_FFFF) as i64) + 1;
        pairs.push((a, b));
    }
    pairs
}

fn make_i32_input(len: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(next_u64(&mut state) as u32 as i32);
    }
    values
}

fn make_sorted_values(len: usize) -> Vec<i32> {
    let mut values = Vec::with_capacity(len);
    for i in 0..len {
        values.push((i as i32) * 2);
    }
    values
}

fn make_search_queries(values: &[i32], seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut queries = Vec::with_capacity(values.len());
    for _ in 0..values.len() {
        let idx = (next_u64(&mut state) as usize) % values.len();
        queries.push(values[idx]);
    }
    queries
}

fn make_f32_matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(n * n);
    for _ in 0..n * n {
        let v = (next_u64(&mut state) & 0xFF) as f32;
        values.push(v / 255.0);
    }
    values
}

fn run_bench(config: Config) {
    let stats = bench_stats(config.bench, &config);
    let start = Instant::now();
    match config.bench {
        Bench::ArgminStd => bench_argmin(config, argmin_std),
        Bench::ArgminScalar => bench_argmin(config, argmin_scalar),
        Bench::ArgminBranchless => bench_argmin(config, argmin_branchless),
        Bench::ArgminMinThenFind => bench_argmin(config, argmin_min_then_find),
        Bench::ArgminBlocked => bench_argmin(config, argmin_blocked),
        Bench::ArgminSimdIndices => bench_argmin(config, argmin_vector_indices),
        Bench::ArgminSimdFiltered => bench_argmin(config, argmin_simd_filtered),
        Bench::BinarySearchStd => bench_binary_search(config, binary_search_std),
        Bench::BinarySearchBranchless => bench_binary_search(config, binary_search_branchless),
        Bench::BinarySearchBranchlessPrefetch => {
            bench_binary_search(config, binary_search_branchless_prefetch);
        }
        Bench::BinarySearchEytzinger => {
            bench_binary_search_eytzinger(config, binary_search_eytzinger);
        }
        Bench::BinarySearchEytzingerPrefetch => {
            bench_binary_search_eytzinger(config, binary_search_eytzinger_prefetch);
        }
        Bench::STreeScalar => bench_s_tree(config, s_tree_search_scalar),
        Bench::STreeNeon => bench_s_tree(config, s_tree_search_neon),
        Bench::SPlusTreeScalar => bench_s_plus_tree(config, s_plus_tree_search_scalar),
        Bench::SPlusTreeNeon => bench_s_plus_tree(config, s_plus_tree_search_neon),
        Bench::MatmulBaseline => bench_matmul(config, matmul_baseline),
        Bench::MatmulTransposed => bench_matmul(config, matmul_transposed),
        Bench::MatmulIkj => bench_matmul(config, matmul_ikj),
        Bench::MatmulRegister2x2 => bench_matmul(config, matmul_register_blocked_2x2),
        Bench::MatmulBlocked => bench_matmul(config, matmul_blocked),
        #[cfg(target_arch = "aarch64")]
        Bench::MatmulNeonBlocked => bench_matmul(config, matmul_neon_blocked),
        Bench::GcdScalar => bench_gcd_scalar(config),
        Bench::GcdBinary => bench_gcd_binary(config),
        Bench::PrefixSumScalar => bench_prefix_sum_scalar(config),
        Bench::PrefixSumScalarInPlace => bench_prefix_sum_scalar_in_place(config),
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeon => bench_prefix_sum_neon(config),
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeonBlocked => bench_prefix_sum_neon_blocked(config),
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeonInterleaved => bench_prefix_sum_neon_interleaved(config),
    }
    let elapsed = start.elapsed();
    if config.report {
        print_report(config.bench, &config, stats, elapsed);
    }
}

struct BenchStats {
    work_items: u128,
    bytes: u128,
    unit: &'static str,
}

fn bench_stats(bench: Bench, config: &Config) -> BenchStats {
    let work_items = (config.len as u128) * (config.iters as u128);
    match bench {
        Bench::ArgminStd
        | Bench::ArgminScalar
        | Bench::ArgminBranchless
        | Bench::ArgminSimdIndices
        | Bench::ArgminSimdFiltered => BenchStats {
            work_items,
            bytes: work_items * 4,
            unit: "elem",
        },
        Bench::BinarySearchStd
        | Bench::BinarySearchBranchless
        | Bench::BinarySearchBranchlessPrefetch
        | Bench::BinarySearchEytzinger
        | Bench::BinarySearchEytzingerPrefetch
        | Bench::STreeScalar
        | Bench::STreeNeon
        | Bench::SPlusTreeScalar
        | Bench::SPlusTreeNeon => BenchStats {
            work_items,
            bytes: work_items * 4,
            unit: "query",
        },
        Bench::ArgminMinThenFind => BenchStats {
            work_items,
            bytes: work_items * 8,
            unit: "elem",
        },
        Bench::ArgminBlocked => BenchStats {
            work_items,
            bytes: work_items * 4 + (BLOCK_SIZE as u128) * (config.iters as u128) * 4,
            unit: "elem",
        },
        Bench::MatmulBaseline
        | Bench::MatmulTransposed
        | Bench::MatmulIkj
        | Bench::MatmulRegister2x2
        | Bench::MatmulBlocked => {
            let n = config.len as u128;
            let iters = config.iters as u128;
            let ops = n * n * n * iters;
            let bytes = n * n * 12u128 * iters;
            BenchStats {
                work_items: ops,
                bytes,
                unit: "mul",
            }
        }
        #[cfg(target_arch = "aarch64")]
        Bench::MatmulNeonBlocked => {
            let n = config.len as u128;
            let iters = config.iters as u128;
            let ops = n * n * n * iters;
            let bytes = n * n * 12u128 * iters;
            BenchStats {
                work_items: ops,
                bytes,
                unit: "mul",
            }
        }
        Bench::GcdScalar | Bench::GcdBinary => BenchStats {
            work_items,
            bytes: work_items * 16,
            unit: "pair",
        },
        Bench::PrefixSumScalar | Bench::PrefixSumScalarInPlace => BenchStats {
            work_items,
            bytes: work_items * 8,
            unit: "elem",
        },
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeon | Bench::PrefixSumNeonBlocked | Bench::PrefixSumNeonInterleaved => {
            BenchStats {
                work_items,
                bytes: work_items * 8,
                unit: "elem",
            }
        }
    }
}

fn print_report(bench: Bench, config: &Config, stats: BenchStats, elapsed: std::time::Duration) {
    let elapsed_s = elapsed.as_secs_f64();
    let items_per_s = stats.work_items as f64 / elapsed_s;
    let bytes_per_s = stats.bytes as f64 / elapsed_s;
    let ns_per_item = (elapsed_s * 1.0e9) / stats.work_items as f64;
    let gflops = match bench {
        Bench::MatmulBaseline
        | Bench::MatmulTransposed
        | Bench::MatmulIkj
        | Bench::MatmulRegister2x2
        | Bench::MatmulBlocked => Some(format_rate(items_per_s * 2.0, "FLOP")),
        #[cfg(target_arch = "aarch64")]
        Bench::MatmulNeonBlocked => Some(format_rate(items_per_s * 2.0, "FLOP")),
        _ => None,
    };

    let mut lines = Vec::with_capacity(5);
    lines.push(format!(
        "bench={} len={} iters={}",
        bench.name(),
        config.len,
        config.iters
    ));
    lines.push(format!(
        "elapsed_s={:.6} ns_per_item={:.3} throughput={}",
        elapsed_s,
        ns_per_item,
        format_rate(items_per_s, stats.unit)
    ));
    lines.push(format!(
        "work_items={} unit={}",
        stats.work_items, stats.unit
    ));
    lines.push(format!(
        "bytes={} byte_throughput={}",
        stats.bytes,
        format_rate(bytes_per_s, "B")
    ));

    if let Some(gflops) = gflops {
        lines.push(format!("gflops={}", gflops));
    }

    println!("{}", lines.join("\n"));
}

fn format_rate(rate: f64, unit: &str) -> String {
    let (value, prefix) = if rate >= 1.0e12 {
        (rate / 1.0e12, "T")
    } else if rate >= 1.0e9 {
        (rate / 1.0e9, "G")
    } else if rate >= 1.0e6 {
        (rate / 1.0e6, "M")
    } else if rate >= 1.0e3 {
        (rate / 1.0e3, "K")
    } else {
        (rate, "")
    };
    format!("{value:.3} {prefix}{unit}/s")
}

fn make_small_matrix(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            out.push(((i * 3 + j * 5) % 7) as f32);
        }
    }
    out
}

fn verify_matmul_variant(func: MatmulFn) {
    let n = 5;
    let a = make_small_matrix(n);
    let b = make_small_matrix(n);
    let mut expected = vec![0.0f32; n * n];
    matmul_baseline(&a, &b, &mut expected, n);

    let mut out = vec![0.0f32; n * n];
    func(&a, &b, &mut out, n);
    assert_eq!(out, expected);
}

fn verify_bench(bench: Bench) {
    match bench {
        Bench::ArgminStd => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_std(&values), Some(1));
        }
        Bench::ArgminScalar => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_scalar(&values), Some(1));
        }
        Bench::ArgminBranchless => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_branchless(&values), Some(1));
        }
        Bench::ArgminMinThenFind => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_min_then_find(&values), Some(1));
        }
        Bench::ArgminBlocked => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_blocked(&values), Some(1));
        }
        Bench::ArgminSimdIndices => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_vector_indices(&values), Some(1));
        }
        Bench::ArgminSimdFiltered => {
            let values = [3, 1, 2, 1];
            assert_eq!(argmin_simd_filtered(&values), Some(1));
        }
        Bench::BinarySearchStd => {
            let values = [1, 3, 5, 7, 9];
            assert_eq!(binary_search_std(&values, 5), Some(2));
            assert_eq!(binary_search_std(&values, 6), None);
        }
        Bench::BinarySearchBranchless => {
            let values = [1, 3, 5, 7, 9];
            assert_eq!(binary_search_branchless(&values, 5), Some(2));
            assert_eq!(binary_search_branchless(&values, 6), None);
        }
        Bench::BinarySearchBranchlessPrefetch => {
            let values = [1, 3, 5, 7, 9];
            assert_eq!(binary_search_branchless_prefetch(&values, 5), Some(2));
            assert_eq!(binary_search_branchless_prefetch(&values, 6), None);
        }
        Bench::BinarySearchEytzinger => {
            let values = [1, 3, 5, 7, 9];
            let layout = EytzingerLayout::new(&values);
            assert_eq!(binary_search_eytzinger(&layout, 5), Some(2));
            assert_eq!(binary_search_eytzinger(&layout, 6), None);
        }
        Bench::BinarySearchEytzingerPrefetch => {
            let values = [1, 3, 5, 7, 9];
            let layout = EytzingerLayout::new(&values);
            assert_eq!(binary_search_eytzinger_prefetch(&layout, 5), Some(2));
            assert_eq!(binary_search_eytzinger_prefetch(&layout, 6), None);
        }
        Bench::STreeScalar => {
            let values = [1, 3, 5, 7, 9];
            let layout = STreeLayout::new(&values);
            assert_eq!(s_tree_search_scalar(&layout, 5), Some(2));
            assert_eq!(s_tree_search_scalar(&layout, 6), None);
        }
        Bench::STreeNeon => {
            let values = [1, 3, 5, 7, 9];
            let layout = STreeLayout::new(&values);
            assert_eq!(s_tree_search_neon(&layout, 5), Some(2));
            assert_eq!(s_tree_search_neon(&layout, 6), None);
        }
        Bench::SPlusTreeScalar => {
            let values = [1, 3, 5, 7, 9];
            let layout = SPlusTreeLayout::new(&values);
            assert_eq!(s_plus_tree_search_scalar(&layout, 5), Some(2));
            assert_eq!(s_plus_tree_search_scalar(&layout, 6), None);
        }
        Bench::SPlusTreeNeon => {
            let values = [1, 3, 5, 7, 9];
            let layout = SPlusTreeLayout::new(&values);
            assert_eq!(s_plus_tree_search_neon(&layout, 5), Some(2));
            assert_eq!(s_plus_tree_search_neon(&layout, 6), None);
        }
        Bench::MatmulBaseline => verify_matmul_variant(matmul_baseline),
        Bench::MatmulTransposed => verify_matmul_variant(matmul_transposed),
        Bench::MatmulIkj => verify_matmul_variant(matmul_ikj),
        Bench::MatmulRegister2x2 => verify_matmul_variant(matmul_register_blocked_2x2),
        Bench::MatmulBlocked => verify_matmul_variant(matmul_blocked),
        #[cfg(target_arch = "aarch64")]
        Bench::MatmulNeonBlocked => verify_matmul_variant(matmul_neon_blocked),
        Bench::GcdScalar => {
            let g = gcd_scalar(21, 14);
            assert_eq!(g, 7);
        }
        Bench::GcdBinary => {
            let g = gcd_binary(21, 14);
            assert_eq!(g, 7);
        }
        Bench::PrefixSumScalar => {
            let input = [1, 2, 3, 4, 5];
            let expected = vec![1, 3, 6, 10, 15];
            let output = prefix_sum_scalar(&input);
            assert_eq!(output, expected);
        }
        Bench::PrefixSumScalarInPlace => {
            let mut input = [1, 2, 3, 4, 5];
            let expected = [1, 3, 6, 10, 15];
            prefix_sum_scalar_in_place(&mut input);
            assert_eq!(input, expected);
        }
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeon => {
            let mut input = [1, 2, 3, 4, 5];
            let expected = [1, 3, 6, 10, 15];
            prefix_sum(&mut input);
            assert_eq!(input, expected);
        }
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeonBlocked => {
            let mut input = [1, 2, 3, 4, 5];
            let expected = [1, 3, 6, 10, 15];
            prefix_sum_blocked(&mut input);
            assert_eq!(input, expected);
        }
        #[cfg(target_arch = "aarch64")]
        Bench::PrefixSumNeonInterleaved => {
            let mut input = [1, 2, 3, 4, 5];
            let expected = [1, 3, 6, 10, 15];
            prefix_sum_interleaved(&mut input);
            assert_eq!(input, expected);
        }
    }
}

fn bench_argmin(config: Config, func: fn(&[i32]) -> Option<usize>) {
    let input = make_i32_input(config.len, config.seed);
    let mut acc = 0usize;
    for _ in 0..config.iters {
        if let Some(idx) = func(black_box(&input)) {
            acc ^= idx;
        }
    }
    black_box(acc);
}

fn bench_binary_search(config: Config, func: BinarySearchFn) {
    if config.len == 0 {
        return;
    }

    let values = make_sorted_values(config.len);
    let queries = make_search_queries(&values, config.seed);
    let mut acc = 0usize;
    for _ in 0..config.iters {
        let haystack = black_box(values.as_slice());
        for &q in &queries {
            if let Some(idx) = func(haystack, black_box(q)) {
                acc ^= idx;
            }
        }
    }
    black_box(acc);
}

fn bench_binary_search_eytzinger(config: Config, func: BinarySearchEytzingerFn) {
    if config.len == 0 {
        return;
    }

    let values = make_sorted_values(config.len);
    let layout = EytzingerLayout::new(&values);
    let queries = make_search_queries(&values, config.seed);
    let mut acc = 0usize;
    for _ in 0..config.iters {
        let layout = black_box(&layout);
        for &q in &queries {
            if let Some(idx) = func(layout, black_box(q)) {
                acc ^= idx;
            }
        }
    }
    black_box(acc);
}

fn bench_s_tree(config: Config, func: STreeFn) {
    if config.len == 0 {
        return;
    }

    let values = make_sorted_values(config.len);
    let layout = STreeLayout::new(&values);
    let queries = make_search_queries(&values, config.seed);
    let mut acc = 0usize;
    for _ in 0..config.iters {
        let layout = black_box(&layout);
        for &q in &queries {
            if let Some(idx) = func(layout, black_box(q)) {
                acc ^= idx;
            }
        }
    }
    black_box(acc);
}

fn bench_s_plus_tree(config: Config, func: SPlusTreeFn) {
    if config.len == 0 {
        return;
    }

    let values = make_sorted_values(config.len);
    let layout = SPlusTreeLayout::new(&values);
    let queries = make_search_queries(&values, config.seed);
    let mut acc = 0usize;
    for _ in 0..config.iters {
        let layout = black_box(&layout);
        for &q in &queries {
            if let Some(idx) = func(layout, black_box(q)) {
                acc ^= idx;
            }
        }
    }
    black_box(acc);
}

fn bench_matmul(config: Config, func: MatmulFn) {
    if config.len == 0 {
        return;
    }

    let n = config.len;
    let a = make_f32_matrix(n, config.seed);
    let b = make_f32_matrix(n, config.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut c = vec![0.0f32; n * n];
    let sample = (n / 2) * n + (n / 2);

    let mut acc = 0.0f64;
    for _ in 0..config.iters {
        func(black_box(&a), black_box(&b), black_box(&mut c), n);
        acc += c[sample] as f64;
    }
    black_box(acc);
}

fn bench_gcd_scalar(config: Config) {
    let pairs = make_pairs(config.len, config.seed);
    let mut acc = 0i64;
    for _ in 0..config.iters {
        for &(a, b) in &pairs {
            acc ^= gcd_scalar(black_box(a), black_box(b));
        }
    }
    black_box(acc);
}

fn bench_gcd_binary(config: Config) {
    let pairs = make_pairs(config.len, config.seed);
    let mut acc = 0i64;
    for _ in 0..config.iters {
        for &(a, b) in &pairs {
            acc ^= gcd_binary(black_box(a), black_box(b));
        }
    }
    black_box(acc);
}

fn bench_prefix_sum_scalar(config: Config) {
    let input = make_i32_input(config.len, config.seed);
    let mut acc = 0i64;
    for _ in 0..config.iters {
        let output = prefix_sum_scalar(black_box(input.as_slice()));
        acc ^= output.last().copied().unwrap_or(0) as i64;
        black_box(&output);
    }
    black_box(acc);
}

fn bench_prefix_sum_scalar_in_place(config: Config) {
    let base = make_i32_input(config.len, config.seed);
    let mut values = base.clone();
    let mut acc = 0i64;
    for _ in 0..config.iters {
        if config.reset {
            values.copy_from_slice(&base);
        }
        prefix_sum_scalar_in_place(black_box(values.as_mut_slice()));
        acc ^= values.last().copied().unwrap_or(0) as i64;
    }
    black_box(acc);
}

#[cfg(target_arch = "aarch64")]
fn bench_prefix_sum_neon(config: Config) {
    let base = make_i32_input(config.len, config.seed);
    let mut values = base.clone();
    let mut acc = 0i64;
    for _ in 0..config.iters {
        if config.reset {
            values.copy_from_slice(&base);
        }
        prefix_sum(black_box(values.as_mut_slice()));
        acc ^= values.last().copied().unwrap_or(0) as i64;
    }
    black_box(acc);
}

#[cfg(target_arch = "aarch64")]
fn bench_prefix_sum_neon_blocked(config: Config) {
    let base = make_i32_input(config.len, config.seed);
    let mut values = base.clone();
    let mut acc = 0i64;
    for _ in 0..config.iters {
        if config.reset {
            values.copy_from_slice(&base);
        }
        prefix_sum_blocked(black_box(values.as_mut_slice()));
        acc ^= values.last().copied().unwrap_or(0) as i64;
    }
    black_box(acc);
}

#[cfg(target_arch = "aarch64")]
fn bench_prefix_sum_neon_interleaved(config: Config) {
    let base = make_i32_input(config.len, config.seed);
    let mut values = base.clone();
    let mut acc = 0i64;
    for _ in 0..config.iters {
        if config.reset {
            values.copy_from_slice(&base);
        }
        prefix_sum_interleaved(black_box(values.as_mut_slice()));
        acc ^= values.last().copied().unwrap_or(0) as i64;
    }
    black_box(acc);
}
