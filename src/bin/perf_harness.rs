use std::env;
use std::hint::black_box;
use std::process;
use std::time::Instant;

use hpc_algorithms::{
    BLOCK_SIZE, argmin_blocked, argmin_branchless, argmin_min_then_find, argmin_scalar,
    argmin_simd_filtered, argmin_std, argmin_vector_indices, gcd_binary, gcd_scalar,
    prefix_sum_scalar, prefix_sum_scalar_in_place,
};

#[cfg(target_arch = "aarch64")]
use hpc_algorithms::{prefix_sum, prefix_sum_blocked, prefix_sum_interleaved};

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
            | Bench::ArgminSimdFiltered => 1_000_000,
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
    println!(
        "bench={} len={} iters={} elapsed_s={:.6} work_items={} unit={} ns_per_item={:.3} throughput={} bytes={} byte_throughput={}",
        bench.name(),
        config.len,
        config.iters,
        elapsed_s,
        stats.work_items,
        stats.unit,
        ns_per_item,
        format_rate(items_per_s, stats.unit),
        stats.bytes,
        format_rate(bytes_per_s, "B"),
    );
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
