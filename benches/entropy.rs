use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(target_arch = "aarch64")]
use hpc_algorithms::entropy_interleaved_neon;
use hpc_algorithms::{entropy_interleaved, shannon_entropy};

const INPUT_SIZES: &[(&str, usize)] = &[
    ("l1_8k", 8 * 1024),
    ("l2_64k", 64 * 1024),
    ("l3_1m", 1024 * 1024),
];

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn make_random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(next_u64(&mut state) as u8);
    }
    out
}

fn make_two_symbol_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let bit = (next_u64(&mut state) & 1) as u8;
        out.push(if bit == 0 { 0 } else { 255 });
    }
    out
}

fn bench_variant(c: &mut Criterion, name: &str, func: fn(&[u8]) -> f64) {
    let mut group = c.benchmark_group(name);
    for &(label, len) in INPUT_SIZES {
        group.throughput(Throughput::Bytes(len as u64));

        let random = make_random_bytes(len, 0x5EED_F00D_CAFE_BAAEu64 ^ len as u64);
        group.bench_function(BenchmarkId::new("random", label), |b| {
            b.iter(|| black_box(func(black_box(&random))));
        });

        let two_symbols = make_two_symbol_bytes(len, 0xDADA_C0DE_F00D_BEEFu64 ^ len as u64);
        group.bench_function(BenchmarkId::new("two_symbols", label), |b| {
            b.iter(|| black_box(func(black_box(&two_symbols))));
        });
    }
    group.finish();
}

fn bench_entropy(c: &mut Criterion) {
    bench_variant(c, "shannon_entropy", shannon_entropy);
    bench_variant(c, "entropy_interleaved", entropy_interleaved);
    #[cfg(target_arch = "aarch64")]
    bench_variant(c, "entropy_interleaved_neon", entropy_interleaved_neon);
}

criterion_group!(benches, bench_entropy);
criterion_main!(benches);
