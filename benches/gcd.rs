use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hpc_algorithms::{gcd_binary, gcd_scalar};

const INPUT_SIZES: &[(&str, usize)] = &[("1k", 1_000), ("10k", 10_000), ("100k", 100_000)];

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

fn bench_gcd_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("gcd_scalar");
    for &(label, len) in INPUT_SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let pairs = make_pairs(len, 0xD00D_FEED_CAFE_BEEFu64 ^ len as u64);
        group.bench_function(BenchmarkId::new("uniform_32b", label), |b| {
            b.iter(|| {
                let mut acc = 0i64;
                for &(a, b0) in pairs.iter() {
                    acc ^= gcd_scalar(black_box(a), black_box(b0));
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

fn bench_gcd_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("gcd_binary");
    for &(label, len) in INPUT_SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let pairs = make_pairs(len, 0xD00D_FEED_CAFE_BEEFu64 ^ len as u64);
        group.bench_function(BenchmarkId::new("uniform_32b", label), |b| {
            b.iter(|| {
                let mut acc = 0i64;
                for &(a, b0) in pairs.iter() {
                    acc ^= gcd_binary(black_box(a), black_box(b0));
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gcd_scalar, bench_gcd_binary);
criterion_main!(benches);
