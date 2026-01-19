use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hpc_algorithms::{
    argmin_blocked, argmin_branchless, argmin_min_then_find, argmin_scalar, argmin_simd_filtered,
    argmin_std, argmin_vector_indices, simd_available,
};

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

fn make_random(len: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(next_u64(&mut state) as i32);
    }
    out
}

fn make_decreasing(len: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push((len - i) as i32);
    }
    out
}

fn bench_variant(c: &mut Criterion, name: &str, func: fn(&[i32]) -> Option<usize>) {
    let mut group = c.benchmark_group(name);
    for &(label, len) in INPUT_SIZES {
        group.throughput(Throughput::Elements(len as u64));

        let random = make_random(len, 0xC0FF_EE42_1234_5678u64 ^ len as u64);
        group.bench_function(BenchmarkId::new("random", label), |b| {
            b.iter(|| black_box(func(black_box(&random))));
        });

        let decreasing = make_decreasing(len);
        group.bench_function(BenchmarkId::new("decreasing", label), |b| {
            b.iter(|| black_box(func(black_box(&decreasing))));
        });
    }
    group.finish();
}

fn bench_argmin(c: &mut Criterion) {
    bench_variant(c, "argmin_std", argmin_std);
    bench_variant(c, "argmin_scalar", argmin_scalar);
    bench_variant(c, "argmin_branchless", argmin_branchless);
    bench_variant(c, "argmin_min_then_find", argmin_min_then_find);
    bench_variant(c, "argmin_blocked", argmin_blocked);

    if simd_available() {
        bench_variant(c, "argmin_simd_indices", argmin_vector_indices);
        bench_variant(c, "argmin_simd_filtered", argmin_simd_filtered);
    }
}

criterion_group!(benches, bench_argmin);
criterion_main!(benches);
