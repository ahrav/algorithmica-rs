use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(target_arch = "aarch64")]
use hpc_algorithms::matmul_neon_blocked;
use hpc_algorithms::{
    matmul_baseline, matmul_blocked, matmul_ikj, matmul_register_blocked_2x2, matmul_transposed,
};

type MatmulFn = fn(&[f32], &[f32], &mut [f32], usize);

const INPUT_SIZES: &[(&str, usize)] = &[("n64", 64), ("n128", 128), ("n256", 256)];

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn make_matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n * n);
    for _ in 0..n * n {
        let v = (next_u64(&mut state) & 0xFF) as f32;
        out.push(v / 255.0);
    }
    out
}

fn bench_variant(c: &mut Criterion, name: &str, func: MatmulFn) {
    let mut group = c.benchmark_group(name);
    for &(label, n) in INPUT_SIZES {
        group.throughput(Throughput::Elements((n as u64).pow(3)));

        let a = make_matrix(n, 0xC0FF_EE42_1234_5678u64 ^ n as u64);
        let b = make_matrix(n, 0xBADC_0FFE_EE11_D00Du64 ^ (n as u64).rotate_left(17));
        let mut out = vec![0.0f32; n * n];
        let sample = (n / 2) * n + (n / 2);

        group.bench_function(BenchmarkId::new("random", label), |bench| {
            bench.iter(|| {
                func(black_box(&a), black_box(&b), black_box(&mut out), n);
                black_box(out[sample]);
            });
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    bench_variant(c, "matmul_baseline", matmul_baseline);
    bench_variant(c, "matmul_transposed", matmul_transposed);
    bench_variant(c, "matmul_ikj", matmul_ikj);
    bench_variant(c, "matmul_register_2x2", matmul_register_blocked_2x2);
    bench_variant(c, "matmul_blocked", matmul_blocked);
    #[cfg(target_arch = "aarch64")]
    bench_variant(c, "matmul_neon_blocked", matmul_neon_blocked);
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
