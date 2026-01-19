use std::cell::RefCell;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use hpc_algorithms::prefix_sum_scalar_in_place;

#[cfg(target_arch = "aarch64")]
use hpc_algorithms::{prefix_sum_blocked, prefix_sum_interleaved};

const INPUT_SIZES: &[(&str, usize)] = &[
    ("l1_8k", 8 * 1024),
    ("l2_64k", 64 * 1024),
    ("l3_1m", 1024 * 1024),
    ("mem_16m", 16 * 1024 * 1024),
];

/// Reset buffer to sequential integers (0, 1, 2, ...) without allocation.
#[inline]
fn reset_sequential(data: &mut [i32]) {
    for (i, slot) in data.iter_mut().enumerate() {
        *slot = i as i32;
    }
}

fn bench_prefix_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_sum");
    for &(label, len) in INPUT_SIZES {
        group.throughput(Throughput::Elements(len as u64));

        // Pre-allocate once, reuse via RefCell to satisfy borrow checker.
        let data = RefCell::new(vec![0i32; len]);

        group.bench_function(BenchmarkId::new("scalar", label), |b| {
            b.iter_batched(
                || reset_sequential(&mut data.borrow_mut()),
                |()| prefix_sum_scalar_in_place(black_box(&mut data.borrow_mut())),
                BatchSize::LargeInput,
            )
        });

        #[cfg(target_arch = "aarch64")]
        {
            let data = RefCell::new(vec![0i32; len]);
            group.bench_function(BenchmarkId::new("neon_blocked", label), |b| {
                b.iter_batched(
                    || reset_sequential(&mut data.borrow_mut()),
                    |()| prefix_sum_blocked(black_box(&mut data.borrow_mut())),
                    BatchSize::LargeInput,
                )
            });

            let data = RefCell::new(vec![0i32; len]);
            group.bench_function(BenchmarkId::new("neon_interleaved", label), |b| {
                b.iter_batched(
                    || reset_sequential(&mut data.borrow_mut()),
                    |()| prefix_sum_interleaved(black_box(&mut data.borrow_mut())),
                    BatchSize::LargeInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_prefix_sum);

criterion_main!(benches);
