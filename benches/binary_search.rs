use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hpc_algorithms::{
    EytzingerLayout, binary_search_branchless, binary_search_branchless_prefetch,
    binary_search_eytzinger, binary_search_eytzinger_prefetch, binary_search_std,
};

type SearchFn = fn(&[i32], i32) -> Option<usize>;
type EytzingerFn = fn(&EytzingerLayout, i32) -> Option<usize>;

const INPUT_SIZES: &[(&str, usize)] = &[
    ("l1_4k", 4 * 1024),
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

fn make_sorted_values(len: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push((i as i32) * 2);
    }
    out
}

fn make_queries_hit(values: &[i32], seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(values.len());
    for _ in 0..values.len() {
        let idx = (next_u64(&mut state) as usize) % values.len();
        out.push(values[idx]);
    }
    out
}

fn make_queries_miss(values: &[i32], seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(values.len());
    for _ in 0..values.len() {
        let idx = (next_u64(&mut state) as usize) % values.len();
        out.push(values[idx].wrapping_add(1));
    }
    out
}

fn bench_variant(c: &mut Criterion, name: &str, func: SearchFn) {
    let mut group = c.benchmark_group(name);
    for &(label, len) in INPUT_SIZES {
        let values = make_sorted_values(len);
        let queries_hit = make_queries_hit(&values, 0xC0FF_EE42_1234_5678u64 ^ len as u64);
        let queries_miss = make_queries_miss(&values, 0xBADC_0FFE_EE11_D00Du64 ^ len as u64);

        group.throughput(Throughput::Elements(queries_hit.len() as u64));
        group.bench_function(BenchmarkId::new("hit", label), |bench| {
            bench.iter(|| {
                let haystack = black_box(&values);
                let mut acc = 0usize;
                for &q in &queries_hit {
                    if let Some(idx) = func(haystack, black_box(q)) {
                        acc ^= idx;
                    }
                }
                black_box(acc);
            });
        });

        group.bench_function(BenchmarkId::new("miss", label), |bench| {
            bench.iter(|| {
                let haystack = black_box(&values);
                let mut acc = 0usize;
                for &q in &queries_miss {
                    if let Some(idx) = func(haystack, black_box(q)) {
                        acc ^= idx;
                    }
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

fn bench_variant_eytzinger(c: &mut Criterion, name: &str, func: EytzingerFn) {
    let mut group = c.benchmark_group(name);
    for &(label, len) in INPUT_SIZES {
        let values = make_sorted_values(len);
        let layout = EytzingerLayout::new(&values);
        let queries_hit = make_queries_hit(&values, 0xC0FF_EE42_1234_5678u64 ^ len as u64);
        let queries_miss = make_queries_miss(&values, 0xBADC_0FFE_EE11_D00Du64 ^ len as u64);

        group.throughput(Throughput::Elements(queries_hit.len() as u64));
        group.bench_function(BenchmarkId::new("hit", label), |bench| {
            bench.iter(|| {
                let layout = black_box(&layout);
                let mut acc = 0usize;
                for &q in &queries_hit {
                    if let Some(idx) = func(layout, black_box(q)) {
                        acc ^= idx;
                    }
                }
                black_box(acc);
            });
        });

        group.bench_function(BenchmarkId::new("miss", label), |bench| {
            bench.iter(|| {
                let layout = black_box(&layout);
                let mut acc = 0usize;
                for &q in &queries_miss {
                    if let Some(idx) = func(layout, black_box(q)) {
                        acc ^= idx;
                    }
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

fn bench_binary_search(c: &mut Criterion) {
    bench_variant(c, "binary_search_std", binary_search_std);
    bench_variant(c, "binary_search_branchless", binary_search_branchless);
    bench_variant(
        c,
        "binary_search_branchless_prefetch",
        binary_search_branchless_prefetch,
    );
    bench_variant_eytzinger(c, "binary_search_eytzinger", binary_search_eytzinger);
    bench_variant_eytzinger(
        c,
        "binary_search_eytzinger_prefetch",
        binary_search_eytzinger_prefetch,
    );
}

criterion_group!(benches, bench_binary_search);
criterion_main!(benches);
