//! Inclusive prefix sum (scan) algorithms.
//!
//! Computes running totals: `[a, b, c, d]` → `[a, a+b, a+b+c, a+b+c+d]`
//!
//! # The Parallelization Challenge
//!
//! Prefix sum has an inherent data dependency: each output depends on all previous
//! inputs. This seems to prevent parallelization, but SIMD can still help through
//! a clever two-phase approach:
//!
//! 1. **Local prefix**: Compute prefix sums within each SIMD lane independently
//! 2. **Accumulate**: Propagate the final value of each chunk to subsequent chunks
//!
//! # Strategies
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`prefix_sum_scalar`] | Simple sequential scan |
//! | [`prefix_sum`] | Best available (NEON on aarch64) |
//! | [`prefix_sum_blocked`] | Block-wise for cache efficiency |
//! | [`prefix_sum_interleaved`] | Overlaps compute with memory access |
//!
//! # References
//!
//! - [Prefix sum chapter](https://en.algorithmica.org/hpc/algorithms/prefix/)

/// Sequential prefix sum returning a new vector.
pub fn prefix_sum_scalar(input: &[i32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());
    let mut sum = 0i32;
    for &value in input {
        sum = sum.wrapping_add(value);
        output.push(sum);
    }
    output
}

/// Sequential prefix sum in-place.
pub fn prefix_sum_scalar_in_place(values: &mut [i32]) {
    let mut sum = 0i32;
    for value in values {
        sum = sum.wrapping_add(*value);
        *value = sum;
    }
}

/// ARM NEON SIMD implementations for aarch64.
#[cfg(target_arch = "aarch64")]
mod neon_scan {
    use std::arch::aarch64::*;
    use std::arch::asm;

    /// Elements per block—tuned for L1 cache (4096 × 4 bytes = 16KB, fits in 32KB L1).
    const BLOCK_ELEMS: usize = 4096;
    /// Elements to process before accumulating—balances ILP vs memory latency.
    const INTERLEAVE_ELEMS: usize = 64;
    /// Prefetch distance in elements—hides ~100 cycle memory latency.
    const PREFETCH_ELEMS: usize = 64;
    /// NEON vector width for i32.
    const LANES: usize = 4;

    /// Default prefix sum using the interleaved strategy.
    pub fn prefix_sum(a: &mut [i32]) {
        unsafe { prefix_neon_interleaved(a) }
    }

    /// Block-wise prefix sum for cache efficiency.
    pub fn prefix_sum_blocked(a: &mut [i32]) {
        unsafe { prefix_neon_blocked(a) }
    }

    /// Interleaved prefix sum that overlaps compute with memory access.
    pub fn prefix_sum_interleaved(a: &mut [i32]) {
        unsafe { prefix_neon_interleaved(a) }
    }

    /// Simple blocked implementation for cache efficiency.
    #[target_feature(enable = "neon")]
    unsafe fn prefix_neon_blocked(a: &mut [i32]) {
        unsafe {
            let n = a.len();
            let mut s = vdupq_n_s32(0);

            // Process in L1-sized blocks to minimize cache misses
            for base in (0..n).step_by(BLOCK_ELEMS) {
                let len = (n - base).min(BLOCK_ELEMS);
                s = local_prefix(a.as_mut_ptr().add(base), len, s);
            }
        }
    }

    /// Interleaved implementation that overlaps Phase 1 and Phase 2.
    ///
    /// The key optimization: while we're doing `prefix_lane` on chunk N,
    /// we simultaneously do `accumulate` on chunk N-INTERLEAVE. This
    /// overlaps compute with memory access, hiding latency.
    #[target_feature(enable = "neon")]
    unsafe fn prefix_neon_interleaved(a: &mut [i32]) {
        unsafe {
            let n = a.len();
            let vec_len = n & !(LANES - 1);

            // Fallback to scalar for tiny arrays
            if vec_len == 0 {
                let mut carry = 0i32;
                let p = a.as_mut_ptr();
                let mut i = 0;
                while i < n {
                    carry = carry.wrapping_add(*p.add(i));
                    *p.add(i) = carry;
                    i += 1;
                }
                return;
            }

            let p = a.as_mut_ptr();
            let mut s = vdupq_n_s32(0);

            if vec_len <= INTERLEAVE_ELEMS {
                s = local_prefix(p, vec_len, s);
            } else {
                // Prime the pipeline: do prefix_lane on first INTERLEAVE_ELEMS
                let mut i = 0;
                while i < INTERLEAVE_ELEMS {
                    prefix_lane(p.add(i));
                    i += LANES;
                }

                // Steady state: prefix_lane on current chunk while
                // accumulating the chunk INTERLEAVE_ELEMS behind
                let end = p.add(vec_len);
                let mut j = INTERLEAVE_ELEMS;
                while j < vec_len {
                    prefix_lane(p.add(j));
                    s = accumulate_prefetch(p.add(j - INTERLEAVE_ELEMS), s, end);
                    j += LANES;
                }

                // Drain: finish accumulating the last INTERLEAVE_ELEMS
                let mut k = vec_len - INTERLEAVE_ELEMS;
                while k < vec_len {
                    s = accumulate_prefetch(p.add(k), s, end);
                    k += LANES;
                }
            }

            // Scalar tail for remaining elements
            let mut carry = vgetq_lane_s32(s, 0);
            let mut idx = vec_len;
            while idx < n {
                carry = carry.wrapping_add(*p.add(idx));
                *p.add(idx) = carry;
                idx += 1;
            }
        }
    }

    /// Issues a prefetch hint to bring data into L1 cache.
    ///
    /// `prfm pldl1keep` = "prefetch for load, L1, keep in cache"
    /// This instruction is a hint—it doesn't stall if the address is invalid.
    #[inline]
    unsafe fn prefetch_read(p: *const i32) {
        unsafe {
            asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack));
        }
    }

    /// Two-phase prefix sum within a block.
    ///
    /// Phase 1: `prefix_lane` computes local prefix sums within each 4-element chunk
    /// Phase 2: `accumulate` propagates the running total across chunks
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn local_prefix(p: *mut i32, len: usize, mut s: int32x4_t) -> int32x4_t {
        unsafe {
            // Phase 1: local prefix within each vector
            let mut i = 0;
            while i + 4 <= len {
                prefix_lane(p.add(i));
                i += 4;
            }

            // Phase 2: propagate carry across vectors
            let mut j = 0;
            while j + 4 <= len {
                s = accumulate(p.add(j), s);
                j += 4;
            }

            // Scalar tail
            let mut carry = vgetq_lane_s32(s, 0);
            while j < len {
                carry = carry.wrapping_add(*p.add(j));
                *p.add(j) = carry;
                j += 1;
            }

            vdupq_n_s32(carry)
        }
    }

    /// Computes prefix sum within a single 4-element NEON vector.
    ///
    /// Uses shift-and-add pattern to compute prefix in O(log lanes) steps:
    ///
    /// ```text
    /// Input:  [a,    b,    c,    d   ]
    /// Step 1: [a,  a+b,  b+c,  c+d  ]  (shift by 1, add)
    /// Step 2: [a,  a+b, a+b+c, a+b+c+d] (shift by 2, add)
    /// ```
    ///
    /// `vextq_s32(zero, x, 3)` shifts x right by 1 lane (inserting zero on left).
    /// For 4 lanes, we need log2(4) = 2 steps.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn prefix_lane(p: *mut i32) {
        unsafe {
            let zero = vdupq_n_s32(0);
            let mut x = vld1q_s32(p);

            // Shift right by 1 lane: [0, a, b, c], then add to get [a, a+b, b+c, c+d]
            let t1 = vextq_s32(zero, x, 3);
            x = vaddq_s32(x, t1);

            // Shift right by 2 lanes: [0, 0, a, a+b], then add to complete prefix
            let t2 = vextq_s32(zero, x, 2);
            x = vaddq_s32(x, t2);

            vst1q_s32(p, x);
        }
    }

    /// Adds running sum to a vector and updates the carry.
    ///
    /// After `prefix_lane`, each vector contains its local prefix sum.
    /// This function adds the running total `s` (broadcast to all lanes)
    /// and extracts lane 3 (the local total) to update the carry.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn accumulate(p: *mut i32, s: int32x4_t) -> int32x4_t {
        unsafe {
            let mut x = vld1q_s32(p);
            // Lane 3 contains sum of all 4 elements (from prefix_lane)
            let local_total = vgetq_lane_s32(x, 3);

            // Add running total to all lanes
            x = vaddq_s32(s, x);
            vst1q_s32(p, x);

            // Update carry: s += local_total (broadcast to all lanes, but we only use lane 0)
            vaddq_s32(s, vdupq_n_s32(local_total))
        }
    }

    /// Same as `accumulate` but also prefetches upcoming data.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn accumulate_prefetch(p: *mut i32, s: int32x4_t, end: *const i32) -> int32x4_t {
        unsafe {
            let prefetch_ptr = p.add(PREFETCH_ELEMS);
            if (prefetch_ptr as usize) < (end as usize) {
                prefetch_read(prefetch_ptr);
            }

            let mut x = vld1q_s32(p);
            let local_total = vgetq_lane_s32(x, 3);

            x = vaddq_s32(s, x);
            vst1q_s32(p, x);

            vaddq_s32(s, vdupq_n_s32(local_total))
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon_scan::{prefix_sum, prefix_sum_blocked, prefix_sum_interleaved};

#[cfg(not(target_arch = "aarch64"))]
pub fn prefix_sum(_: &mut [i32]) {
    panic!("This implementation requires aarch64");
}

#[cfg(not(target_arch = "aarch64"))]
pub fn prefix_sum_blocked(_: &mut [i32]) {
    panic!("This implementation requires aarch64");
}

#[cfg(not(target_arch = "aarch64"))]
pub fn prefix_sum_interleaved(_: &mut [i32]) {
    panic!("This implementation requires aarch64");
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestRunner;

    fn assert_prefix_sum_property(input: &[i32], output: &[i32]) -> Result<(), TestCaseError> {
        prop_assert_eq!(output.len(), input.len());
        if input.is_empty() {
            return Ok(());
        }

        prop_assert_eq!(output[0], input[0]);
        for i in 1..input.len() {
            let delta = output[i].wrapping_sub(output[i - 1]);
            prop_assert_eq!(delta, input[i]);
        }

        Ok(())
    }

    fn run_prefix_sum_cases(cases: u32, max_len: usize) {
        let mut runner = TestRunner::new(ProptestConfig {
            cases,
            ..ProptestConfig::default()
        });
        let strat = proptest::collection::vec(any::<i32>(), 0..=max_len);

        runner
            .run(&strat, |input: Vec<i32>| {
                let scalar = prefix_sum_scalar(&input);
                assert_prefix_sum_property(&input, &scalar)?;

                let mut inplace = input.clone();
                prefix_sum_scalar_in_place(&mut inplace);
                prop_assert_eq!(&scalar, &inplace);

                #[cfg(target_arch = "aarch64")]
                {
                    let mut neon = input.clone();
                    prefix_sum(&mut neon);
                    prop_assert_eq!(&scalar, &neon);

                    let mut blocked = input.clone();
                    prefix_sum_blocked(&mut blocked);
                    prop_assert_eq!(&scalar, &blocked);

                    let mut interleaved = input.clone();
                    prefix_sum_interleaved(&mut interleaved);
                    prop_assert_eq!(&scalar, &interleaved);
                }

                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prop_prefix_sum_fast() {
        run_prefix_sum_cases(256, 2048);
    }

    #[test]
    #[ignore]
    fn prop_prefix_sum_deep() {
        run_prefix_sum_cases(4096, 16384);
    }

    #[test]
    fn prefix_sum_scalar_basic() {
        let input = [1, 2, 3, 4, 5];
        let expected = vec![1, 3, 6, 10, 15];
        let result = prefix_sum_scalar(&input);
        assert_eq!(result, expected);
    }

    #[test]
    fn prefix_sum_scalar_empty() {
        let input: [i32; 0] = [];
        let result = prefix_sum_scalar(&input);
        assert!(result.is_empty());
    }

    #[test]
    fn prefix_sum_scalar_in_place_basic() {
        let mut input = [1, 2, 3, 4, 5];
        let expected = [1, 3, 6, 10, 15];
        prefix_sum_scalar_in_place(&mut input);
        assert_eq!(input, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prefix_sum_neon_basic() {
        let mut input = [1, 2, 3, 4, 5];
        let expected = [1, 3, 6, 10, 15];
        super::prefix_sum(&mut input);
        assert_eq!(input, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prefix_sum_neon_blocked_basic() {
        let mut input = [1, 2, 3, 4, 5];
        let expected = [1, 3, 6, 10, 15];
        super::prefix_sum_blocked(&mut input);
        assert_eq!(input, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prefix_sum_neon_interleaved_basic() {
        let mut input = [1, 2, 3, 4, 5];
        let expected = [1, 3, 6, 10, 15];
        super::prefix_sum_interleaved(&mut input);
        assert_eq!(input, expected);
    }
}
