pub fn prefix_sum_scalar(input: &[i32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());
    let mut sum = 0i32;
    for &value in input {
        sum = sum.wrapping_add(value);
        output.push(sum);
    }
    output
}

pub fn prefix_sum_scalar_in_place(values: &mut [i32]) {
    let mut sum = 0i32;
    for value in values {
        sum = sum.wrapping_add(*value);
        *value = sum;
    }
}

#[cfg(target_arch = "aarch64")]
mod neon_scan {
    use std::arch::aarch64::*;
    use std::arch::asm;

    const BLOCK_ELEMS: usize = 4096; // elements, not bytes
    const INTERLEAVE_ELEMS: usize = 64;
    const PREFETCH_ELEMS: usize = 64;
    const LANES: usize = 4;

    pub fn prefix_sum(a: &mut [i32]) {
        unsafe { prefix_neon_interleaved(a) }
    }

    pub fn prefix_sum_blocked(a: &mut [i32]) {
        unsafe { prefix_neon_blocked(a) }
    }

    pub fn prefix_sum_interleaved(a: &mut [i32]) {
        unsafe { prefix_neon_interleaved(a) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn prefix_neon_blocked(a: &mut [i32]) {
        unsafe {
            let n = a.len();
            let mut s = vdupq_n_s32(0);

            for base in (0..n).step_by(BLOCK_ELEMS) {
                let len = (n - base).min(BLOCK_ELEMS);
                s = local_prefix(a.as_mut_ptr().add(base), len, s);
            }
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn prefix_neon_interleaved(a: &mut [i32]) {
        unsafe {
            let n = a.len();
            let vec_len = n & !(LANES - 1);

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
                let mut i = 0;
                while i < INTERLEAVE_ELEMS {
                    prefix_lane(p.add(i));
                    i += LANES;
                }

                let end = p.add(vec_len);
                let mut j = INTERLEAVE_ELEMS;
                while j < vec_len {
                    prefix_lane(p.add(j));
                    s = accumulate_prefetch(p.add(j - INTERLEAVE_ELEMS), s, end);
                    j += LANES;
                }

                let mut k = vec_len - INTERLEAVE_ELEMS;
                while k < vec_len {
                    s = accumulate_prefetch(p.add(k), s, end);
                    k += LANES;
                }
            }

            let mut carry = vgetq_lane_s32(s, 0);
            let mut idx = vec_len;
            while idx < n {
                carry = carry.wrapping_add(*p.add(idx));
                *p.add(idx) = carry;
                idx += 1;
            }
        }
    }

    #[inline]
    unsafe fn prefetch_read(p: *const i32) {
        unsafe {
            asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack));
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn local_prefix(p: *mut i32, len: usize, mut s: int32x4_t) -> int32x4_t {
        unsafe {
            let mut i = 0;
            while i + 4 <= len {
                prefix_lane(p.add(i));
                i += 4;
            }

            let mut j = 0;
            while j + 4 <= len {
                s = accumulate(p.add(j), s);
                j += 4;
            }

            let mut carry = vgetq_lane_s32(s, 0);
            while j < len {
                carry = carry.wrapping_add(*p.add(j));
                *p.add(j) = carry;
                j += 1;
            }

            vdupq_n_s32(carry)
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn prefix_lane(p: *mut i32) {
        unsafe {
            let zero = vdupq_n_s32(0);
            let mut x = vld1q_s32(p);

            let t1 = vextq_s32(zero, x, 3);
            x = vaddq_s32(x, t1);

            let t2 = vextq_s32(zero, x, 2);
            x = vaddq_s32(x, t2);

            vst1q_s32(p, x);
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn accumulate(p: *mut i32, s: int32x4_t) -> int32x4_t {
        unsafe {
            let mut x = vld1q_s32(p);
            let local_total = vgetq_lane_s32(x, 3);

            x = vaddq_s32(s, x);
            vst1q_s32(p, x);

            vaddq_s32(s, vdupq_n_s32(local_total))
        }
    }

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
