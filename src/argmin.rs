//! Finding the index of the minimum element in an array.
//!
//! This module demonstrates several optimization strategies for argmin, from simple
//! scalar code to sophisticated SIMD approaches. The key insight is that the naive
//! approach suffers from ~50% branch misprediction on random data.
//!
//! # Strategies
//!
//! | Function | Strategy | Best For |
//! |----------|----------|----------|
//! | [`argmin_scalar`] | Simple loop with branch | Small arrays, sorted data |
//! | [`argmin_branchless`] | Conditional moves | Random data (avoids mispredictions) |
//! | [`argmin_min_then_find`] | Two-pass: find min, then index | Large arrays |
//! | [`argmin_blocked`] | Block-wise min then precise search | Very large arrays |
//! | [`argmin_vector_indices`] | SIMD with parallel index tracking | Large arrays with AVX2/AVX-512/NEON |
//! | [`argmin_simd_filtered`] | Speculative block filtering | Large arrays with AVX2/AVX-512/NEON |
//!
//! # Why Branches Hurt
//!
//! Modern CPUs predict branches speculatively. When searching for a minimum in random
//! data, the condition `v < min` is true with decreasing probability (roughly `1/i` at
//! position `i`), averaging to ~50% early on. Each misprediction costs 10-20 cycles.
//!
//! # References
//!
//! - [Argmin chapter](https://en.algorithmica.org/hpc/algorithms/argmin/)

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

/// Block size for [`argmin_blocked`]. Tuned for L1 cache residency.
pub const BLOCK_SIZE: usize = 256;
/// Block size for SIMD filtered search. Must be a multiple of the SIMD width (4 for NEON, 8 for AVX2, 16 for AVX-512).
pub const SIMD_FILTER_BLOCK: usize = 32;

#[cfg(target_arch = "aarch64")]
const SIMD_LANES: usize = 4;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SIMD_LANES: usize = 8;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
const SIMD_LANES: usize = 1;

/// Simple scalar argmin with branching.
///
/// Each comparison `v < min` is a branch. On random data, this branch is ~50%
/// mispredicted early in the array, causing pipeline stalls.
pub fn argmin_scalar(values: &[i32]) -> Option<usize> {
    let mut iter = values.iter().enumerate();
    let (mut idx, mut min) = match iter.next() {
        Some((i, &v)) => (i, v),
        None => return None,
    };

    for (i, &v) in iter {
        if v < min {
            min = v;
            idx = i;
        }
    }

    Some(idx)
}

/// Branchless argmin using conditional moves.
///
/// Instead of branching on `v < min`, this uses conditional assignment which compiles
/// to `CMOV` instructions on x86 or conditional select on ARM. While this executes
/// more instructions per iteration, it eliminates branch mispredictions entirely.
///
/// On random data, this is often faster than [`argmin_scalar`] despite doing more work,
/// because mispredictions cost 10-20 cycles each while `CMOV` is just 1-2 cycles.
pub fn argmin_branchless(values: &[i32]) -> Option<usize> {
    let mut iter = values.iter().enumerate();
    let (mut idx, mut min) = match iter.next() {
        Some((i, &v)) => (i, v),
        None => return None,
    };

    for (i, &v) in iter {
        let lt = v < min;
        // These compile to CMOV instructions—no branch, no misprediction.
        min = if lt { v } else { min };
        idx = if lt { i } else { idx };
    }

    Some(idx)
}

/// Standard library argmin using `Iterator::min_by_key`.
pub fn argmin_std(values: &[i32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .min_by_key(|&(_, value)| value)
        .map(|(idx, _)| idx)
}

/// Finds the minimum value (without its index).
pub fn min_scalar(values: &[i32]) -> Option<i32> {
    values.iter().copied().min()
}

/// Finds the first index where `values[i] == needle`.
pub fn find_scalar(values: &[i32], needle: i32) -> Option<usize> {
    values.iter().position(|&v| v == needle)
}

/// Runtime detection for AVX2 support.
pub fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Runtime detection for AVX-512 support.
pub fn avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
fn avx512_enabled() -> bool {
    // Opt-in toggle for AVX-512 to keep AVX2 as the default path.
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ARGMIN_AVX512")
            .map(|v| {
                let v = v.to_ascii_lowercase();
                matches!(v.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

/// Runtime availability for SIMD implementations (AVX2/AVX-512 on x86, NEON on aarch64).
pub fn simd_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        avx2_available() || avx512_available()
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// SIMD-accelerated minimum value search with runtime dispatch.
///
/// Uses AVX2 (or AVX-512 when enabled) on x86 and NEON on aarch64, falling back to scalar otherwise.
/// The SIMD version processes multiple elements per iteration with unrolling for ILP.
pub fn min_simd(values: &[i32]) -> Option<i32> {
    if values.is_empty() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON availability.
        Some(unsafe { aarch64_neon::min_neon(values) })
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2_ok = avx2_available();
            if avx512_available() && (avx512_enabled() || !avx2_ok) {
                // SAFETY: guarded by AVX-512 runtime detection and non-empty slice.
                Some(unsafe { x86_avx512::min_avx512(values) })
            } else if avx2_ok {
                // SAFETY: guarded by AVX2 runtime detection and non-empty slice.
                Some(unsafe { x86_avx2::min_avx2(values) })
            } else {
                min_scalar(values)
            }
        }

        #[cfg(target_arch = "x86")]
        {
            if avx2_available() {
                // SAFETY: guarded by AVX2 runtime detection and non-empty slice.
                Some(unsafe { x86_avx2::min_avx2(values) })
            } else {
                min_scalar(values)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        min_scalar(values)
    }
}

/// SIMD-accelerated linear search with runtime dispatch.
///
/// Uses AVX2 (or AVX-512 when enabled) on x86 or NEON on aarch64, processing multiple elements per iteration.
/// Early exits on first match using vector compare masks.
pub fn find_simd(values: &[i32], needle: i32) -> Option<usize> {
    if values.is_empty() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON availability.
        unsafe { aarch64_neon::find_neon(values, needle) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2_ok = avx2_available();
            if avx512_available() && (avx512_enabled() || !avx2_ok) {
                // SAFETY: guarded by AVX-512 runtime detection.
                unsafe { x86_avx512::find_avx512(values, needle) }
            } else if avx2_ok {
                // SAFETY: guarded by AVX2 runtime detection.
                unsafe { x86_avx2::find_avx2(values, needle) }
            } else {
                find_scalar(values, needle)
            }
        }

        #[cfg(target_arch = "x86")]
        {
            if avx2_available() {
                // SAFETY: guarded by AVX2 runtime detection.
                unsafe { x86_avx2::find_avx2(values, needle) }
            } else {
                find_scalar(values, needle)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        find_scalar(values, needle)
    }
}

/// SIMD argmin tracking indices in parallel with values.
///
/// Maintains two vectors: one for minimum values, one for their indices. On each
/// iteration, uses `blendv` (blend based on comparison mask) to conditionally
/// update both vectors simultaneously—8 (AVX2) or 16 (AVX-512) elements per cycle on x86.
///
/// This avoids the scalar loop's serial dependency while tracking exact positions.
pub fn argmin_vector_indices(values: &[i32]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }

    if values.len() < SIMD_LANES || values.len() > i32::MAX as usize {
        return argmin_scalar(values);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON availability and length guard.
        Some(unsafe { aarch64_neon::argmin_vector_indices(values) })
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2_ok = avx2_available();
            if avx512_available() && (avx512_enabled() || !avx2_ok) {
                // SAFETY: guarded by AVX-512 runtime detection and length guard.
                Some(unsafe { x86_avx512::argmin_vector_indices(values) })
            } else if avx2_ok {
                // SAFETY: guarded by AVX2 runtime detection and length guard.
                Some(unsafe { x86_avx2::argmin_vector_indices(values) })
            } else {
                argmin_scalar(values)
            }
        }

        #[cfg(target_arch = "x86")]
        {
            if avx2_available() {
                // SAFETY: guarded by AVX2 runtime detection and length guard.
                Some(unsafe { x86_avx2::argmin_vector_indices(values) })
            } else {
                argmin_scalar(values)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        argmin_scalar(values)
    }
}

/// SIMD argmin with speculative block filtering.
///
/// Instead of tracking indices for every element, this approach:
/// 1. Computes a fast SIMD min over each block (32 elements)
/// 2. Only if the block contains a new minimum, scans it precisely
///
/// This exploits the fact that as we progress through the array, finding a new
/// minimum becomes increasingly rare (probability ~1/i at position i). Most blocks
/// can be skipped entirely after a quick SIMD comparison.
pub fn argmin_simd_filtered(values: &[i32]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON availability.
        Some(unsafe { aarch64_neon::argmin_simd_filtered(values) })
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2_ok = avx2_available();
            if avx512_available() && (avx512_enabled() || !avx2_ok) {
                // SAFETY: guarded by AVX-512 runtime detection.
                Some(unsafe { x86_avx512::argmin_simd_filtered(values) })
            } else if avx2_ok {
                // SAFETY: guarded by AVX2 runtime detection.
                Some(unsafe { x86_avx2::argmin_simd_filtered(values) })
            } else {
                argmin_scalar(values)
            }
        }

        #[cfg(target_arch = "x86")]
        {
            if avx2_available() {
                // SAFETY: guarded by AVX2 runtime detection.
                Some(unsafe { x86_avx2::argmin_simd_filtered(values) })
            } else {
                argmin_scalar(values)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        argmin_scalar(values)
    }
}

/// Two-pass argmin: find minimum value, then search for its index.
///
/// Counterintuitively, two passes can be faster than one because:
/// 1. Finding the min is a simple reduction—no index tracking overhead
/// 2. Linear search for a known value can early-exit (and is SIMD-friendly)
///
/// This separates concerns: `min_simd` uses pure reduction, `find_simd` uses early-exit.
pub fn argmin_min_then_find(values: &[i32]) -> Option<usize> {
    let min_val = min_simd(values)?;
    find_simd(values, min_val)
}

/// Block-wise argmin for very large arrays.
///
/// Strategy:
/// 1. Divide array into L1-cache-sized blocks
/// 2. Find the minimum value in each block (SIMD-accelerated)
/// 3. Identify which block contains the global minimum
/// 4. Search only that block for the exact index
///
/// This minimizes cache misses by ensuring each block fits in L1 cache and
/// avoids scanning blocks that can't contain the minimum.
pub fn argmin_blocked(values: &[i32]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }

    let mut best_val = i32::MAX;
    let mut best_block = 0usize;
    let mut i = 0usize;

    // Phase 1: Find which block contains the minimum
    while i < values.len() {
        let end = (i + BLOCK_SIZE).min(values.len());
        let block_min = min_simd(&values[i..end]).unwrap_or(i32::MAX);
        if block_min < best_val {
            best_val = block_min;
            best_block = i;
        }
        i += BLOCK_SIZE;
    }

    // Phase 2: Search only the winning block for the exact index
    let end = (best_block + BLOCK_SIZE).min(values.len());
    let rel = find_simd(&values[best_block..end], best_val)?;
    Some(best_block + rel)
}

/// NEON SIMD implementations for aarch64.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod aarch64_neon {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn mask_any(mask: uint32x4_t) -> bool {
        let mut tmp = [0u32; 4];
        vst1q_u32(tmp.as_mut_ptr(), mask);
        tmp.iter().any(|&v| v != 0)
    }

    #[inline]
    unsafe fn first_lane(mask: uint32x4_t) -> Option<usize> {
        let mut tmp = [0u32; 4];
        vst1q_u32(tmp.as_mut_ptr(), mask);
        for (lane, &v) in tmp.iter().enumerate() {
            if v != 0 {
                return Some(lane);
            }
        }
        None
    }

    /// NEON minimum with 4x unrolling for instruction-level parallelism.
    #[target_feature(enable = "neon")]
    pub unsafe fn min_neon(values: &[i32]) -> i32 {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let mut min_v = vdupq_n_s32(i32::MAX);

        let vec_len = len & !15;
        while i < vec_len {
            let v0 = vld1q_s32(ptr.add(i));
            let v1 = vld1q_s32(ptr.add(i + 4));
            let v2 = vld1q_s32(ptr.add(i + 8));
            let v3 = vld1q_s32(ptr.add(i + 12));
            min_v = vminq_s32(min_v, v0);
            min_v = vminq_s32(min_v, v1);
            min_v = vminq_s32(min_v, v2);
            min_v = vminq_s32(min_v, v3);
            i += 16;
        }

        while i + 4 <= len {
            let v = vld1q_s32(ptr.add(i));
            min_v = vminq_s32(min_v, v);
            i += 4;
        }

        let mut tmp = [0i32; 4];
        vst1q_s32(tmp.as_mut_ptr(), min_v);
        let mut min_val = tmp[0];
        for &v in &tmp[1..] {
            if v < min_val {
                min_val = v;
            }
        }

        while i < len {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
            }
            i += 1;
        }

        min_val
    }

    /// NEON linear search with early exit.
    #[target_feature(enable = "neon")]
    pub unsafe fn find_neon(values: &[i32], needle: i32) -> Option<usize> {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let needle_v = vdupq_n_s32(needle);

        let vec_len = len & !15;
        while i < vec_len {
            let m0 = vceqq_s32(needle_v, vld1q_s32(ptr.add(i)));
            if let Some(lane) = first_lane(m0) {
                return Some(i + lane);
            }
            let m1 = vceqq_s32(needle_v, vld1q_s32(ptr.add(i + 4)));
            if let Some(lane) = first_lane(m1) {
                return Some(i + 4 + lane);
            }
            let m2 = vceqq_s32(needle_v, vld1q_s32(ptr.add(i + 8)));
            if let Some(lane) = first_lane(m2) {
                return Some(i + 8 + lane);
            }
            let m3 = vceqq_s32(needle_v, vld1q_s32(ptr.add(i + 12)));
            if let Some(lane) = first_lane(m3) {
                return Some(i + 12 + lane);
            }
            i += 16;
        }

        while i + 4 <= len {
            let m = vceqq_s32(needle_v, vld1q_s32(ptr.add(i)));
            if let Some(lane) = first_lane(m) {
                return Some(i + lane);
            }
            i += 4;
        }

        while i < len {
            if *ptr.add(i) == needle {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// Parallel argmin tracking both values and indices in NEON registers.
    #[target_feature(enable = "neon")]
    pub unsafe fn argmin_vector_indices(values: &[i32]) -> usize {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let vec_len = len & !3;

        let mut min_v = vdupq_n_s32(i32::MAX);
        let mut idx_v = vdupq_n_s32(0);
        let indices = [0i32, 1, 2, 3];
        let mut cur = vld1q_s32(indices.as_ptr());
        let step = vdupq_n_s32(4);

        while i < vec_len {
            let x = vld1q_s32(ptr.add(i));
            let mask = vcgtq_s32(min_v, x);

            let min_u = vreinterpretq_u32_s32(min_v);
            let x_u = vreinterpretq_u32_s32(x);
            let blended = vbslq_u32(mask, x_u, min_u);
            min_v = vreinterpretq_s32_u32(blended);

            let idx_u = vreinterpretq_u32_s32(idx_v);
            let cur_u = vreinterpretq_u32_s32(cur);
            let idx_blended = vbslq_u32(mask, cur_u, idx_u);
            idx_v = vreinterpretq_s32_u32(idx_blended);

            cur = vaddq_s32(cur, step);
            i += 4;
        }

        let mut min_arr = [0i32; 4];
        let mut idx_arr = [0i32; 4];
        vst1q_s32(min_arr.as_mut_ptr(), min_v);
        vst1q_s32(idx_arr.as_mut_ptr(), idx_v);

        let mut best_min = min_arr[0];
        let mut best_idx = idx_arr[0] as usize;
        for lane in 1..4 {
            let v = min_arr[lane];
            let idx = idx_arr[lane] as usize;
            if v < best_min || (v == best_min && idx < best_idx) {
                best_min = v;
                best_idx = idx;
            }
        }

        while i < len {
            let v = *ptr.add(i);
            if v < best_min {
                best_min = v;
                best_idx = i;
            }
            i += 1;
        }

        best_idx
    }

    /// Speculative block-filtering argmin using NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn argmin_simd_filtered(values: &[i32]) -> usize {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut min_val = i32::MAX;
        let mut block_idx = 0usize;
        let mut min_v = vdupq_n_s32(min_val);

        let mut i = 0usize;
        let vec_len = len & !(super::SIMD_FILTER_BLOCK - 1);
        while i < vec_len {
            let v0 = vld1q_s32(ptr.add(i));
            let v1 = vld1q_s32(ptr.add(i + 4));
            let v2 = vld1q_s32(ptr.add(i + 8));
            let v3 = vld1q_s32(ptr.add(i + 12));
            let v4 = vld1q_s32(ptr.add(i + 16));
            let v5 = vld1q_s32(ptr.add(i + 20));
            let v6 = vld1q_s32(ptr.add(i + 24));
            let v7 = vld1q_s32(ptr.add(i + 28));

            let m01 = vminq_s32(v0, v1);
            let m23 = vminq_s32(v2, v3);
            let m45 = vminq_s32(v4, v5);
            let m67 = vminq_s32(v6, v7);
            let m0123 = vminq_s32(m01, m23);
            let m4567 = vminq_s32(m45, m67);
            let y = vminq_s32(m0123, m4567);

            let mask = vcgtq_s32(min_v, y);
            if mask_any(mask) {
                block_idx = i;
                let block_end = i + super::SIMD_FILTER_BLOCK;
                let mut local_min = min_val;
                for j in i..block_end {
                    let v = *ptr.add(j);
                    if v < local_min {
                        local_min = v;
                    }
                }
                min_val = local_min;
                min_v = vdupq_n_s32(min_val);
            }

            i += super::SIMD_FILTER_BLOCK;
        }

        let mut min_idx = block_idx;
        let mut tail_exact = false;
        while i < len {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
                min_idx = i;
                tail_exact = true;
            }
            i += 1;
        }

        if tail_exact {
            return min_idx;
        }

        let block_end = (block_idx + super::SIMD_FILTER_BLOCK).min(len);
        for j in block_idx..block_end {
            if *ptr.add(j) == min_val {
                return j;
            }
        }

        block_idx
    }
}

/// AVX2 SIMD implementations for x86/x86_64.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
mod x86_avx2 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86 as arch;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64 as arch;

    use arch::{
        __m256i, _mm256_add_epi32, _mm256_blendv_epi8, _mm256_castsi256_ps, _mm256_cmpeq_epi32,
        _mm256_cmpgt_epi32, _mm256_loadu_si256, _mm256_min_epi32, _mm256_movemask_ps,
        _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_testz_si256,
    };

    /// AVX2 minimum with 4x unrolling for instruction-level parallelism.
    ///
    /// Processes 32 elements (4 × 8-lane vectors) per iteration. The 4x unrolling
    /// allows the CPU to execute multiple independent min operations in parallel,
    /// hiding the latency of each instruction.
    #[target_feature(enable = "avx2")]
    pub unsafe fn min_avx2(values: &[i32]) -> i32 {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let mut min_v = _mm256_set1_epi32(i32::MAX);

        // Main loop: 4x unrolled for ILP (32 elements per iteration)
        let vec_len = len & !31;
        while i < vec_len {
            let v0 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let v1 = _mm256_loadu_si256(ptr.add(i + 8) as *const __m256i);
            let v2 = _mm256_loadu_si256(ptr.add(i + 16) as *const __m256i);
            let v3 = _mm256_loadu_si256(ptr.add(i + 24) as *const __m256i);
            // Four independent min operations can execute in parallel
            min_v = _mm256_min_epi32(min_v, v0);
            min_v = _mm256_min_epi32(min_v, v1);
            min_v = _mm256_min_epi32(min_v, v2);
            min_v = _mm256_min_epi32(min_v, v3);
            i += 32;
        }

        // Handle remaining full vectors
        while i + 8 <= len {
            let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            min_v = _mm256_min_epi32(min_v, v);
            i += 8;
        }

        // Horizontal reduction: extract 8 lanes and find minimum
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, min_v);
        let mut min_val = tmp[0];
        for &v in &tmp[1..] {
            if v < min_val {
                min_val = v;
            }
        }

        // Scalar tail for remaining elements
        while i < len {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
            }
            i += 1;
        }

        min_val
    }

    /// AVX2 linear search with early exit.
    ///
    /// Uses `cmpeq` to compare 8 elements at once, then `movemask` to extract
    /// comparison results as a bitmask. `trailing_zeros()` on the mask gives
    /// the index of the first match.
    ///
    /// The trick: `movemask_ps` on i32 data works because we cast the comparison
    /// mask to float—we only care about the sign bits, which `movemask` extracts.
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_avx2(values: &[i32], needle: i32) -> Option<usize> {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let needle_v = _mm256_set1_epi32(needle);

        let vec_len = len & !31;
        while i < vec_len {
            // Compare 32 elements across 4 vectors
            let m1 = _mm256_cmpeq_epi32(needle_v, _mm256_loadu_si256(ptr.add(i) as *const __m256i));
            let m2 = _mm256_cmpeq_epi32(
                needle_v,
                _mm256_loadu_si256(ptr.add(i + 8) as *const __m256i),
            );
            let m3 = _mm256_cmpeq_epi32(
                needle_v,
                _mm256_loadu_si256(ptr.add(i + 16) as *const __m256i),
            );
            let m4 = _mm256_cmpeq_epi32(
                needle_v,
                _mm256_loadu_si256(ptr.add(i + 24) as *const __m256i),
            );

            // Combine masks: movemask extracts sign bits (set by cmpeq on match)
            // Cast to ps is safe—we only care about the sign bit pattern
            let mask = (_mm256_movemask_ps(_mm256_castsi256_ps(m1)) as u32)
                | ((_mm256_movemask_ps(_mm256_castsi256_ps(m2)) as u32) << 8)
                | ((_mm256_movemask_ps(_mm256_castsi256_ps(m3)) as u32) << 16)
                | ((_mm256_movemask_ps(_mm256_castsi256_ps(m4)) as u32) << 24);

            if mask != 0 {
                // trailing_zeros gives position of first set bit = first match
                return Some(i + mask.trailing_zeros() as usize);
            }

            i += 32;
        }

        while i + 8 <= len {
            let m = _mm256_cmpeq_epi32(needle_v, _mm256_loadu_si256(ptr.add(i) as *const __m256i));
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(m)) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 8;
        }

        // Scalar tail
        while i < len {
            if *ptr.add(i) == needle {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// Parallel argmin tracking both values and indices in SIMD registers.
    ///
    /// Maintains two 8-lane vectors:
    /// - `min_v`: current minimum values for each lane
    /// - `idx_v`: indices where those minimums were found
    ///
    /// On each iteration, `cmpgt` creates a mask of lanes where the new value
    /// is smaller, and `blendv` uses that mask to conditionally update both
    /// the value and index vectors simultaneously.
    #[target_feature(enable = "avx2")]
    pub unsafe fn argmin_vector_indices(values: &[i32]) -> usize {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let vec_len = len & !7;

        let mut min_v = _mm256_set1_epi32(i32::MAX);
        let mut idx_v = _mm256_setzero_si256();
        // Current indices: [0, 1, 2, 3, 4, 5, 6, 7], incremented by 8 each iteration
        let mut cur = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let step = _mm256_set1_epi32(8);

        while i < vec_len {
            let x = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            // mask[lane] = 0xFFFFFFFF where min_v[lane] > x[lane]
            let mask = _mm256_cmpgt_epi32(min_v, x);
            // blendv: select x where mask is set, else keep min_v
            min_v = _mm256_blendv_epi8(min_v, x, mask);
            idx_v = _mm256_blendv_epi8(idx_v, cur, mask);
            cur = _mm256_add_epi32(cur, step);
            i += 8;
        }

        // Horizontal reduction across 8 lanes
        let mut min_arr = [0i32; 8];
        let mut idx_arr = [0i32; 8];
        _mm256_storeu_si256(min_arr.as_mut_ptr() as *mut __m256i, min_v);
        _mm256_storeu_si256(idx_arr.as_mut_ptr() as *mut __m256i, idx_v);

        let mut best_min = min_arr[0];
        let mut best_idx = idx_arr[0] as usize;
        for lane in 1..8 {
            let v = min_arr[lane];
            let idx = idx_arr[lane] as usize;
            // Tie-break by index for stable results
            if v < best_min || (v == best_min && idx < best_idx) {
                best_min = v;
                best_idx = idx;
            }
        }

        // Scalar tail
        while i < len {
            let v = *ptr.add(i);
            if v < best_min {
                best_min = v;
                best_idx = i;
            }
            i += 1;
        }

        best_idx
    }

    /// Speculative block-filtering argmin.
    ///
    /// Key insight: as we progress through the array, finding a new minimum becomes
    /// increasingly unlikely. This algorithm speculatively skips blocks that can't
    /// possibly contain a new minimum.
    ///
    /// For each 32-element block:
    /// 1. Quickly compute block's minimum using SIMD tree reduction
    /// 2. Compare against current best with `testz` (sets ZF if no lane passes)
    /// 3. Only if block *might* contain a better value, scan it precisely
    ///
    /// The `testz` check is nearly free and lets us skip most blocks entirely.
    #[target_feature(enable = "avx2")]
    pub unsafe fn argmin_simd_filtered(values: &[i32]) -> usize {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut min_val = i32::MAX;
        let mut block_idx = 0usize;
        let mut min_v = _mm256_set1_epi32(min_val);

        let mut i = 0usize;
        let vec_len = len & !(super::SIMD_FILTER_BLOCK - 1);
        while i < vec_len {
            // Tree reduction: 32 elements → 8 via SIMD min
            let y1 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let y2 = _mm256_loadu_si256(ptr.add(i + 8) as *const __m256i);
            let y3 = _mm256_loadu_si256(ptr.add(i + 16) as *const __m256i);
            let y4 = _mm256_loadu_si256(ptr.add(i + 24) as *const __m256i);

            let y1 = _mm256_min_epi32(y1, y2);
            let y3 = _mm256_min_epi32(y3, y4);
            let y = _mm256_min_epi32(y1, y3);

            // Check if any lane is smaller than current minimum
            let mask = _mm256_cmpgt_epi32(min_v, y);
            // testz returns 1 if all mask bits are zero (no improvement possible)
            if _mm256_testz_si256(mask, mask) == 0 {
                // Block contains a potential new minimum—scan precisely
                block_idx = i;
                let block_end = i + super::SIMD_FILTER_BLOCK;
                let mut local_min = min_val;
                for j in i..block_end {
                    let v = *ptr.add(j);
                    if v < local_min {
                        local_min = v;
                    }
                }
                min_val = local_min;
                min_v = _mm256_set1_epi32(min_val);
            }

            i += super::SIMD_FILTER_BLOCK;
        }

        // Handle tail elements
        let mut min_idx = block_idx;
        let mut tail_exact = false;
        while i < len {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
                min_idx = i;
                tail_exact = true;
            }
            i += 1;
        }

        if tail_exact {
            return min_idx;
        }

        // Find exact index within the winning block
        let block_end = (block_idx + super::SIMD_FILTER_BLOCK).min(len);
        for j in block_idx..block_end {
            if *ptr.add(j) == min_val {
                return j;
            }
        }

        block_idx
    }
}

/// AVX-512 SIMD implementations for x86_64.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod x86_avx512 {
    use std::arch::x86_64 as arch;

    use arch::{
        __m512i, _mm512_add_epi32, _mm512_cmpeq_epi32_mask, _mm512_cmpgt_epi32_mask,
        _mm512_loadu_si512, _mm512_mask_blend_epi32, _mm512_min_epi32, _mm512_set1_epi32,
        _mm512_setr_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    };

    /// AVX-512 minimum with 4x unrolling for instruction-level parallelism.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn min_avx512(values: &[i32]) -> i32 {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let mut min_v = _mm512_set1_epi32(i32::MAX);

        let vec_len = len & !63;
        while i < vec_len {
            let v0 = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
            let v1 = _mm512_loadu_si512(ptr.add(i + 16) as *const __m512i);
            let v2 = _mm512_loadu_si512(ptr.add(i + 32) as *const __m512i);
            let v3 = _mm512_loadu_si512(ptr.add(i + 48) as *const __m512i);
            min_v = _mm512_min_epi32(min_v, v0);
            min_v = _mm512_min_epi32(min_v, v1);
            min_v = _mm512_min_epi32(min_v, v2);
            min_v = _mm512_min_epi32(min_v, v3);
            i += 64;
        }

        while i + 16 <= len {
            let v = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
            min_v = _mm512_min_epi32(min_v, v);
            i += 16;
        }

        let mut tmp = [0i32; 16];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, min_v);
        let mut min_val = tmp[0];
        for &v in &tmp[1..] {
            if v < min_val {
                min_val = v;
            }
        }

        while i < len {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
            }
            i += 1;
        }

        min_val
    }

    /// AVX-512 linear search with early exit.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn find_avx512(values: &[i32], needle: i32) -> Option<usize> {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let needle_v = _mm512_set1_epi32(needle);

        let vec_len = len & !63;
        while i < vec_len {
            let m1 =
                _mm512_cmpeq_epi32_mask(needle_v, _mm512_loadu_si512(ptr.add(i) as *const __m512i));
            let m2 = _mm512_cmpeq_epi32_mask(
                needle_v,
                _mm512_loadu_si512(ptr.add(i + 16) as *const __m512i),
            );
            let m3 = _mm512_cmpeq_epi32_mask(
                needle_v,
                _mm512_loadu_si512(ptr.add(i + 32) as *const __m512i),
            );
            let m4 = _mm512_cmpeq_epi32_mask(
                needle_v,
                _mm512_loadu_si512(ptr.add(i + 48) as *const __m512i),
            );

            let mask =
                (m1 as u64) | ((m2 as u64) << 16) | ((m3 as u64) << 32) | ((m4 as u64) << 48);
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }

            i += 64;
        }

        while i + 16 <= len {
            let m =
                _mm512_cmpeq_epi32_mask(needle_v, _mm512_loadu_si512(ptr.add(i) as *const __m512i));
            if m != 0 {
                return Some(i + m.trailing_zeros() as usize);
            }
            i += 16;
        }

        while i < len {
            if *ptr.add(i) == needle {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// Parallel argmin tracking both values and indices in SIMD registers.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn argmin_vector_indices(values: &[i32]) -> usize {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut i = 0usize;
        let vec_len = len & !15;

        let mut min_v = _mm512_set1_epi32(i32::MAX);
        let mut idx_v = _mm512_setzero_si512();
        let mut cur = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let step = _mm512_set1_epi32(16);

        while i < vec_len {
            let x = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
            let mask = _mm512_cmpgt_epi32_mask(min_v, x);
            min_v = _mm512_mask_blend_epi32(mask, min_v, x);
            idx_v = _mm512_mask_blend_epi32(mask, idx_v, cur);
            cur = _mm512_add_epi32(cur, step);
            i += 16;
        }

        let mut min_arr = [0i32; 16];
        let mut idx_arr = [0i32; 16];
        _mm512_storeu_si512(min_arr.as_mut_ptr() as *mut __m512i, min_v);
        _mm512_storeu_si512(idx_arr.as_mut_ptr() as *mut __m512i, idx_v);

        let mut best_min = min_arr[0];
        let mut best_idx = idx_arr[0] as usize;
        for lane in 1..16 {
            let v = min_arr[lane];
            let idx = idx_arr[lane] as usize;
            if v < best_min || (v == best_min && idx < best_idx) {
                best_min = v;
                best_idx = idx;
            }
        }

        while i < len {
            let v = *ptr.add(i);
            if v < best_min {
                best_min = v;
                best_idx = i;
            }
            i += 1;
        }

        best_idx
    }

    /// Speculative block-filtering argmin using AVX-512.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn argmin_simd_filtered(values: &[i32]) -> usize {
        let len = values.len();
        let ptr = values.as_ptr();
        let mut min_val = i32::MAX;
        let mut block_idx = 0usize;
        let mut min_v = _mm512_set1_epi32(min_val);

        let mut i = 0usize;
        let vec_len = len & !(super::SIMD_FILTER_BLOCK - 1);
        while i < vec_len {
            let y1 = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
            let y2 = _mm512_loadu_si512(ptr.add(i + 16) as *const __m512i);
            let y = _mm512_min_epi32(y1, y2);

            let mask = _mm512_cmpgt_epi32_mask(min_v, y);
            if mask != 0 {
                block_idx = i;
                let block_end = i + super::SIMD_FILTER_BLOCK;
                let mut local_min = min_val;
                for j in i..block_end {
                    let v = *ptr.add(j);
                    if v < local_min {
                        local_min = v;
                    }
                }
                min_val = local_min;
                min_v = _mm512_set1_epi32(min_val);
            }

            i += super::SIMD_FILTER_BLOCK;
        }

        let mut min_idx = block_idx;
        let mut tail_exact = false;
        while i < len {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
                min_idx = i;
                tail_exact = true;
            }
            i += 1;
        }

        if tail_exact {
            return min_idx;
        }

        let block_end = (block_idx + super::SIMD_FILTER_BLOCK).min(len);
        for j in block_idx..block_end {
            if *ptr.add(j) == min_val {
                return j;
            }
        }

        block_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference_argmin(values: &[i32]) -> Option<usize> {
        values
            .iter()
            .enumerate()
            .min_by_key(|&(_, value)| value)
            .map(|(idx, _)| idx)
    }

    proptest! {
        #[test]
        fn argmin_variants_match_reference(values in proptest::collection::vec(any::<i32>(), 0..=2048)) {
            let expected = reference_argmin(&values);
            prop_assert_eq!(argmin_scalar(&values), expected);
            prop_assert_eq!(argmin_branchless(&values), expected);
            prop_assert_eq!(argmin_std(&values), expected);
            prop_assert_eq!(argmin_vector_indices(&values), expected);
            prop_assert_eq!(argmin_simd_filtered(&values), expected);
            prop_assert_eq!(argmin_min_then_find(&values), expected);
            prop_assert_eq!(argmin_blocked(&values), expected);

            prop_assert_eq!(min_scalar(&values), min_simd(&values));

            let needle = match expected {
                Some(idx) => values[idx],
                None => 0,
            };
            prop_assert_eq!(find_scalar(&values, needle), find_simd(&values, needle));
        }
    }

    #[test]
    fn argmin_basic_cases() {
        let empty: [i32; 0] = [];
        assert_eq!(argmin_scalar(&empty), None);

        let single = [42];
        assert_eq!(argmin_scalar(&single), Some(0));

        let values = [3, 1, 2, 1];
        assert_eq!(argmin_scalar(&values), Some(1));
        assert_eq!(argmin_blocked(&values), Some(1));
        assert_eq!(argmin_min_then_find(&values), Some(1));
    }

    #[test]
    fn find_scalar_basic_cases() {
        let values = [5, 4, 3, 2, 1];
        assert_eq!(find_scalar(&values, 3), Some(2));
        assert_eq!(find_scalar(&values, 9), None);
    }
}
