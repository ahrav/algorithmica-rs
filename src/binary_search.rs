//! Binary search variants for sorted arrays.
//!
//! This module implements several binary search styles from the reference chapter,
//! focusing on branchless code and the Eytzinger layout for cache-friendly access.
//!
//! # Strategies
//!
//! | Function | Strategy | Best For |
//! |----------|----------|----------|
//! | [`binary_search_std`] | Stdlib binary search | Baseline, clarity |
//! | [`binary_search_branchless`] | Branchless lower-bound loop | Random queries (fewer mispredicts) |
//! | [`binary_search_branchless_prefetch`] | Branchless + mid-point prefetch | Larger arrays |
//! | [`binary_search_eytzinger`] | Eytzinger layout traversal | Cache-friendly layout |
//! | [`binary_search_eytzinger_prefetch`] | Eytzinger + prefetch | Larger arrays |
//!
//! # Performance notes
//!
//! - **Branchless**: Removes the data-dependent branch in the compare by using arithmetic
//!   selection. Results are compiler- and workload-dependent.
//! - **Eytzinger**: Stores the implicit search tree in level order for better spatial locality.
//!   If the array is cache-line aligned, the first 15 `i32` values (4 levels) can share one line.
//! - **Prefetch**: Software prefetch can hide memory latency on large arrays but adds overhead;
//!   results depend on hardware and alignment.
//!
//! # References
//!
//! - [Binary search chapter](https://en.algorithmica.org/hpc/data-structures/binary-search/)

/// Prefetch 4 levels ahead (16 nodes) as in the reference; alignment affects cache-line grouping.
const EYT_PREFETCH_DISTANCE: usize = 16;

/// Arrays smaller than this (in bytes) skip prefetching.
/// 32KB is a conservative L1 data cache size across modern CPUs.
const PREFETCH_THRESHOLD_BYTES: usize = 32 * 1024;

/// Minimum element count to enable prefetching (i32 = 4 bytes).
const PREFETCH_THRESHOLD_ELEMENTS: usize = PREFETCH_THRESHOLD_BYTES / std::mem::size_of::<i32>();

/// Eytzinger (level-order) layout for a sorted array.
///
/// Elements are stored in BFS order of an implicit binary search tree:
/// - Index 1: root
/// - Index 2k: left child of k
/// - Index 2k+1: right child of k
///
/// Index 0 is unused (simplifies child arithmetic).
#[derive(Clone)]
pub struct EytzingerLayout {
    values: Vec<i32>,
    indices: Vec<usize>,
}

impl EytzingerLayout {
    /// Builds the Eytzinger layout from a sorted slice.
    pub fn new(sorted: &[i32]) -> Self {
        let n = sorted.len();
        let mut values = vec![0i32; n + 1];
        let mut indices = vec![0usize; n + 1];
        let mut i = 0usize;
        build_eytzinger(1, n, sorted, &mut values, &mut indices, &mut i);
        Self { values, indices }
    }

    /// Number of elements in the layout.
    pub fn len(&self) -> usize {
        self.values.len().saturating_sub(1)
    }

    /// Returns true when the layout is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn build_eytzinger(
    k: usize,
    n: usize,
    sorted: &[i32],
    values: &mut [i32],
    indices: &mut [usize],
    i: &mut usize,
) {
    if k > n {
        return;
    }
    build_eytzinger(k * 2, n, sorted, values, indices, i);
    values[k] = sorted[*i];
    indices[k] = *i;
    *i += 1;
    build_eytzinger(k * 2 + 1, n, sorted, values, indices, i);
}

/// Standard library binary search.
pub fn binary_search_std(values: &[i32], needle: i32) -> Option<usize> {
    values.binary_search(&needle).ok()
}

/// Branchless binary search using a lower-bound loop.
///
/// Uses arithmetic selection for the midpoint compare to avoid a data-dependent branch.
/// Returns the first matching index when duplicates exist.
pub fn binary_search_branchless(values: &[i32], needle: i32) -> Option<usize> {
    if values.is_empty() {
        return None;
    }

    let mut base = 0usize;
    let mut len = values.len();
    while len > 1 {
        let half = len / 2;
        let mid = base + half - 1;
        // SAFETY: Loop invariant: base + len <= values.len()
        // Since len > 1 and half = len/2, we have half >= 1
        // So mid = base + half - 1 < base + len <= values.len()
        let cmp = unsafe { *values.get_unchecked(mid) < needle };
        base += cmp as usize * half;
        len -= half;
    }

    // SAFETY: base < values.len() because base + len <= values.len() and len >= 1
    if unsafe { *values.get_unchecked(base) } == needle {
        Some(base)
    } else {
        None
    }
}

/// Branchless binary search with prefetching of the next midpoints.
pub fn binary_search_branchless_prefetch(values: &[i32], needle: i32) -> Option<usize> {
    // For small arrays that fit in L1/L2 cache, prefetching adds overhead without benefit
    if values.len() < PREFETCH_THRESHOLD_ELEMENTS {
        return binary_search_branchless(values, needle);
    }

    let mut base = 0usize;
    let mut len = values.len();
    while len > 1 {
        let half = len / 2;
        let left_len = len - half;

        if left_len > 1 {
            let idx = base + left_len / 2 - 1;
            prefetch_index(values, idx);
        }
        if half > 1 {
            let idx = base + half + half / 2 - 1;
            prefetch_index(values, idx);
        }

        let mid = base + half - 1;
        // SAFETY: Same invariant as binary_search_branchless - mid < values.len()
        let cmp = unsafe { *values.get_unchecked(mid) < needle };
        base += cmp as usize * half;
        len = left_len;
    }

    // SAFETY: base < values.len() because base + len <= values.len() and len >= 1
    if unsafe { *values.get_unchecked(base) } == needle {
        Some(base)
    } else {
        None
    }
}

/// Binary search over an Eytzinger layout.
///
/// The layout groups tree levels into contiguous memory, improving cache
/// locality on large arrays. Uses branchless child selection and bit
/// operations to recover the lower-bound candidate.
pub fn binary_search_eytzinger(layout: &EytzingerLayout, needle: i32) -> Option<usize> {
    let n = layout.len();
    if n == 0 {
        return None;
    }

    let mut k = 1usize;
    while k <= n {
        // SAFETY: k is in range [1, n] during iteration
        let cmp = unsafe { *layout.values.get_unchecked(k) < needle };
        k = (k << 1) + cmp as usize;
    }

    // After traversal, k is past a leaf. Backtrack to the last visited node:
    // trailing_ones + 1 matches ffs(~k) in the reference implementation.
    let shift = (k.trailing_ones() + 1) as usize;
    k >>= shift;

    // SAFETY: k is in range [1, n] after backtracking (k=0 only possible if n=0, handled above)
    if k > 0 && unsafe { *layout.values.get_unchecked(k) } == needle {
        Some(unsafe { *layout.indices.get_unchecked(k) })
    } else {
        None
    }
}

/// Binary search over an Eytzinger layout with prefetching.
pub fn binary_search_eytzinger_prefetch(layout: &EytzingerLayout, needle: i32) -> Option<usize> {
    // For small arrays that fit in L1/L2 cache, prefetching adds overhead without benefit
    if layout.len() < PREFETCH_THRESHOLD_ELEMENTS {
        return binary_search_eytzinger(layout, needle);
    }

    let n = layout.len();

    let mut k = 1usize;
    while k <= n {
        if let Some(idx) = k.checked_mul(EYT_PREFETCH_DISTANCE).filter(|&idx| idx <= n) {
            prefetch_index(&layout.values, idx);
        }
        // SAFETY: k is in range [1, n] during iteration
        let cmp = unsafe { *layout.values.get_unchecked(k) < needle };
        k = (k << 1) + cmp as usize;
    }

    let shift = (k.trailing_ones() + 1) as usize;
    k >>= shift;

    // SAFETY: k is in range [1, n] after backtracking (k=0 only possible if n=0, handled above)
    if k > 0 && unsafe { *layout.values.get_unchecked(k) } == needle {
        Some(unsafe { *layout.indices.get_unchecked(k) })
    } else {
        None
    }
}

#[inline]
fn prefetch_index(values: &[i32], idx: usize) {
    if idx < values.len() {
        // SAFETY: idx is bounds-checked above.
        let ptr = unsafe { values.as_ptr().add(idx) };
        prefetch_read_i32(ptr);
    }
}

#[inline]
fn prefetch_read_i32(ptr: *const i32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags, readonly)
        );
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn eytzinger_finds_all_values() {
        let values = [1, 3, 5, 7, 9, 11, 13];
        let layout = EytzingerLayout::new(&values);

        for (idx, &value) in values.iter().enumerate() {
            assert_eq!(binary_search_eytzinger(&layout, value), Some(idx));
            assert_eq!(binary_search_eytzinger_prefetch(&layout, value), Some(idx));
        }

        assert_eq!(binary_search_eytzinger(&layout, 6), None);
        assert_eq!(binary_search_eytzinger_prefetch(&layout, 6), None);
    }

    proptest! {
        #[test]
        fn variants_match_std(mut values in prop::collection::vec(any::<i32>(), 0..256), needle in any::<i32>()) {
            values.sort();
            values.dedup();

            let expected = binary_search_std(&values, needle);
            let layout = EytzingerLayout::new(&values);

            prop_assert_eq!(binary_search_branchless(&values, needle), expected);
            prop_assert_eq!(binary_search_branchless_prefetch(&values, needle), expected);
            prop_assert_eq!(binary_search_eytzinger(&layout, needle), expected);
            prop_assert_eq!(binary_search_eytzinger_prefetch(&layout, needle), expected);
        }
    }
}
