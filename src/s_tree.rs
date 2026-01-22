//! Static B-tree (S-tree) and B+ tree (S+ tree) layouts for fast search.
//!
//! These layouts follow the "S-tree" and "S+ tree" constructions from the
//! reference, using wide node fanout (B=16) and SIMD comparisons to find the
//! first key >= needle inside each node.
//!
//! # Strategies
//!
//! | Function | Strategy | Best For |
//! |----------|----------|----------|
//! | [`s_tree_search_scalar`] | Implicit B-tree + scalar rank | Baseline, clarity |
//! | [`s_tree_search_neon`] | Implicit B-tree + NEON rank | aarch64 throughput |
//! | [`s_plus_tree_search_scalar`] | Implicit B+ tree + scalar rank | Baseline, clarity |
//! | [`s_plus_tree_search_neon`] | Implicit B+ tree + NEON rank | aarch64 throughput |
//!
//! # References
//!
//! - [S-tree / S+ tree chapter](https://en.algorithmica.org/hpc/data-structures/s-tree/)

const NODE_KEYS: usize = 16;
const NODE_FANOUT: usize = NODE_KEYS + 1;

/// Implicit B-tree (S-tree) layout.
#[derive(Clone)]
pub struct STreeLayout {
    values: Vec<i32>,
    indices: Vec<usize>,
    len: usize,
    nblocks: usize,
}

impl STreeLayout {
    /// Builds the S-tree layout from a sorted slice.
    pub fn new(sorted: &[i32]) -> Self {
        let len = sorted.len();
        if len == 0 {
            return Self {
                values: Vec::new(),
                indices: Vec::new(),
                len: 0,
                nblocks: 0,
            };
        }

        let nblocks = blocks(len);
        let mut values = vec![i32::MAX; nblocks * NODE_KEYS];
        let mut indices = vec![len; nblocks * NODE_KEYS];
        let mut t = 0usize;
        build_s_tree(0, nblocks, sorted, &mut values, &mut indices, &mut t, len);

        Self {
            values,
            indices,
            len,
            nblocks,
        }
    }

    /// Number of elements in the original sorted array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when the layout is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Implicit B+ tree (S+ tree) layout.
#[derive(Clone)]
pub struct SPlusTreeLayout {
    values: Vec<i32>,
    offsets: Vec<usize>,
    len: usize,
    height: usize,
}

impl SPlusTreeLayout {
    /// Builds the S+ tree layout from a sorted slice.
    pub fn new(sorted: &[i32]) -> Self {
        let len = sorted.len();
        if len == 0 {
            return Self {
                values: Vec::new(),
                offsets: vec![0],
                len: 0,
                height: 0,
            };
        }

        let offsets = s_plus_offsets(len);
        let height = offsets.len().saturating_sub(1);
        let size = *offsets.last().unwrap_or(&0);
        let mut values = vec![i32::MAX; size];
        values[..len].copy_from_slice(sorted);

        for h in 1..height {
            let layer_start = offsets[h];
            let layer_end = offsets[h + 1];
            let keys_in_layer = layer_end - layer_start;
            for i in 0..keys_in_layer {
                let k = i / NODE_KEYS;
                let j = i - k * NODE_KEYS;
                let mut child = k * NODE_FANOUT + j + 1;
                for _ in 0..(h - 1) {
                    child *= NODE_FANOUT;
                }
                let leaf_index = child * NODE_KEYS;
                values[layer_start + i] = if leaf_index < len {
                    values[leaf_index]
                } else {
                    i32::MAX
                };
            }
        }

        Self {
            values,
            offsets,
            len,
            height,
        }
    }

    /// Number of elements in the original sorted array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when the layout is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Scalar S-tree search using a per-node linear scan.
pub fn s_tree_search_scalar(layout: &STreeLayout, needle: i32) -> Option<usize> {
    s_tree_search_impl(layout, needle, rank_scalar)
}

/// NEON-accelerated S-tree search (falls back to scalar off aarch64).
pub fn s_tree_search_neon(layout: &STreeLayout, needle: i32) -> Option<usize> {
    s_tree_search_impl(layout, needle, rank_simd)
}

/// Scalar S+ tree search using a per-node linear scan.
pub fn s_plus_tree_search_scalar(layout: &SPlusTreeLayout, needle: i32) -> Option<usize> {
    s_plus_tree_search_impl(layout, needle, rank_scalar)
}

/// NEON-accelerated S+ tree search (falls back to scalar off aarch64).
pub fn s_plus_tree_search_neon(layout: &SPlusTreeLayout, needle: i32) -> Option<usize> {
    s_plus_tree_search_impl(layout, needle, rank_simd)
}

type RankFn = fn(&[i32], i32) -> usize;

#[inline]
fn rank_scalar(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS);
    for (i, &value) in node.iter().take(NODE_KEYS).enumerate() {
        if value >= needle {
            return i;
        }
    }
    NODE_KEYS
}

#[inline]
fn rank_simd(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_rank::rank_ge(node.as_ptr(), needle)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        rank_scalar(node, needle)
    }
}

fn s_tree_search_impl(layout: &STreeLayout, needle: i32, rank: RankFn) -> Option<usize> {
    if layout.is_empty() {
        return None;
    }

    let mut k = 0usize;
    let mut res_idx = layout.len;
    let mut res_val = i32::MAX;

    while k < layout.nblocks {
        let node_start = k * NODE_KEYS;
        let node = &layout.values[node_start..node_start + NODE_KEYS];
        let i = rank(node, needle);
        if i < NODE_KEYS {
            let idx = layout.indices[node_start + i];
            if idx < layout.len {
                res_idx = idx;
                res_val = layout.values[node_start + i];
            }
        }
        k = go(k, i);
    }

    if res_idx < layout.len && res_val == needle {
        Some(res_idx)
    } else {
        None
    }
}

fn s_plus_tree_search_impl(layout: &SPlusTreeLayout, needle: i32, rank: RankFn) -> Option<usize> {
    if layout.is_empty() {
        return None;
    }

    let mut node_offset = 0usize;
    for h in (1..layout.height).rev() {
        let layer_start = layout.offsets[h];
        let node_start = layer_start + node_offset;
        let node = &layout.values[node_start..node_start + NODE_KEYS];
        let i = rank(node, needle);
        node_offset = node_offset * NODE_FANOUT + i * NODE_KEYS;
    }

    let leaf_start = node_offset;
    let node = &layout.values[leaf_start..leaf_start + NODE_KEYS];
    let i = rank(node, needle);
    let idx = leaf_start + i;

    if idx < layout.len && layout.values[idx] == needle {
        Some(idx)
    } else {
        None
    }
}

#[inline]
fn go(k: usize, i: usize) -> usize {
    k * NODE_FANOUT + i + 1
}

#[inline]
fn blocks(n: usize) -> usize {
    n.div_ceil(NODE_KEYS)
}

fn s_plus_offsets(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![0];
    }
    let height = s_plus_height(n);
    let mut offsets = Vec::with_capacity(height + 1);
    let mut total = 0usize;
    let mut keys = n;
    offsets.push(0);
    for _ in 0..height {
        let layer_size = blocks(keys) * NODE_KEYS;
        total += layer_size;
        offsets.push(total);
        if keys <= NODE_KEYS {
            break;
        }
        keys = s_plus_prev_keys(keys);
    }
    offsets
}

fn s_plus_height(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n <= NODE_KEYS {
        return 1;
    }
    1 + s_plus_height(s_plus_prev_keys(n))
}

fn s_plus_prev_keys(n: usize) -> usize {
    ((blocks(n) + NODE_KEYS) / NODE_FANOUT) * NODE_KEYS
}

fn build_s_tree(
    k: usize,
    nblocks: usize,
    sorted: &[i32],
    values: &mut [i32],
    indices: &mut [usize],
    t: &mut usize,
    len: usize,
) {
    if k >= nblocks {
        return;
    }

    for i in 0..NODE_KEYS {
        build_s_tree(go(k, i), nblocks, sorted, values, indices, t, len);
        let dst = k * NODE_KEYS + i;
        if *t < len {
            values[dst] = sorted[*t];
            indices[dst] = *t;
            *t += 1;
        } else {
            values[dst] = i32::MAX;
            indices[dst] = len;
        }
    }

    build_s_tree(go(k, NODE_KEYS), nblocks, sorted, values, indices, t, len);
}

#[cfg(target_arch = "aarch64")]
mod neon_rank {
    use std::arch::aarch64::*;

    use super::NODE_KEYS;

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn rank_ge(node: *const i32, needle: i32) -> usize {
        unsafe {
            let x = vdupq_n_s32(needle);

            let v0 = vld1q_s32(node);
            let v1 = vld1q_s32(node.add(4));
            let v2 = vld1q_s32(node.add(8));
            let v3 = vld1q_s32(node.add(12));

            let m0 = vcgeq_s32(v0, x);
            let m1 = vcgeq_s32(v1, x);
            let m2 = vcgeq_s32(v2, x);
            let m3 = vcgeq_s32(v3, x);

            let mask0 = mask4(m0);
            let mask1 = mask4(m1);
            let mask2 = mask4(m2);
            let mask3 = mask4(m3);

            let mut mask = mask0 | (mask1 << 4) | (mask2 << 8) | (mask3 << 12);
            mask |= 1u32 << (NODE_KEYS as u32);
            mask.trailing_zeros() as usize
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mask4(cmp: uint32x4_t) -> u32 {
        ((vgetq_lane_u32(cmp, 0) >> 31) & 1)
            | (((vgetq_lane_u32(cmp, 1) >> 31) & 1) << 1)
            | (((vgetq_lane_u32(cmp, 2) >> 31) & 1) << 2)
            | (((vgetq_lane_u32(cmp, 3) >> 31) & 1) << 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_search_std;
    use proptest::prelude::*;

    #[test]
    fn stree_finds_values() {
        let values = [1, 3, 5, 7, 9, 11, 13];
        let layout = STreeLayout::new(&values);
        let splus = SPlusTreeLayout::new(&values);

        for (idx, &value) in values.iter().enumerate() {
            assert_eq!(s_tree_search_scalar(&layout, value), Some(idx));
            assert_eq!(s_tree_search_neon(&layout, value), Some(idx));
            assert_eq!(s_plus_tree_search_scalar(&splus, value), Some(idx));
            assert_eq!(s_plus_tree_search_neon(&splus, value), Some(idx));
        }

        assert_eq!(s_tree_search_scalar(&layout, 6), None);
        assert_eq!(s_plus_tree_search_neon(&splus, 6), None);
    }

    proptest! {
        #[test]
        fn variants_match_std(mut values in prop::collection::vec(any::<i32>(), 0..256), needle in any::<i32>()) {
            values.sort();
            values.dedup();

            let expected = binary_search_std(&values, needle);
            let layout = STreeLayout::new(&values);
            let splus = SPlusTreeLayout::new(&values);

            prop_assert_eq!(s_tree_search_scalar(&layout, needle), expected);
            prop_assert_eq!(s_tree_search_neon(&layout, needle), expected);
            prop_assert_eq!(s_plus_tree_search_scalar(&splus, needle), expected);
            prop_assert_eq!(s_plus_tree_search_neon(&splus, needle), expected);
        }
    }
}
