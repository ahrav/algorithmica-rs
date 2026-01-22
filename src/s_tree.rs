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
//! | [`s_plus_tree32_search_scalar`] | B+ tree (B=32) + scalar rank | Larger nodes |
//! | [`s_plus_tree32_search_neon`] | B+ tree (B=32) + NEON rank | Larger nodes |
//!
//! # References
//!
//! - [S-tree / S+ tree chapter](https://en.algorithmica.org/hpc/data-structures/s-tree/)

const NODE_KEYS: usize = 16;
const NODE_FANOUT: usize = NODE_KEYS + 1;
const NODE_KEYS_32: usize = 32;
const NODE_FANOUT_32: usize = NODE_KEYS_32 + 1;

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
        build_s_plus_internal(&mut values, &offsets, len, NODE_KEYS, NODE_FANOUT);
        permute_internal_nodes_16(&mut values, &offsets);

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

/// Implicit B+ tree (S+ tree) layout with B=32.
#[derive(Clone)]
pub struct SPlusTree32Layout {
    values: Vec<i32>,
    offsets: Vec<usize>,
    len: usize,
    height: usize,
}

impl SPlusTree32Layout {
    /// Builds the S+ tree (B=32) layout from a sorted slice.
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

        let offsets = s_plus_offsets_with(len, NODE_KEYS_32);
        let height = offsets.len().saturating_sub(1);
        let size = *offsets.last().unwrap_or(&0);
        let mut values = vec![i32::MAX; size];
        values[..len].copy_from_slice(sorted);

        build_s_plus_internal(&mut values, &offsets, len, NODE_KEYS_32, NODE_FANOUT_32);
        permute_internal_nodes_32(&mut values, &offsets);

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
    s_tree_search_impl(layout, needle, rank_direct_scalar_16)
}

/// NEON-accelerated S-tree search (falls back to scalar off aarch64).
pub fn s_tree_search_neon(layout: &STreeLayout, needle: i32) -> Option<usize> {
    s_tree_search_impl(layout, needle, rank_direct_simd_16)
}

/// Scalar S+ tree search using a per-node linear scan.
pub fn s_plus_tree_search_scalar(layout: &SPlusTreeLayout, needle: i32) -> Option<usize> {
    s_plus_tree_search_impl(
        layout,
        needle,
        rank_permuted_scalar_16,
        rank_direct_scalar_16,
    )
}

/// NEON-accelerated S+ tree search (falls back to scalar off aarch64).
pub fn s_plus_tree_search_neon(layout: &SPlusTreeLayout, needle: i32) -> Option<usize> {
    s_plus_tree_search_impl(layout, needle, rank_permuted_simd_16, rank_direct_simd_16)
}

/// Scalar S+ tree (B=32) search using a per-node linear scan.
pub fn s_plus_tree32_search_scalar(layout: &SPlusTree32Layout, needle: i32) -> Option<usize> {
    s_plus_tree32_search_impl(
        layout,
        needle,
        rank_permuted_scalar_32,
        rank_direct_scalar_32,
    )
}

/// NEON-accelerated S+ tree (B=32) search (falls back to scalar off aarch64).
pub fn s_plus_tree32_search_neon(layout: &SPlusTree32Layout, needle: i32) -> Option<usize> {
    s_plus_tree32_search_impl(layout, needle, rank_permuted_simd_32, rank_direct_simd_32)
}

type RankFn = fn(&[i32], i32) -> usize;

#[inline]
fn rank_direct_scalar_16(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS);
    for (i, &value) in node.iter().take(NODE_KEYS).enumerate() {
        if value >= needle {
            return i;
        }
    }
    NODE_KEYS
}

#[inline]
fn rank_permuted_scalar_16(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS);
    let mut base = 0usize;
    for &group in PERM_GROUP_ORDER_16.iter() {
        let start = group * 4;
        for i in 0..4 {
            if node[start + i] >= needle {
                return base + i;
            }
        }
        base += 4;
    }
    NODE_KEYS
}

#[inline]
fn rank_direct_scalar_32(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS_32);
    for (i, &value) in node.iter().take(NODE_KEYS_32).enumerate() {
        if value >= needle {
            return i;
        }
    }
    NODE_KEYS_32
}

#[inline]
fn rank_permuted_scalar_32(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS_32);
    let mut base = 0usize;
    for &group in PERM_GROUP_ORDER_32.iter() {
        let start = group * 4;
        for i in 0..4 {
            if node[start + i] >= needle {
                return base + i;
            }
        }
        base += 4;
    }
    NODE_KEYS_32
}

#[inline]
fn rank_direct_simd_16(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_rank::rank16_direct(node.as_ptr(), needle)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        rank_direct_scalar_16(node, needle)
    }
}

#[inline]
fn rank_permuted_simd_16(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_rank::rank16_permuted(node.as_ptr(), needle)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        rank_permuted_scalar_16(node, needle)
    }
}

#[inline]
fn rank_direct_simd_32(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS_32);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_rank::rank32_direct(node.as_ptr(), needle)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        rank_direct_scalar_32(node, needle)
    }
}

#[inline]
fn rank_permuted_simd_32(node: &[i32], needle: i32) -> usize {
    debug_assert!(node.len() >= NODE_KEYS_32);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_rank::rank32_permuted(node.as_ptr(), needle)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        rank_permuted_scalar_32(node, needle)
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

fn s_plus_tree_search_impl(
    layout: &SPlusTreeLayout,
    needle: i32,
    rank_internal: RankFn,
    rank_leaf: RankFn,
) -> Option<usize> {
    if layout.is_empty() {
        return None;
    }

    let mut node_offset = 0usize;
    for h in (1..layout.height).rev() {
        let layer_start = layout.offsets[h];
        let node_start = layer_start + node_offset;
        let node = &layout.values[node_start..node_start + NODE_KEYS];
        let i = rank_internal(node, needle);
        node_offset = node_offset * NODE_FANOUT + i * NODE_KEYS;
    }

    let leaf_start = node_offset;
    let node = &layout.values[leaf_start..leaf_start + NODE_KEYS];
    let i = rank_leaf(node, needle);
    let idx = leaf_start + i;

    if idx < layout.len && layout.values[idx] == needle {
        Some(idx)
    } else {
        None
    }
}

fn s_plus_tree32_search_impl(
    layout: &SPlusTree32Layout,
    needle: i32,
    rank_internal: RankFn,
    rank_leaf: RankFn,
) -> Option<usize> {
    if layout.is_empty() {
        return None;
    }

    let mut node_offset = 0usize;
    for h in (1..layout.height).rev() {
        let layer_start = layout.offsets[h];
        let node_start = layer_start + node_offset;
        let node = &layout.values[node_start..node_start + NODE_KEYS_32];
        let i = rank_internal(node, needle);
        node_offset = node_offset * NODE_FANOUT_32 + i * NODE_KEYS_32;
    }

    let leaf_start = node_offset;
    let node = &layout.values[leaf_start..leaf_start + NODE_KEYS_32];
    let i = rank_leaf(node, needle);
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

#[inline]
fn blocks_with(n: usize, block: usize) -> usize {
    n.div_ceil(block)
}

fn s_plus_offsets(n: usize) -> Vec<usize> {
    s_plus_offsets_with(n, NODE_KEYS)
}

fn s_plus_offsets_with(n: usize, block: usize) -> Vec<usize> {
    if n == 0 {
        return vec![0];
    }
    let height = s_plus_height_with(n, block);
    let mut offsets = Vec::with_capacity(height + 1);
    let mut total = 0usize;
    let mut keys = n;
    offsets.push(0);
    for _ in 0..height {
        let layer_size = blocks_with(keys, block) * block;
        total += layer_size;
        offsets.push(total);
        if keys <= block {
            break;
        }
        keys = s_plus_prev_keys_with(keys, block);
    }
    offsets
}

fn s_plus_height_with(n: usize, block: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n <= block {
        return 1;
    }
    1 + s_plus_height_with(s_plus_prev_keys_with(n, block), block)
}

fn s_plus_prev_keys_with(n: usize, block: usize) -> usize {
    let fanout = block + 1;
    ((blocks_with(n, block) + block) / fanout) * block
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

const PERM_GROUP_ORDER_16: [usize; 4] = [0, 2, 1, 3];
const PERM_GROUP_ORDER_32: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

fn build_s_plus_internal(
    values: &mut [i32],
    offsets: &[usize],
    len: usize,
    node_keys: usize,
    fanout: usize,
) {
    let height = offsets.len().saturating_sub(1);
    for h in 1..height {
        let layer_start = offsets[h];
        let layer_end = offsets[h + 1];
        let keys_in_layer = layer_end - layer_start;
        for i in 0..keys_in_layer {
            let k = i / node_keys;
            let j = i - k * node_keys;
            let mut child = k * fanout + j + 1;
            for _ in 0..(h - 1) {
                child *= fanout;
            }
            let leaf_index = child * node_keys;
            values[layer_start + i] = if leaf_index < len {
                values[leaf_index]
            } else {
                i32::MAX
            };
        }
    }
}

fn permute_internal_nodes_16(values: &mut [i32], offsets: &[usize]) {
    if offsets.len() < 3 {
        return;
    }
    for h in 1..offsets.len() - 1 {
        let layer_start = offsets[h];
        let layer_end = offsets[h + 1];
        for node_start in (layer_start..layer_end).step_by(NODE_KEYS) {
            permute16(&mut values[node_start..node_start + NODE_KEYS]);
        }
    }
}

fn permute_internal_nodes_32(values: &mut [i32], offsets: &[usize]) {
    if offsets.len() < 3 {
        return;
    }
    for h in 1..offsets.len() - 1 {
        let layer_start = offsets[h];
        let layer_end = offsets[h + 1];
        for node_start in (layer_start..layer_end).step_by(NODE_KEYS_32) {
            permute32(&mut values[node_start..node_start + NODE_KEYS_32]);
        }
    }
}

fn permute16(node: &mut [i32]) {
    debug_assert_eq!(node.len(), NODE_KEYS);
    let (left, right) = node.split_at_mut(8);
    left[4..8].swap_with_slice(&mut right[0..4]);
}

fn permute32(node: &mut [i32]) {
    debug_assert_eq!(node.len(), NODE_KEYS_32);
    let (left, right) = node.split_at_mut(16);
    left[8..16].swap_with_slice(&mut right[0..8]);
    permute16(left);
    permute16(right);
}

#[cfg(target_arch = "aarch64")]
mod neon_rank {
    use std::arch::aarch64::*;

    use super::{NODE_KEYS, NODE_KEYS_32};

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn rank16_direct(node: *const i32, needle: i32) -> usize {
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

            let mut mask = mask16_from_cmps(m0, m1, m2, m3);
            mask |= 1u32 << (NODE_KEYS as u32);
            mask.trailing_zeros() as usize
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn rank16_permuted(node: *const i32, needle: i32) -> usize {
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

            let mut mask = mask16_from_cmps(m0, m2, m1, m3);
            mask |= 1u32 << (NODE_KEYS as u32);
            mask.trailing_zeros() as usize
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn rank32_direct(node: *const i32, needle: i32) -> usize {
        unsafe {
            let x = vdupq_n_s32(needle);

            let v0 = vld1q_s32(node);
            let v1 = vld1q_s32(node.add(4));
            let v2 = vld1q_s32(node.add(8));
            let v3 = vld1q_s32(node.add(12));
            let v4 = vld1q_s32(node.add(16));
            let v5 = vld1q_s32(node.add(20));
            let v6 = vld1q_s32(node.add(24));
            let v7 = vld1q_s32(node.add(28));

            let m0 = vcgeq_s32(v0, x);
            let m1 = vcgeq_s32(v1, x);
            let m2 = vcgeq_s32(v2, x);
            let m3 = vcgeq_s32(v3, x);
            let m4 = vcgeq_s32(v4, x);
            let m5 = vcgeq_s32(v5, x);
            let m6 = vcgeq_s32(v6, x);
            let m7 = vcgeq_s32(v7, x);

            let lo = mask16_from_cmps(m0, m1, m2, m3) as u64;
            let hi = mask16_from_cmps(m4, m5, m6, m7) as u64;
            let mask = lo | (hi << 16) | (1u64 << (NODE_KEYS_32 as u64));
            mask.trailing_zeros() as usize
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn rank32_permuted(node: *const i32, needle: i32) -> usize {
        unsafe {
            let x = vdupq_n_s32(needle);

            let v0 = vld1q_s32(node);
            let v1 = vld1q_s32(node.add(4));
            let v2 = vld1q_s32(node.add(8));
            let v3 = vld1q_s32(node.add(12));
            let v4 = vld1q_s32(node.add(16));
            let v5 = vld1q_s32(node.add(20));
            let v6 = vld1q_s32(node.add(24));
            let v7 = vld1q_s32(node.add(28));

            let m0 = vcgeq_s32(v0, x);
            let m1 = vcgeq_s32(v1, x);
            let m2 = vcgeq_s32(v2, x);
            let m3 = vcgeq_s32(v3, x);
            let m4 = vcgeq_s32(v4, x);
            let m5 = vcgeq_s32(v5, x);
            let m6 = vcgeq_s32(v6, x);
            let m7 = vcgeq_s32(v7, x);

            let lo = mask16_from_cmps(m0, m2, m4, m6) as u64;
            let hi = mask16_from_cmps(m1, m3, m5, m7) as u64;
            let mask = lo | (hi << 16) | (1u64 << (NODE_KEYS_32 as u64));
            mask.trailing_zeros() as usize
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mask16_from_cmps(
        c0: uint32x4_t,
        c1: uint32x4_t,
        c2: uint32x4_t,
        c3: uint32x4_t,
    ) -> u32 {
        unsafe {
            let n0 = vmovn_u32(vshrq_n_u32(c0, 31));
            let n1 = vmovn_u32(vshrq_n_u32(c1, 31));
            let n2 = vmovn_u32(vshrq_n_u32(c2, 31));
            let n3 = vmovn_u32(vshrq_n_u32(c3, 31));

            let p0 = vcombine_u16(n0, n1);
            let p1 = vcombine_u16(n2, n3);
            let q0 = vmovn_u16(p0);
            let q1 = vmovn_u16(p1);
            let bytes = vcombine_u8(q0, q1);

            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), bytes);

            let lo = pack8(&out[0..8]);
            let hi = pack8(&out[8..16]);
            lo | (hi << 8)
        }
    }

    #[inline]
    fn pack8(bytes: &[u8]) -> u32 {
        (bytes[0] as u32)
            | ((bytes[1] as u32) << 1)
            | ((bytes[2] as u32) << 2)
            | ((bytes[3] as u32) << 3)
            | ((bytes[4] as u32) << 4)
            | ((bytes[5] as u32) << 5)
            | ((bytes[6] as u32) << 6)
            | ((bytes[7] as u32) << 7)
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
        let splus32 = SPlusTree32Layout::new(&values);

        for (idx, &value) in values.iter().enumerate() {
            assert_eq!(s_tree_search_scalar(&layout, value), Some(idx));
            assert_eq!(s_tree_search_neon(&layout, value), Some(idx));
            assert_eq!(s_plus_tree_search_scalar(&splus, value), Some(idx));
            assert_eq!(s_plus_tree_search_neon(&splus, value), Some(idx));
            assert_eq!(s_plus_tree32_search_scalar(&splus32, value), Some(idx));
            assert_eq!(s_plus_tree32_search_neon(&splus32, value), Some(idx));
        }

        assert_eq!(s_tree_search_scalar(&layout, 6), None);
        assert_eq!(s_plus_tree_search_neon(&splus, 6), None);
        assert_eq!(s_plus_tree32_search_neon(&splus32, 6), None);
    }

    proptest! {
        #[test]
        fn variants_match_std(mut values in prop::collection::vec(any::<i32>(), 0..256), needle in any::<i32>()) {
            values.sort();
            values.dedup();

            let expected = binary_search_std(&values, needle);
            let layout = STreeLayout::new(&values);
            let splus = SPlusTreeLayout::new(&values);
            let splus32 = SPlusTree32Layout::new(&values);

            prop_assert_eq!(s_tree_search_scalar(&layout, needle), expected);
            prop_assert_eq!(s_tree_search_neon(&layout, needle), expected);
            prop_assert_eq!(s_plus_tree_search_scalar(&splus, needle), expected);
            prop_assert_eq!(s_plus_tree_search_neon(&splus, needle), expected);
            prop_assert_eq!(s_plus_tree32_search_scalar(&splus32, needle), expected);
            prop_assert_eq!(s_plus_tree32_search_neon(&splus32, needle), expected);
        }
    }
}
