//! Matrix multiplication (C = A x B) algorithms.
//!
//! All matrices are `n x n` in row-major order.
//!
//! # Strategies
//!
//! | Function | Strategy | Best For |
//! |----------|----------|----------|
//! | [`matmul_baseline`] | Naive i-j-k loops | Tiny matrices, clarity |
//! | [`matmul_transposed`] | Transpose B for contiguous access | Moderate sizes |
//! | [`matmul_ikj`] | Loop reordering (i-k-j) | Better cache reuse |
//! | [`matmul_register_blocked_2x2`] | 2x2 register kernel | Small/medium sizes |
//! | [`matmul_blocked`] | L3/L2/L1 cache blocking + 4x4 kernel | Larger matrices |
//! | [`matmul_neon_blocked`] | Packed B + NEON 4x8 kernel | aarch64 only |
//!
//! # Cache-aware blocking
//!
//! The blocked variant uses fixed tile sizes (s3/s2/s1) that match the reference
//! macro-kernel structure. These constants are not detected at runtime; adjust
//! them for your CPU.
//! We use three levels of blocking:
//! - B columns (j) in L3: `BLOCK_L3`
//! - A rows (i) in L2: `BLOCK_L2`
//! - B rows (k) in L1: `BLOCK_L1`
//!
//! With the current sizes, the working set is:
//!   B panel: 240 * 64 * 4 = 61,440 bytes (fits L1)
//!   A panel: 120 * 240 * 4 = 115,200 bytes
//!   C tile: 120 * 64 * 4 = 30,720 bytes
//!
//! # References
//!
//! - [Matrix multiplication chapter](https://en.algorithmica.org/hpc/algorithms/matmul/)

/// L1 data cache size (bytes) assumed for tuning.
///
/// Fixed constant; not queried at runtime.
pub const L1_BYTES: usize = 64 * 1024;

/// L2 cache size (bytes) assumed for tuning.
///
/// Fixed constant; not queried at runtime.
pub const L2_BYTES: usize = 4 * 1024 * 1024;

/// L3 tile size used by [`matmul_blocked`] for B columns (j-block).
///
/// # Derivation
///
/// The reference uses 64, which keeps `BLOCK_L1 * BLOCK_L3 * 4` under L1.
pub const BLOCK_L3: usize = 64;

/// L2 tile size used by [`matmul_blocked`] for A rows (i-block).
///
/// # Derivation
///
/// Chosen to match the reference (multiple of MICRO_TILE).
pub const BLOCK_L2: usize = 120;

/// L1 tile size used by [`matmul_blocked`] for B rows (k-block).
///
/// # Derivation
///
/// We want the B panel (k-block x j-block) to fit in L1:
/// ```text
/// BLOCK_L1 * BLOCK_L3 * sizeof(f32) <= L1_BYTES
/// 240 * 64 * 4 = 61,440 bytes (fits in 64 KB)
/// ```
///
/// The reference maximizes s1 (k-block) under this constraint.
pub const BLOCK_L1: usize = 240;

/// Tile sizes used by the NEON packed-B path on aarch64.
///
/// The packed kernel has different reuse characteristics, so it uses separate
/// block sizes tuned for the current arm64 target.
pub const NEON_BLOCK_L2: usize = 256;
pub const NEON_BLOCK_L1: usize = 64;

/// Micro-kernel tile dimension for the scalar 4x4 kernel.
///
/// This is the portable scalar fallback used by [`matmul_blocked`]. The reference
/// uses a wider vector kernel; on arm64 we use a NEON 4x8 kernel in
/// [`matmul_neon_blocked`] instead.
const MICRO_TILE: usize = 4;

#[inline]
fn check_dims(a: &[f32], b: &[f32], c: &[f32], n: usize) {
    let len = n * n;
    debug_assert_eq!(a.len(), len);
    debug_assert_eq!(b.len(), len);
    debug_assert_eq!(c.len(), len);
}

/// Transpose matrix B from row-major to column-major layout.
///
/// In the baseline i-j-k order, accessing `B[k][j]` in the inner loop is stride-n.
/// Transposing makes the inner loop stride-1. This primarily enables SIMD-friendly
/// access and more predictable prefetching; in scalar code the speedup can be small
/// and depends on size and cache behavior.
#[inline]
fn transpose(b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        let row = i * n;
        for j in 0..n {
            out[j * n + i] = b[row + j];
        }
    }
    out
}

/// Baseline matrix multiplication (i-j-k order).
pub fn matmul_baseline(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    for i in 0..n {
        let a_row = i * n;
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[a_row + k] * b[k * n + j];
            }
            c[a_row + j] = sum;
        }
    }
}

/// Multiply with B transposed to make the inner loop contiguous.
pub fn matmul_transposed(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    let b_t = transpose(b, n);
    for i in 0..n {
        let a_row = i * n;
        for j in 0..n {
            let b_row = j * n;
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[a_row + k] * b_t[b_row + k];
            }
            c[a_row + j] = sum;
        }
    }
}

/// Loop-reordered multiplication (i-k-j order) for better cache locality.
///
/// The baseline i-j-k order has poor locality in the innermost loop:
/// ```text
/// for k in 0..n:
///     sum += A[i][k] * B[k][j]  // B access is stride-n (cache miss per iteration)
/// ```
///
/// Reordering to i-k-j makes the inner loop access both B and C with stride-1:
/// ```text
/// for j in 0..n:
///     C[i][j] += a_ik * B[k][j]  // B and C are both stride-1 (sequential)
/// ```
///
/// This changes B access from strided (stride-n) to sequential (stride-1), improving
/// cache line utilization and prefetching. The tradeoff is that C must be initialized
/// to zero and accumulated into (vs. computed in one pass).
pub fn matmul_ikj(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    c.fill(0.0);
    for i in 0..n {
        let row = i * n;
        for k in 0..n {
            let a_ik = a[row + k];
            let b_row = k * n;
            for j in 0..n {
                c[row + j] += a_ik * b[b_row + j];
            }
        }
    }
}

/// 2x2 register-blocked kernel (from the register reuse section).
pub fn matmul_register_blocked_2x2(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    c.fill(0.0);

    let mut i = 0;
    while i + 1 < n {
        let mut j = 0;
        while j + 1 < n {
            let mut c00 = 0.0f32;
            let mut c01 = 0.0f32;
            let mut c10 = 0.0f32;
            let mut c11 = 0.0f32;

            for k in 0..n {
                let a0 = a[i * n + k];
                let a1 = a[(i + 1) * n + k];
                let b0 = b[k * n + j];
                let b1 = b[k * n + j + 1];

                c00 += a0 * b0;
                c01 += a0 * b1;
                c10 += a1 * b0;
                c11 += a1 * b1;
            }

            c[i * n + j] = c00;
            c[i * n + j + 1] = c01;
            c[(i + 1) * n + j] = c10;
            c[(i + 1) * n + j + 1] = c11;

            j += 2;
        }

        if j < n {
            let mut c0 = 0.0f32;
            let mut c1 = 0.0f32;
            for k in 0..n {
                let a0 = a[i * n + k];
                let a1 = a[(i + 1) * n + k];
                let b0 = b[k * n + j];
                c0 += a0 * b0;
                c1 += a1 * b0;
            }
            c[i * n + j] = c0;
            c[(i + 1) * n + j] = c1;
        }

        i += 2;
    }

    if i < n {
        let row = i * n;
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[row + k] * b[k * n + j];
            }
            c[row + j] = sum;
        }
    }
}

/// 4x4 scalar micro-kernel: computes C[i..i+4, j..j+4] += A[i..i+4, k0..k1] x B[k0..k1, j..j+4].
///
/// # Register Blocking Strategy
///
/// This kernel uses **register blocking** to maximize data reuse:
/// - **16 accumulators** (c00..c33): Hold the 4x4 C tile across all k iterations.
/// - **4 A operands** (a0..a3): One column of the A tile, reloaded each k.
/// - **4 B operands** (b0..b3): One row of the B tile, reloaded each k.
///
/// The compiler typically keeps these scalars in registers when possible, but
/// this remains a scalar kernel (no explicit SIMD).
///
/// # Compute-to-Load Ratio
///
/// Each iteration of the k-loop performs:
/// - 8 loads (4 from A row, 4 from B row)
/// - 16 mul-adds (each A value multiplies all 4 B values)
///
/// This yields a **2:1 compute-to-load ratio** (16 mul-adds / 8 loads). Larger
/// tiles would improve this ratio but increase register pressure and code size.
///
/// # Memory Access Pattern
///
/// - **A access**: Row i at indices `[i*n + k]` - stride-1 within each row (good).
/// - **B access**: Row k at indices `[k*n + j..j+3]` - contiguous within the row,
///   but stride-n across rows.
///
/// The blocked caller ensures that the A and B tiles fit in L1 cache, so these
/// accesses stay mostly in cache even though B's pattern is strided.
#[allow(clippy::too_many_arguments)]
#[inline]
fn kernel_4x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    i: usize,
    j: usize,
    k0: usize,
    k1: usize,
) {
    let r0 = i * n;
    let r1 = (i + 1) * n;
    let r2 = (i + 2) * n;
    let r3 = (i + 3) * n;

    let mut c00 = c[r0 + j];
    let mut c01 = c[r0 + j + 1];
    let mut c02 = c[r0 + j + 2];
    let mut c03 = c[r0 + j + 3];

    let mut c10 = c[r1 + j];
    let mut c11 = c[r1 + j + 1];
    let mut c12 = c[r1 + j + 2];
    let mut c13 = c[r1 + j + 3];

    let mut c20 = c[r2 + j];
    let mut c21 = c[r2 + j + 1];
    let mut c22 = c[r2 + j + 2];
    let mut c23 = c[r2 + j + 3];

    let mut c30 = c[r3 + j];
    let mut c31 = c[r3 + j + 1];
    let mut c32 = c[r3 + j + 2];
    let mut c33 = c[r3 + j + 3];

    for k in k0..k1 {
        let a0 = a[r0 + k];
        let a1 = a[r1 + k];
        let a2 = a[r2 + k];
        let a3 = a[r3 + k];

        let b_row = k * n + j;
        let b0 = b[b_row];
        let b1 = b[b_row + 1];
        let b2 = b[b_row + 2];
        let b3 = b[b_row + 3];

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;

        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;

        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;

        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
    }

    c[r0 + j] = c00;
    c[r0 + j + 1] = c01;
    c[r0 + j + 2] = c02;
    c[r0 + j + 3] = c03;

    c[r1 + j] = c10;
    c[r1 + j + 1] = c11;
    c[r1 + j + 2] = c12;
    c[r1 + j + 3] = c13;

    c[r2 + j] = c20;
    c[r2 + j + 1] = c21;
    c[r2 + j + 2] = c22;
    c[r2 + j + 3] = c23;

    c[r3 + j] = c30;
    c[r3 + j + 1] = c31;
    c[r3 + j + 2] = c32;
    c[r3 + j + 3] = c33;
}

/// Cache-blocked multiplication with a 4x4 scalar micro-kernel.
///
/// # Three-Level Blocking (B-First)
///
/// Following the reference macro-kernel, we use three cache levels:
///
/// ```text
/// B columns (j) in L3 -> A rows (i) in L2 -> B rows (k) in L1 -> 4x4 micro-kernel
/// ```
///
/// # Loop Ordering (j -> i -> k)
///
/// Within each tile, loops are ordered j-i-k because:
/// - **B columns** stay resident while iterating A rows.
/// - **A rows** are reused across the inner k loop.
/// - **B rows** are streamed in k-sized panels that fit in L1.
/// - **C** accumulates in-place.
///
/// # Fringe Handling
///
/// When matrix dimensions aren't divisible by [`MICRO_TILE`], the residual rows/columns
/// are handled by scalar cleanup loops. These fringe cases use a simple i-k-j loop,
/// trading off performance for correctness on non-aligned boundaries.
pub fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    c.fill(0.0);

    let l3 = BLOCK_L3.min(n);
    let l2 = BLOCK_L2.min(n);
    let l1 = BLOCK_L1.min(n);

    for jj3 in (0..n).step_by(l3) {
        let j3_max = (jj3 + l3).min(n);
        for ii2 in (0..n).step_by(l2) {
            let i2_max = (ii2 + l2).min(n);
            for kk1 in (0..n).step_by(l1) {
                let k1_max = (kk1 + l1).min(n);

                let mut i = ii2;
                while i + MICRO_TILE <= i2_max {
                    let mut j = jj3;
                    while j + MICRO_TILE <= j3_max {
                        kernel_4x4(a, b, c, n, i, j, kk1, k1_max);
                        j += MICRO_TILE;
                    }

                    if j < j3_max {
                        for ii in i..i + MICRO_TILE {
                            let a_row = ii * n;
                            let c_row = ii * n;
                            for k in kk1..k1_max {
                                let a_ik = a[a_row + k];
                                let b_row = k * n;
                                for jj in j..j3_max {
                                    c[c_row + jj] += a_ik * b[b_row + jj];
                                }
                            }
                        }
                    }

                    i += MICRO_TILE;
                }

                if i < i2_max {
                    for ii in i..i2_max {
                        let a_row = ii * n;
                        let c_row = ii * n;
                        for k in kk1..k1_max {
                            let a_ik = a[a_row + k];
                            let b_row = k * n;
                            for jj in jj3..j3_max {
                                c[c_row + jj] += a_ik * b[b_row + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Packed-B blocked multiplication with a NEON 4×8 micro-kernel (aarch64 only).
///
/// The scalar [`matmul_blocked`] accesses B with a strided pattern: each k-iteration
/// jumps by `n` elements to reach the next row. This causes three problems:
///
/// 1. **Cache line waste**: Each 64-byte cache line (16 floats) delivers only 4-8
///    useful floats when accessing non-contiguous columns.
/// 2. **TLB thrashing**: Strided access touches many pages, exhausting TLB entries.
/// 3. **Prefetcher failure**: Hardware prefetchers detect sequential patterns; strided
///    access appears random and defeats prefetching.
///
/// Packing B into a contiguous buffer converts strided access to sequential access,
/// solving all three issues. The packing cost is amortized because the packed B tile
/// is reused across all i-iterations.
///
/// Unlike [`matmul_blocked`] (j -> i -> k), this function iterates k -> j -> i so
/// it can pack B once per (k, j) tile and reuse it across all i blocks:
///
/// ```text
/// for each (k-block, j-block):
///     Pack B[k-block, j-block] once
///     for each i-block:
///         kernel_4x8(A, packed_B, C)  // Reuses packed B across all i
/// ```
///
/// This restructuring ensures the packing operation happens once per (k, j)
/// tile, then the packed data is reused across all row blocks of A.
///
/// NOTE: This path uses `NEON_BLOCK_L1`/`NEON_BLOCK_L2` (two-level blocking) to
/// tune for arm64 and the packed-B kernel. It intentionally diverges from the
/// reference 3-level macro-kernel used by [`matmul_blocked`].
#[cfg(target_arch = "aarch64")]
pub fn matmul_neon_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    c.fill(0.0);
    // SAFETY: aarch64 guarantees NEON availability.
    unsafe { aarch64_neon::matmul_neon_blocked(a, b, c, n) }
}

/// Fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_neon_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    matmul_blocked(a, b, c, n);
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod aarch64_neon {
    use std::arch::aarch64::*;

    use super::{NEON_BLOCK_L1, NEON_BLOCK_L2};

    /// Number of rows in the NEON micro-kernel tile.
    ///
    /// Fixed at 4 because we process 4 rows of A, each broadcast to multiply against
    /// the B vectors. This matches the scalar kernel's row count.
    const MICRO_M: usize = 4;

    /// Number of columns in the NEON micro-kernel tile.
    ///
    /// NEON has 32 vector registers (128-bit each, holding 4×f32). The 4×8 kernel uses:
    /// - **8 accumulator vectors** (c00, c01, c10, c11, c20, c21, c30, c31): 4 rows × 2 vectors/row
    /// - **2 B vectors** (b0, b1): One 8-element row of packed B
    /// - **4 broadcast A scalars** (a0v, a1v, a2v, a3v): Duplicated into vectors
    ///
    /// Total: 14 vector registers, well within the 32 available. The 4×8 shape doubles
    /// throughput over 4×4 by processing 8 columns per iteration while maintaining
    /// the same A-row reuse pattern.
    const MICRO_N: usize = 8;

    /// 4×8 NEON micro-kernel: computes C[i..i+4, j..j+8] += A × B using SIMD.
    ///
    /// # NEON SIMD Strategy
    ///
    /// This kernel exploits ARM NEON's 128-bit vectors (4×f32) to process 8 output
    /// columns per iteration:
    ///
    /// ```text
    /// For each k in 0..k_len:
    ///     Load B row: b0 = B[k, j..j+4], b1 = B[k, j+4..j+8]  (2 vector loads)
    ///     For each row r in 0..4:
    ///         a_rv = broadcast(A[r, k])           (scalar → vector)
    ///         C[r, j..j+4]   += a_rv * b0         (vfmaq_f32)
    ///         C[r, j+4..j+8] += a_rv * b1         (vfmaq_f32)
    /// ```
    ///
    /// Each k-iteration performs 8 FMA operations across 4 rows = 32 FMAs, using only
    /// 2 B-vector loads (amortized across all 4 A rows).
    ///
    /// # Packed B Layout
    ///
    /// The `b_packed` pointer references a contiguous buffer where B tile rows are
    /// stored sequentially: `[row0: j..j+n_block][row1: j..j+n_block]...`
    ///
    /// The `b_stride` parameter indicates the number of elements per packed row
    /// (typically the J-block width). This layout ensures sequential memory access
    /// for the B vectors, enabling efficient prefetching.
    ///
    /// # Safety
    ///
    /// - Caller must ensure `i + 4 <= n` and `j + 8 <= n` (no bounds checking).
    /// - `b_packed` must point to a valid packed B buffer with at least `k_len * b_stride` elements.
    /// - Must be called with NEON feature enabled (guaranteed on aarch64).
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon")]
    unsafe fn kernel_4x8(
        a: &[f32],
        b_packed: *const f32,
        c: &mut [f32],
        n: usize,
        i: usize,
        j: usize,
        k0: usize,
        k_len: usize,
        b_stride: usize,
    ) {
        let a_ptr = a.as_ptr();
        let c_ptr = c.as_mut_ptr();

        let r0 = i * n + j;
        let r1 = (i + 1) * n + j;
        let r2 = (i + 2) * n + j;
        let r3 = (i + 3) * n + j;

        let mut c00 = vld1q_f32(c_ptr.add(r0));
        let mut c01 = vld1q_f32(c_ptr.add(r0 + 4));
        let mut c10 = vld1q_f32(c_ptr.add(r1));
        let mut c11 = vld1q_f32(c_ptr.add(r1 + 4));
        let mut c20 = vld1q_f32(c_ptr.add(r2));
        let mut c21 = vld1q_f32(c_ptr.add(r2 + 4));
        let mut c30 = vld1q_f32(c_ptr.add(r3));
        let mut c31 = vld1q_f32(c_ptr.add(r3 + 4));

        let a_row0 = i * n;
        let a_row1 = (i + 1) * n;
        let a_row2 = (i + 2) * n;
        let a_row3 = (i + 3) * n;

        for kk in 0..k_len {
            let b_row = b_packed.add(kk * b_stride);
            let b0 = vld1q_f32(b_row);
            let b1 = vld1q_f32(b_row.add(4));

            let a0 = *a_ptr.add(a_row0 + k0 + kk);
            let a1 = *a_ptr.add(a_row1 + k0 + kk);
            let a2 = *a_ptr.add(a_row2 + k0 + kk);
            let a3 = *a_ptr.add(a_row3 + k0 + kk);

            let a0v = vdupq_n_f32(a0);
            let a1v = vdupq_n_f32(a1);
            let a2v = vdupq_n_f32(a2);
            let a3v = vdupq_n_f32(a3);

            c00 = vfmaq_f32(c00, b0, a0v);
            c01 = vfmaq_f32(c01, b1, a0v);
            c10 = vfmaq_f32(c10, b0, a1v);
            c11 = vfmaq_f32(c11, b1, a1v);
            c20 = vfmaq_f32(c20, b0, a2v);
            c21 = vfmaq_f32(c21, b1, a2v);
            c30 = vfmaq_f32(c30, b0, a3v);
            c31 = vfmaq_f32(c31, b1, a3v);
        }

        vst1q_f32(c_ptr.add(r0), c00);
        vst1q_f32(c_ptr.add(r0 + 4), c01);
        vst1q_f32(c_ptr.add(r1), c10);
        vst1q_f32(c_ptr.add(r1 + 4), c11);
        vst1q_f32(c_ptr.add(r2), c20);
        vst1q_f32(c_ptr.add(r2 + 4), c21);
        vst1q_f32(c_ptr.add(r3), c30);
        vst1q_f32(c_ptr.add(r3 + 4), c31);
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn matmul_neon_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
        if n == 0 {
            return;
        }

        let l2 = NEON_BLOCK_L2.min(n);
        let l1 = NEON_BLOCK_L1.min(n);
        let mut packed_b: Vec<f32> = vec![0.0f32; l1 * l1];

        for kk2 in (0..n).step_by(l2) {
            let k2_max = (kk2 + l2).min(n);
            for jj2 in (0..n).step_by(l2) {
                let j2_max = (jj2 + l2).min(n);

                for kk1 in (kk2..k2_max).step_by(l1) {
                    let k1_max = (kk1 + l1).min(k2_max);
                    let k_block = k1_max - kk1;

                    for jj1 in (jj2..j2_max).step_by(l1) {
                        let j1_max = (jj1 + l1).min(j2_max);
                        let n_block = j1_max - jj1;
                        let required = k_block * n_block;
                        if packed_b.len() < required {
                            packed_b.resize(required, 0.0);
                        }

                        // Pack B tile: copy B[kk1..kk1+k_block, jj1..j1_max] into contiguous buffer.
                        // Layout transforms from strided (stride = n) to packed (stride = n_block):
                        //   Original B:  row k at offset (kk1+k)*n + jj1, next row at +n
                        //   Packed B:    row k at offset k*n_block, next row at +n_block
                        // This makes kernel_4x8's B access sequential instead of strided.
                        for k in 0..k_block {
                            let src = &b[(kk1 + k) * n + jj1..(kk1 + k) * n + j1_max];
                            let dst = &mut packed_b[k * n_block..k * n_block + n_block];
                            dst.copy_from_slice(src);
                        }

                        for ii2 in (0..n).step_by(l2) {
                            let i2_max = (ii2 + l2).min(n);
                            for ii1 in (ii2..i2_max).step_by(l1) {
                                let i1_max = (ii1 + l1).min(i2_max);
                                let mut i = ii1;
                                while i + MICRO_M <= i1_max {
                                    let mut j = jj1;
                                    while j + MICRO_N <= j1_max {
                                        let b_ptr = packed_b.as_ptr().add(j - jj1);
                                        kernel_4x8(a, b_ptr, c, n, i, j, kk1, k_block, n_block);
                                        j += MICRO_N;
                                    }

                                    if j < j1_max {
                                        for ii in i..i + MICRO_M {
                                            let a_row = ii * n;
                                            let c_row = ii * n;
                                            for k in kk1..k1_max {
                                                let a_ik = a[a_row + k];
                                                let b_row = k * n;
                                                for jj in j..j1_max {
                                                    c[c_row + jj] += a_ik * b[b_row + jj];
                                                }
                                            }
                                        }
                                    }
                                    i += MICRO_M;
                                }

                                if i < i1_max {
                                    for ii in i..i1_max {
                                        let a_row = ii * n;
                                        let c_row = ii * n;
                                        for k in kk1..k1_max {
                                            let a_ik = a[a_row + k];
                                            let b_row = k * n;
                                            for jj in jj1..j1_max {
                                                c[c_row + jj] += a_ik * b[b_row + jj];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_small_matrix(n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                out.push(((i * 3 + j * 5) % 7) as f32);
            }
        }
        out
    }

    fn assert_matrix_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(*x, *y, "idx={idx} x={x} y={y}");
        }
    }

    #[test]
    fn baseline_identity() {
        let n = 4;
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as f32;
        }
        let b = make_small_matrix(n);
        let mut c = vec![0.0f32; n * n];
        matmul_baseline(&a, &b, &mut c, n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(c[i * n + j], a[i * n + i] * b[i * n + j]);
            }
        }
    }

    #[test]
    fn variants_match_baseline() {
        let n = 5;
        let a = make_small_matrix(n);
        let b = make_small_matrix(n);

        let mut expected = vec![0.0f32; n * n];
        matmul_baseline(&a, &b, &mut expected, n);

        let mut out = vec![0.0f32; n * n];
        matmul_transposed(&a, &b, &mut out, n);
        assert_matrix_eq(&expected, &out);

        matmul_ikj(&a, &b, &mut out, n);
        assert_matrix_eq(&expected, &out);

        matmul_register_blocked_2x2(&a, &b, &mut out, n);
        assert_matrix_eq(&expected, &out);

        matmul_blocked(&a, &b, &mut out, n);
        assert_matrix_eq(&expected, &out);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_baseline() {
        let n = 6;
        let a = make_small_matrix(n);
        let b = make_small_matrix(n);
        let mut expected = vec![0.0f32; n * n];
        let mut out = vec![0.0f32; n * n];

        matmul_baseline(&a, &b, &mut expected, n);
        matmul_neon_blocked(&a, &b, &mut out, n);
        assert_matrix_eq(&expected, &out);
    }
}
