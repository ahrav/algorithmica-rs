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
//! | [`matmul_blocked`] | L2/L1 cache blocking + 4x4 kernel | Larger matrices |
//!
//! # Cache-aware blocking
//!
//! The blocked variant chooses tile sizes based on this machine's cache sizes
//! (L1 = 64KB, L2 = 4MB via `sysctl hw.l1dcachesize` / `hw.l2cachesize`).
//! With `BLOCK_L1 = 64`, the working set for A/B/C tiles is ~48KB
//! (3 x 64 x 64 x 4 bytes), which fits in L1.
//!
//! # References
//!
//! - [Matrix multiplication chapter](https://en.algorithmica.org/hpc/algorithms/matmul/)

/// L1 data cache size (bytes) on this machine.
pub const L1_BYTES: usize = 64 * 1024;
/// L2 cache size (bytes) on this machine.
pub const L2_BYTES: usize = 4 * 1024 * 1024;

/// L1 tile size used by [`matmul_blocked`].
pub const BLOCK_L1: usize = 64;
/// L2 tile size used by [`matmul_blocked`].
pub const BLOCK_L2: usize = 256;

const MICRO_TILE: usize = 4;

#[inline]
fn check_dims(a: &[f32], b: &[f32], c: &[f32], n: usize) {
    let len = n * n;
    debug_assert_eq!(a.len(), len);
    debug_assert_eq!(b.len(), len);
    debug_assert_eq!(c.len(), len);
}

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

/// Cache-blocked multiplication with a 4x4 micro-kernel.
pub fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    check_dims(a, b, c, n);
    c.fill(0.0);

    let l2 = BLOCK_L2.min(n);
    let l1 = BLOCK_L1.min(n);

    for ii2 in (0..n).step_by(l2) {
        let i2_max = (ii2 + l2).min(n);
        for kk2 in (0..n).step_by(l2) {
            let k2_max = (kk2 + l2).min(n);
            for jj2 in (0..n).step_by(l2) {
                let j2_max = (jj2 + l2).min(n);
                for ii1 in (ii2..i2_max).step_by(l1) {
                    let i1_max = (ii1 + l1).min(i2_max);
                    for kk1 in (kk2..k2_max).step_by(l1) {
                        let k1_max = (kk1 + l1).min(k2_max);
                        for jj1 in (jj2..j2_max).step_by(l1) {
                            let j1_max = (jj1 + l1).min(j2_max);

                            let mut i = ii1;
                            while i + MICRO_TILE <= i1_max {
                                let mut j = jj1;
                                while j + MICRO_TILE <= j1_max {
                                    kernel_4x4(a, b, c, n, i, j, kk1, k1_max);
                                    j += MICRO_TILE;
                                }

                                if j < j1_max {
                                    for ii in i..i + MICRO_TILE {
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

                                i += MICRO_TILE;
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
}
