//! Greatest Common Divisor (GCD) algorithms.
//!
//! This module provides two GCD implementations that illustrate the performance difference
//! between division-based and division-free algorithms on modern CPUs.
//!
//! # Algorithms
//!
//! - **Euclidean** ([`gcd_scalar`]): Uses repeated modulo operations. Simple but division
//!   is slow on most CPUs (10-90 cycles depending on operand sizes).
//!
//! - **Binary/Stein's** ([`gcd_binary`]): Replaces division with bit shifts and subtraction.
//!   While it performs more operations, each operation is much cheaper (1-3 cycles).
//!
//! # Performance
//!
//! On modern x86 and ARM CPUs, the binary algorithm is typically 2-4x faster than Euclidean
//! because `trailing_zeros()` compiles to a single `CTZ`/`TZCNT` instruction and subtraction
//! is vastly cheaper than division.
//!
//! # References
//!
//! - [GCD chapter](https://en.algorithmica.org/hpc/algorithms/gcd/)

/// Computes the GCD using the Euclidean algorithm.
///
/// Uses repeated modulo: `gcd(a, b) = gcd(b, a % b)` until `b = 0`.
/// Simple and correct, but division is expensive (~10-90 cycles per operation).
pub fn gcd_scalar(a: i64, b: i64) -> i64 {
    gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Computes the GCD using Stein's binary algorithm (division-free).
///
/// This algorithm exploits three identities to avoid expensive division:
///
/// 1. **gcd(2a, 2b) = 2 · gcd(a, b)** — factor out common powers of 2
/// 2. **gcd(2a, b) = gcd(a, b)** when b is odd — remove factors of 2 from one operand
/// 3. **gcd(a, b) = gcd(|a - b|, min(a, b))** — subtraction preserves GCD
///
/// The key insight is that `trailing_zeros()` compiles to a single CPU instruction
/// (`CTZ` on ARM, `TZCNT` on x86), making the "remove factors of 2" step extremely fast.
pub fn gcd_binary(a: i64, b: i64) -> i64 {
    let mut a = a.unsigned_abs();
    let mut b = b.unsigned_abs();
    if a == 0 {
        return b as i64;
    }
    if b == 0 {
        return a as i64;
    }

    // Factor out common powers of 2: gcd(2^k · a', 2^k · b') = 2^k · gcd(a', b')
    // We'll multiply the result by 2^shift at the end.
    let mut az = a.trailing_zeros();
    let bz = b.trailing_zeros();
    let shift = az.min(bz);
    b >>= bz; // b is now odd (all trailing zeros removed)

    while a != 0 {
        // Remove factors of 2 from a (doesn't affect GCD since b is odd)
        a >>= az;
        if a == b {
            return (b << shift) as i64;
        }
        // Apply identity: gcd(a, b) = gcd(|a - b|, min(a, b))
        // The difference of two odd numbers is always even, so next iteration
        // will have trailing zeros to remove—this is what makes the algorithm converge.
        let (min_ab, diff) = if a < b { (a, b - a) } else { (b, a - b) };
        az = diff.trailing_zeros(); // single CTZ instruction
        b = min_ab;
        a = diff;
    }

    (b << shift) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// gcd(a, b) == gcd(b, a)
        #[test]
        fn commutative(a in any::<i64>(), b in any::<i64>()) {
            prop_assert_eq!(gcd_scalar(a, b), gcd_scalar(b, a));
        }

        /// gcd(a, b) divides both a and b (for non-zero gcd)
        #[test]
        fn divisibility(a in any::<i64>(), b in any::<i64>()) {
            let g = gcd_scalar(a, b);
            if g != 0 {
                prop_assert_eq!(a % g, 0);
                prop_assert_eq!(b % g, 0);
            }
        }

        /// Result is non-negative for non-negative inputs
        #[test]
        fn non_negative(a in 0i64..=i64::MAX, b in 0i64..=i64::MAX) {
            prop_assert!(gcd_scalar(a, b) >= 0);
        }

        /// gcd(a, 0) == abs(a) and gcd(0, b) == abs(b)
        #[test]
        fn identity_with_zero(a in any::<i64>()) {
            let expected = a.unsigned_abs() as i64;
            prop_assert_eq!(gcd_scalar(a, 0), expected);
            prop_assert_eq!(gcd_scalar(0, a), expected);
        }

        /// gcd(a, a) == abs(a) for non-negative a
        #[test]
        fn idempotent(a in 0i64..=i64::MAX) {
            prop_assert_eq!(gcd_scalar(a, a), a);
        }

        /// binary gcd matches scalar gcd for non-negative inputs
        #[test]
        fn binary_matches_scalar_non_negative(a in 0i64..=i64::MAX, b in 0i64..=i64::MAX) {
            prop_assert_eq!(gcd_binary(a, b), gcd_scalar(a, b));
        }
    }

    #[test]
    fn binary_known_cases() {
        let cases = [
            (0, 0, 0),
            (0, 7, 7),
            (7, 0, 7),
            (1, 1, 1),
            (54, 24, 6),
            (48, 18, 6),
            (1024, 64, 64),
            (270, 192, 6),
            (7, 13, 1),
        ];

        for (a, b, expected) in cases {
            assert_eq!(gcd_binary(a, b), expected, "a={a} b={b}");
        }
    }

    #[test]
    fn binary_matches_scalar_small_grid() {
        for a in -512i64..=512 {
            for b in -512i64..=512 {
                assert_eq!(gcd_binary(a, b), gcd_scalar(a, b), "a={a} b={b}");
            }
        }
    }
}
