pub fn gcd_scalar(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd_scalar(b, a % b) }
}

pub fn gcd_binary(a: i64, b: i64) -> i64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    if a < 0 || b < 0 {
        return gcd_scalar(a, b);
    }

    let mut a = a as u64;
    let mut b = b as u64;

    let mut az = a.trailing_zeros();
    let bz = b.trailing_zeros();
    let shift = az.min(bz);
    b >>= bz;

    while a != 0 {
        a >>= az;
        if a == b {
            return (b << shift) as i64;
        }
        let (min_ab, diff) = if a < b { (a, b - a) } else { (b, a - b) };
        az = diff.trailing_zeros();
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
            prop_assert_eq!(gcd_scalar(a, 0), a);
            prop_assert_eq!(gcd_scalar(0, a), a);
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
