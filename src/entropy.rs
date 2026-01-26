//! Shannon entropy over byte slices.
//!
//! Computes the entropy `H = -∑ p_i log2(p_i)` for the distribution of byte values.
//! The implementation is the canonical histogram-based approach:
//!
//! 1) Count the 256 byte frequencies.
//! 2) Convert counts to probabilities and accumulate `-p log2(p)`.
//!
//! This runs in O(n + 256) time and uses a fixed 256-element histogram.
//!
//! # References
//!
//! - C.E. Shannon, "A Mathematical Theory of Communication" (1948)

/// Computes Shannon entropy (in bits) of a byte slice.
///
/// Returns `0.0` for empty inputs.
pub fn shannon_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }

    let len = bytes.len() as f64;
    let inv_len = 1.0 / len;
    let mut entropy = 0.0f64;

    for count in counts {
        if count == 0 {
            continue;
        }
        let p = (count as f64) * inv_len;
        entropy -= p * p.log2();
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference_entropy(bytes: &[u8]) -> f64 {
        if bytes.is_empty() {
            return 0.0;
        }

        let mut counts = [0usize; 256];
        for &b in bytes {
            counts[b as usize] += 1;
        }

        let len = bytes.len() as f64;
        let log_len = len.log2();
        let mut sum = 0.0f64;
        for &count in &counts {
            if count != 0 {
                let c = count as f64;
                sum += c * c.log2();
            }
        }

        log_len - sum / len
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        let scale = 1.0 + a.abs().max(b.abs());
        (a - b).abs() <= 1.0e-12 * scale
    }

    proptest! {
        #[test]
        fn entropy_matches_reference(values in proptest::collection::vec(any::<u8>(), 0..=4096)) {
            let actual = shannon_entropy(&values);
            let expected = reference_entropy(&values);
            prop_assert!(approx_eq(actual, expected));
        }

        #[test]
        fn entropy_bounds(values in proptest::collection::vec(any::<u8>(), 0..=4096)) {
            let entropy = shannon_entropy(&values);
            if values.is_empty() {
                prop_assert!(approx_eq(entropy, 0.0));
            } else {
                let mut seen = [false; 256];
                for &b in &values {
                    seen[b as usize] = true;
                }
                let symbols = seen.iter().filter(|&&v| v).count();
                let max_entropy = (symbols as f64).log2();
                prop_assert!(entropy >= -1.0e-12);
                prop_assert!(entropy <= max_entropy + 1.0e-12);
            }
        }

        #[test]
        fn entropy_invariant_under_repeat(values in proptest::collection::vec(any::<u8>(), 0..=4096)) {
            let base = shannon_entropy(&values);
            let mut doubled = values.clone();
            doubled.extend_from_slice(&values);
            let doubled_entropy = shannon_entropy(&doubled);
            prop_assert!(approx_eq(base, doubled_entropy));
        }
    }

    #[test]
    fn entropy_basic_cases() {
        assert!(approx_eq(shannon_entropy(&[]), 0.0));
        assert!(approx_eq(shannon_entropy(&[42]), 0.0));
        assert!(approx_eq(shannon_entropy(&[0, 1]), 1.0));
        assert!(approx_eq(shannon_entropy(&[0, 1, 2, 3]), 2.0));

        let skewed = [0u8, 0, 0, 1];
        let expected = -(0.75f64 * 0.75f64.log2() + 0.25f64 * 0.25f64.log2());
        assert!(approx_eq(shannon_entropy(&skewed), expected));
    }

    #[test]
    fn entropy_uniform_256() {
        let mut values = Vec::with_capacity(256);
        for i in 0u16..=255 {
            values.push(i as u8);
        }
        assert!(approx_eq(shannon_entropy(&values), 8.0));
    }
}
