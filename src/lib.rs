//! Rust implementations of algorithms from [Algorithms for Modern Hardware].
//!
//! This crate explores high-performance algorithm design with a focus on understanding
//! *why* certain optimizations work, not just *what* the code does. Each module includes
//! educational documentation explaining the performance implications of different approaches.
//!
//! # Algorithms
//!
//! - **GCD** ([`gcd_scalar`], [`gcd_binary`]) — Euclidean and binary (Stein's) algorithms
//! - **Argmin** ([`argmin_scalar`], [`argmin_simd_filtered`], etc.) — Finding the minimum index
//! - **Prefix Sum** ([`prefix_sum_scalar`], [`prefix_sum`]) — Inclusive prefix sums with SIMD
//!
//! # References
//!
//! - [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)
//!
//! [Algorithms for Modern Hardware]: https://en.algorithmica.org/hpc/

mod argmin;
mod gcd;
mod prefix_sum;

pub use argmin::*;
pub use gcd::*;
pub use prefix_sum::*;
