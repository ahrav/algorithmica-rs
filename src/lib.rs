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
//! - **Binary Search** ([`binary_search_std`], [`binary_search_eytzinger`], etc.) — Branchless and layout-aware search
//! - **S-tree / S+ tree** ([`s_tree_search_neon`], [`s_plus_tree_search_neon`]) — Implicit B-tree/B+ tree layouts (B=16/32)
//! - **Matrix Multiplication** ([`matmul_baseline`], [`matmul_blocked`], etc.) — Cache-aware GEMM
//! - **Prefix Sum** ([`prefix_sum_scalar`], [`prefix_sum`]) — Inclusive prefix sums with SIMD
//! - **Shannon Entropy** ([`shannon_entropy`], [`entropy_interleaved`], [`entropy_interleaved_neon`]) — Byte-frequency entropy in bits
//!
//! # References
//!
//! - [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)
//!
//! [Algorithms for Modern Hardware]: https://en.algorithmica.org/hpc/

mod argmin;
mod binary_search;
mod entropy;
mod gcd;
mod matmul;
mod prefix_sum;
mod s_tree;

pub use argmin::*;
pub use binary_search::*;
pub use entropy::*;
pub use gcd::*;
pub use matmul::*;
pub use prefix_sum::*;
pub use s_tree::*;
