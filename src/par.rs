//! Conditional parallelism macros.
//!
//! When the `parallel` feature is enabled, these macros expand to rayon's
//! parallel iterators. Without the feature, they expand to standard iterators.
//! This allows algorithm code to use `maybe_par_iter!(data)` and get parallelism
//! on native targets while remaining single-threaded for WASM.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

/// Parallel or sequential immutable iterator over a slice.
#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_iter {
    ($slice:expr) => {
        $slice.par_iter()
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_iter {
    ($slice:expr) => {
        $slice.iter()
    };
}

/// Parallel or sequential mutable iterator over a slice.
#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_iter_mut {
    ($slice:expr) => {
        $slice.par_iter_mut()
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_iter_mut {
    ($slice:expr) => {
        $slice.iter_mut()
    };
}

/// Parallel or sequential chunks iterator.
#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_chunks_mut {
    ($slice:expr, $chunk_size:expr) => {
        $slice.par_chunks_mut($chunk_size)
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_chunks_mut {
    ($slice:expr, $chunk_size:expr) => {
        $slice.chunks_mut($chunk_size)
    };
}

/// Parallel or sequential (immutable) chunks iterator.
///
/// Useful for deterministic parallel reductions: map each fixed-size chunk to a
/// sequential partial sum, then combine the partials in index order. Because the
/// chunk boundaries and combination order are fixed, the result is independent of
/// the thread count (bit-for-bit reproducible).
#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_chunks {
    ($slice:expr, $chunk_size:expr) => {
        $slice.par_chunks($chunk_size)
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_chunks {
    ($slice:expr, $chunk_size:expr) => {
        $slice.chunks($chunk_size)
    };
}
