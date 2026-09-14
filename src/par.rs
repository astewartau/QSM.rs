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

/// Parallel or sequential `map` with a reusable mutable state produced by
/// `$init`.
///
/// Lets a hot per-item body hoist its scratch buffers out of the loop: with
/// rayon each worker thread gets its own state and reuses it across the items
/// it steals, and without the feature a single state is reused for the whole
/// iteration. The per-item body must fully overwrite whatever it reads from the
/// state, so results stay independent of the thread count.
#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_map_init {
    ($slice:expr, $init:expr, $f:expr) => {
        $slice.par_iter().map_init($init, $f)
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_map_init {
    ($slice:expr, $init:expr, $f:expr) => {{
        let mut state = ($init)();
        let mut f = $f;
        $slice.iter().map(move |item| f(&mut state, item))
    }};
}
