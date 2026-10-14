//! Progress and intermediate outputs reported by pipeline stages as they run.
//!
//! A stage that produces more than its return value (QSMART's vessel mask and per-stage local
//! fields and χ, for example) hands each intermediate to an [`Observer`] as soon as it exists.
//! The observer gets a borrowed slice, so a host that only wants the final result keeps no extra
//! copies, and a host that wants to show or save an intermediate copies or writes just that one.
//!
//! Any `FnMut(usize, usize)` closure is an observer that takes progress and ignores outputs, and
//! `&mut ()` is an observer that ignores both.

/// A borrowed volume reported by a stage: real-valued data or a binary mask, in the stage's
/// grid (same dimensions and voxel order as its inputs).
#[derive(Clone, Copy, Debug)]
pub enum VolumeRef<'a> {
    /// Real-valued volume (a field or susceptibility map, in the units the stage documents).
    Real(&'a [f64]),
    /// Binary mask (1 = inside, 0 = outside).
    Mask(&'a [u8]),
}

/// Receives a stage's progress and its intermediate outputs as they are produced.
///
/// Both methods default to doing nothing, so an implementation overrides only what it uses.
pub trait Observer {
    /// Step `done` of `total` has started.
    fn progress(&mut self, _done: usize, _total: usize) {}

    /// An intermediate output named `name` is ready. The stage documents its names; the slice is
    /// only valid for the duration of the call.
    fn output(&mut self, _name: &'static str, _volume: VolumeRef<'_>) {}
}

/// A progress callback is an observer that ignores intermediate outputs.
impl<F: FnMut(usize, usize)> Observer for F {
    fn progress(&mut self, done: usize, total: usize) {
        self(done, total)
    }
}

/// Ignores progress and outputs.
impl Observer for () {}
