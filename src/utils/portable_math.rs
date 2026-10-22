//! Elementary functions that give the same bits on every target.
//!
//! `f64::sin`, `exp`, `powf` and the rest call the platform's maths library: glibc (or the
//! system's libm) natively, and on `wasm32-unknown-unknown` a Rust port of musl's. Neither is
//! correctly rounded, and they disagree in the last bit for a few percent of inputs. That is
//! harmless where the result is only used as a number, but where it feeds a discrete decision
//! (a rounding, a threshold, a sign test, a hash) a one-ulp difference can change the decision,
//! and the browser and `qsmxt run` then give visibly different outputs from the same input.
//!
//! These wrappers call the [`libm`] crate (musl's algorithms in plain Rust, using only IEEE-754
//! arithmetic, which Rust never contracts or reorders), so a given input gives the same output
//! on every target. Use them for any value a decision depends on; elsewhere the standard methods
//! are fine. Each differs from the correctly rounded result by less than one ulp, as the
//! standard ones do.

/// `e^x`.
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Natural logarithm.
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// `x^y`.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Sine (radians).
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine (radians).
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Arccosine, on `[0, π]`.
#[inline]
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

/// Four-quadrant arctangent of `y / x`, on `[-π, π]`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// `sqrt(x² + y²)` without undue overflow or underflow.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}
