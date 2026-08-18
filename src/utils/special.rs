//! Special functions: error function family.
//!
//! `erf`, `erfc`, and the scaled complementary error function `erfcx(x) = exp(x^2) * erfc(x)`.
//!
//! Ported from W. J. Cody's `CALERF` (SPECFUN / netlib, "Rational Chebyshev
//! approximation for the error function", Math. Comp. 1969). This is the same
//! algorithm MATLAB's `erf`/`erfc`/`erfcx` use, so results agree to ~1e-15.
//!
//! `erfcx` is needed by the AMP-PE GAMP parameter estimation, where the argument
//! can be large and positive — `exp(x^2)*erfc(x)` would overflow, whereas `erfcx`
//! stays bounded (it decays like 1/(x*sqrt(pi))).
//!
//! The coefficient tables are the canonical SPECFUN constants, quoted verbatim
//! (a few carry more digits than an `f64` can hold — rounded on load).
#![allow(clippy::excessive_precision)]

// Coefficients for approximation to erf in |x| <= 0.5
const A: [f64; 5] = [
    3.16112374387056560e00,
    1.13864154151050156e02,
    3.77485237685302021e02,
    3.20937758913846947e03,
    1.85777706184603153e-1,
];
const B: [f64; 4] = [
    2.36012909523441209e01,
    2.44024637934444173e02,
    1.28261652607737228e03,
    2.84423683343917062e03,
];
// Coefficients for approximation to erfc in 0.46875 <= |x| <= 4.0
const C: [f64; 9] = [
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
    2.15311535474403846e-8,
];
const D: [f64; 8] = [
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
];
// Coefficients for approximation to erfc in |x| > 4.0
const P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
const Q: [f64; 5] = [
    2.56852019228982242e00,
    1.87295284992346047e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];

const SQRPI: f64 = 5.6418958354775628695e-1; // 1/sqrt(pi)
const THRESH: f64 = 0.46875;

/// Core CALERF routine. `jint` selects the result:
/// 0 => erf(x), 1 => erfc(x), 2 => erfcx(x).
fn calerf(x: f64, jint: u8) -> f64 {
    let y = x.abs();
    let mut result;

    if y <= THRESH {
        // erf for |x| <= 0.46875
        let mut ysq = 0.0;
        if y > 1.11e-16 {
            ysq = y * y;
        }
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        result = x * (xnum + A[3]) / (xden + B[3]);
        if jint != 0 {
            result = 1.0 - result;
        }
        if jint == 2 {
            result *= ysq.exp();
        }
        return result;
    } else if y <= 4.0 {
        // erfc for 0.46875 <= |x| <= 4.0
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        result = (xnum + C[7]) / (xden + D[7]);
        if jint != 2 {
            let ysq = (y * 16.0).trunc() / 16.0;
            let del = (y - ysq) * (y + ysq);
            result *= (-ysq * ysq).exp() * (-del).exp();
        }
    } else {
        // erfc for |x| > 4.0
        result = 0.0;
        if y < 26.543 || jint == 2 {
            let ysq = 1.0 / (y * y);
            let mut xnum = P[5] * ysq;
            let mut xden = ysq;
            for i in 0..4 {
                xnum = (xnum + P[i]) * ysq;
                xden = (xden + Q[i]) * ysq;
            }
            result = ysq * (xnum + P[4]) / (xden + Q[4]);
            result = (SQRPI - result) / y;
            if jint != 2 {
                let ysq = (y * 16.0).trunc() / 16.0;
                let del = (y - ysq) * (y + ysq);
                result *= (-ysq * ysq).exp() * (-del).exp();
            }
        }
    }

    // Fix up for negative argument, erf, etc.
    match jint {
        0 => {
            result = 0.5 - result + 0.5;
            if x < 0.0 {
                result = -result;
            }
        }
        1 => {
            if x < 0.0 {
                result = 2.0 - result;
            }
        }
        _ => {
            // jint == 2, erfcx
            if x < 0.0 {
                let ysq = (x * 16.0).trunc() / 16.0;
                let del = (x - ysq) * (x + ysq);
                let y2 = (ysq * ysq).exp() * del.exp();
                result = (y2 + y2) - result;
            }
        }
    }
    result
}

/// Error function `erf(x)`.
#[inline]
pub fn erf(x: f64) -> f64 {
    calerf(x, 0)
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    calerf(x, 1)
}

/// Scaled complementary error function `erfcx(x) = exp(x^2) * erfc(x)`.
///
/// Stable for large positive `x` (decays like `1/(x*sqrt(pi))`), where the naive
/// `exp(x^2)*erfc(x)` would overflow / underflow to `0 * inf`.
#[inline]
pub fn erfcx(x: f64) -> f64 {
    calerf(x, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        assert!((erf(0.0)).abs() < 1e-15);
        assert!((erf(0.5) - 0.5204998778130465).abs() < 1e-12);
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-12);
        assert!((erf(-1.0) + 0.8427007929497149).abs() < 1e-12);
        assert!((erf(2.0) - 0.9953222650189527).abs() < 1e-12);
    }

    #[test]
    fn erfc_known_values() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-15);
        assert!((erfc(0.5) - 0.4795001221869535).abs() < 1e-12);
        assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-12);
        assert!((erfc(-1.0) - 1.8427007929497148).abs() < 1e-12);
    }

    #[test]
    fn erfcx_known_values() {
        // Reference values from MATLAB's erfcx (Cody CALERF), agree to ~1e-14.
        let cases = [
            (0.0, 1.0),
            (1.0, 0.427583576155807),
            (2.0, 0.25539567631050575),
            (5.0, 0.11070463773306863),
            (10.0, 0.056140992743822594),
            (20.0, 0.02817434874105132),
            (30.0, 0.01879588886141675),
            (-2.0, 108.94090438997797),
            (-0.5, 1.952360489182557),
        ];
        for (x, want) in cases {
            let got = erfcx(x);
            assert!((got - want).abs() / want.abs().max(1e-300) < 1e-13, "erfcx({x})={got} want {want}");
        }
        // erfcx must stay finite for large x (no overflow)
        assert!(erfcx(100.0).is_finite() && erfcx(100.0) > 0.0);
        assert!(erfcx(1000.0).is_finite());
    }

    #[test]
    fn erfcx_consistency() {
        // For moderate x, erfcx(x) == exp(x^2)*erfc(x)
        for &x in &[-3.0, -1.0, -0.3, 0.2, 0.8, 2.5] {
            let lhs = erfcx(x);
            let rhs = (x * x).exp() * erfc(x);
            assert!((lhs - rhs).abs() / rhs.abs().max(1e-300) < 1e-10, "x={x}");
        }
    }
}
