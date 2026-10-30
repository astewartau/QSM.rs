//! MATLAB's elementary functions, bit for bit: a Rust port of FDLIBM 5.3.
//!
//! MATLAB R2023b's `exp`, `log`, `sin`, `cos`, `atan2`, `hypot`, `abs` and `angle` of a complex
//! number, and its complex `exp`, return exactly FDLIBM 5.3's results. That was measured, not
//! assumed: `scripts/lsq_trr_matlab/elem_probe.m` ran each function in MATLAB on 400 000 to
//! 600 000 inputs over the ranges DECOMPOSE uses (and wider), as whole vectors, 8-element rows
//! and scalars, on AMD EPYC 7643 (Zen 3, AVX2), 9454 (Zen 4) and 9745 (Zen 5, AVX-512) nodes.
//! The three CPUs gave identical bits, and every output equalled FDLIBM 5.3 (compiled from
//! netlib's C) on every input. Neither platform library does: against MATLAB, glibc 2.44 agrees
//! on 85–97 % of inputs and musl on 92–100 % (musl's `atan2` is FDLIBM's; its `exp`, `log`,
//! `sin`, `cos` and `hypot` are not). Each differs from the correctly rounded result by under one
//! ulp, as glibc's and musl's do, so the choice only matters where a computation must reproduce
//! MATLAB's bits: a finite-difference Jacobian, for one, divides a last-bit difference by
//! `h ≈ 1.5e-8`, and a least-squares fit can then take another path.
//!
//! MATLAB's complex `exp(z)` is `exp(re)·(cos im, sin im)` from these functions. Its complex
//! `log(z)` has imaginary part `atan2(im, re)`, and real part `log(hypot(re, im))` except near the
//! unit circle, where it is `½·log1p((a·a − 1) + b·b)` with `a = max(|re|, |im|)` and
//! `b = min(|re|, |im|)`, and `log(|re|)` when `im = 0`: see [`log_abs`].
//!
//! Only IEEE-754 arithmetic is used (Rust neither contracts `a*b+c` into an FMA nor reorders
//! floating-point expressions), so the results are the same on every target, glibc, static
//! musl and wasm alike.
//!
//! The port follows the C line by line, with the same constants, branches and evaluation order.
//! FDLIBM's notice, which its licence requires to be kept:
//!
//! > Copyright (C) 1993 (2004 for e_exp.c) by Sun Microsystems, Inc. All rights reserved.
//! >
//! > Developed at SunSoft, a Sun Microsystems, Inc. business.
//! > Permission to use, copy, modify, and distribute this
//! > software is freely granted, provided that this notice
//! > is preserved.

#![allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::excessive_precision,
    clippy::explicit_counter_loop,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

#[inline]
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}

#[inline]
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline]
fn with_hi(x: f64, h: i32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff) | ((h as u32 as u64) << 32))
}

#[inline]
fn from_words(h: i32, l: u32) -> f64 {
    f64::from_bits(((h as u32 as u64) << 32) | l as u64)
}

/// `e^x` (`e_exp.c`, `__ieee754_exp`).
pub fn exp(x: f64) -> f64 {
    const HALF: [f64; 2] = [0.5, -0.5];
    const HUGE: f64 = 1.0e+300;
    const TWOM1000: f64 = 9.33263618503218878990e-302;
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const U_THRESHOLD: f64 = -7.45133219101941108420e+02;
    const LN2HI: [f64; 2] = [6.93147180369123816490e-01, -6.93147180369123816490e-01];
    const LN2LO: [f64; 2] = [1.90821492927058770002e-10, -1.90821492927058770002e-10];
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const P1: f64 = 1.66666666666666019037e-01;
    const P2: f64 = -2.77777777770155933842e-03;
    const P3: f64 = 6.61375632143793436117e-05;
    const P4: f64 = -1.65339022054652515390e-06;
    const P5: f64 = 4.13813679705723846039e-08;

    let mut x = x;
    let mut hx = hi(x) as u32;
    let xsb = ((hx >> 31) & 1) as usize;
    hx &= 0x7fff_ffff;
    let (mut hi_, mut lo_) = (0.0, 0.0);
    let k: i32;
    if hx >= 0x4086_2E42 {
        if hx >= 0x7ff0_0000 {
            if ((hx & 0xfffff) | lo(x)) != 0 {
                return x + x;
            }
            return if xsb == 0 { x } else { 0.0 };
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE;
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000;
        }
    }
    if hx > 0x3fd6_2e42 {
        if hx < 0x3FF0_A2B2 {
            hi_ = x - LN2HI[xsb];
            lo_ = LN2LO[xsb];
            k = 1 - xsb as i32 - xsb as i32;
        } else {
            k = (INVLN2 * x + HALF[xsb]) as i32;
            let t = k as f64;
            hi_ = x - t * LN2HI[0];
            lo_ = t * LN2LO[0];
        }
        x = hi_ - lo_;
    } else if hx < 0x3e30_0000 {
        if HUGE + x > 1.0 {
            return 1.0 + x;
        }
        k = 0;
    } else {
        k = 0;
    }
    let t = x * x;
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo_ - (x * c) / (2.0 - c)) - hi_);
    if k >= -1021 {
        with_hi(y, hi(y).wrapping_add(k << 20))
    } else {
        with_hi(y, hi(y).wrapping_add((k + 1000) << 20)) * TWOM1000
    }
}

const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const TWO54: f64 = 1.80143985094819840000e+16;

/// Natural logarithm (`e_log.c`, `__ieee754_log`).
pub fn log(x: f64) -> f64 {
    const LG1: f64 = 6.666666666666735130e-01;
    const LG2: f64 = 3.999999999940941908e-01;
    const LG3: f64 = 2.857142874366239149e-01;
    const LG4: f64 = 2.222219843214978396e-01;
    const LG5: f64 = 1.818357216161805012e-01;
    const LG6: f64 = 1.531383769920937332e-01;
    const LG7: f64 = 1.479819860511658591e-01;

    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x10_0000;
    x = with_hi(x, hx | (i ^ 0x3ff0_0000));
    k += i >> 20;
    let f = x - 1.0;
    if (0x000f_ffff & (2 + hx)) < 3 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f64;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.33333333333333333 * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let mut i = hx - 0x6147a;
    let w = z * z;
    let j = 0x6b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

/// `log(1 + x)` (`s_log1p.c`).
pub fn log1p(x: f64) -> f64 {
    const LP1: f64 = 6.666666666666735130e-01;
    const LP2: f64 = 3.999999999940941908e-01;
    const LP3: f64 = 2.857142874366239149e-01;
    const LP4: f64 = 2.222219843214978396e-01;
    const LP5: f64 = 1.818357216161805012e-01;
    const LP6: f64 = 1.531383769920937332e-01;
    const LP7: f64 = 1.479819860511658591e-01;

    let hx = hi(x);
    let ax = hx & 0x7fff_ffff;
    let mut k: i32 = 1;
    let mut f = 0.0;
    let mut hu: i32 = 0;
    let mut c = 0.0;
    if hx < 0x3FDA_827A {
        if ax >= 0x3ff0_0000 {
            return if x == -1.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        }
        if ax < 0x3e20_0000 {
            if TWO54 + x > 0.0 && ax < 0x3c90_0000 {
                return x;
            }
            return x - x * x * 0.5;
        }
        if hx > 0 || hx <= 0xbfd2_bec3_u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x4340_0000 {
            u = 1.0 + x;
            hu = hi(u);
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = hi(u);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000f_ffff;
        if hu < 0x6a09e {
            u = with_hi(u, hu | 0x3ff0_0000);
        } else {
            k += 1;
            u = with_hi(u, hu | 0x3fe0_0000);
            hu = (0x0010_0000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let kf = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c += kf * LN2_LO;
            return kf * LN2_HI + c;
        }
        let r = hfsq * (1.0 - 0.66666666666666666 * f);
        if k == 0 {
            return f - r;
        }
        return kf * LN2_HI - ((r - (kf * LN2_LO + c)) - f);
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let r = z * (LP1 + z * (LP2 + z * (LP3 + z * (LP4 + z * (LP5 + z * (LP6 + z * LP7))))));
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        kf * LN2_HI - ((hfsq - (s * (hfsq + r) + (kf * LN2_LO + c))) - f)
    }
}

/// `k_sin.c`: sine on `[−π/4, π/4]` of `x + y` (`iy = 0` when `y` is zero).
fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;
    let ix = hi(x) & 0x7fff_ffff;
    if ix < 0x3e40_0000 && (x as i32) == 0 {
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

/// `k_cos.c`: cosine on `[−π/4, π/4]` of `x + y`.
fn kernel_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;
    let ix = hi(x) & 0x7fff_ffff;
    if ix < 0x3e40_0000 && (x as i32) == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD3_3333 {
        return 1.0 - (0.5 * z - (z * r - x * y));
    }
    let qx = if ix > 0x3fe9_0000 {
        0.28125
    } else {
        from_words(ix - 0x0020_0000, 0)
    };
    let hz = 0.5 * z - qx;
    let a = 1.0 - qx;
    a - (hz - (z * r - x * y))
}

const TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

const NPIO2_HW: [i32; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB,
    0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB,
    0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C,
    0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

/// `e_rem_pio2.c`: `x − n·π/2` as `y[0] + y[1]`, and `n`.
fn rem_pio2(x: f64, y: &mut [f64; 2]) -> i32 {
    const TWO24: f64 = 1.67772160000000000000e+07;
    const INVPIO2: f64 = 6.36619772367581382433e-01;
    const PIO2_1: f64 = 1.57079632673412561417e+00;
    const PIO2_1T: f64 = 6.07710050650619224932e-11;
    const PIO2_2: f64 = 6.07710050630396597660e-11;
    const PIO2_2T: f64 = 2.02226624879595063154e-21;
    const PIO2_3: f64 = 2.02226624871116645580e-21;
    const PIO2_3T: f64 = 8.47842766036889956997e-32;

    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        y[0] = x;
        y[1] = 0.0;
        return 0;
    }
    if ix < 0x4002_d97c {
        if hx > 0 {
            let mut z = x - PIO2_1;
            if ix != 0x3ff9_21fb {
                y[0] = z - PIO2_1T;
                y[1] = (z - y[0]) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y[0] = z - PIO2_2T;
                y[1] = (z - y[0]) - PIO2_2T;
            }
            return 1;
        }
        let mut z = x + PIO2_1;
        if ix != 0x3ff9_21fb {
            y[0] = z + PIO2_1T;
            y[1] = (z - y[0]) + PIO2_1T;
        } else {
            z += PIO2_2;
            y[0] = z + PIO2_2T;
            y[1] = (z - y[0]) + PIO2_2T;
        }
        return -1;
    }
    if ix <= 0x4139_21fb {
        let mut t = x.abs();
        let n = (t * INVPIO2 + 0.5) as i32;
        let fn_ = n as f64;
        let mut r = t - fn_ * PIO2_1;
        let mut w = fn_ * PIO2_1T;
        if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
            y[0] = r - w;
        } else {
            let j = ix >> 20;
            y[0] = r - w;
            let mut i = j - ((hi(y[0]) >> 20) & 0x7ff);
            if i > 16 {
                t = r;
                w = fn_ * PIO2_2;
                r = t - w;
                w = fn_ * PIO2_2T - ((t - r) - w);
                y[0] = r - w;
                i = j - ((hi(y[0]) >> 20) & 0x7ff);
                if i > 49 {
                    t = r;
                    w = fn_ * PIO2_3;
                    r = t - w;
                    w = fn_ * PIO2_3T - ((t - r) - w);
                    y[0] = r - w;
                }
            }
        }
        y[1] = (r - y[0]) - w;
        if hx < 0 {
            y[0] = -y[0];
            y[1] = -y[1];
            return -n;
        }
        return n;
    }
    if ix >= 0x7ff0_0000 {
        y[0] = x - x;
        y[1] = y[0];
        return 0;
    }
    let e0 = (ix >> 20) - 1046;
    let mut z = from_words(ix - (e0 << 20), lo(x));
    let mut tx = [0.0; 3];
    for t in tx.iter_mut().take(2) {
        *t = (z as i32) as f64;
        z = (z - *t) * TWO24;
    }
    tx[2] = z;
    let mut nx = 3;
    while tx[nx - 1] == 0.0 {
        nx -= 1;
    }
    let n = kernel_rem_pio2(&tx[..nx], y, e0);
    if hx < 0 {
        y[0] = -y[0];
        y[1] = -y[1];
        return -n;
    }
    n
}

/// `scalbn` (`s_scalbn.c`) for the finite arguments `kernel_rem_pio2` gives it.
fn scalbn(x: f64, n: i32) -> f64 {
    const TWOM54: f64 = 5.55111512312578270212e-17;
    let mut x = x;
    let mut hx = hi(x);
    let mut k = (hx & 0x7ff0_0000) >> 20;
    if k == 0 {
        if (lo(x) | (hx & 0x7fff_ffff) as u32) == 0 {
            return x;
        }
        x *= TWO54;
        hx = hi(x);
        k = ((hx & 0x7ff0_0000) >> 20) - 54;
    }
    if k == 0x7ff {
        return x + x;
    }
    k += n;
    if k > 0x7fe {
        return f64::INFINITY.copysign(x);
    }
    if k > 0 {
        return with_hi(x, (hx & 0x800f_ffff_u32 as i32) | (k << 20));
    }
    if k <= -54 {
        return 0.0_f64.copysign(x);
    }
    k += 54;
    with_hi(x, (hx & 0x800f_ffff_u32 as i32) | (k << 20)) * TWOM54
}

/// `k_rem_pio2.c` with `prec = 2` (double), as `rem_pio2` calls it for `|x| > 2¹⁹·π/2`.
fn kernel_rem_pio2(x: &[f64], y: &mut [f64; 2], e0: i32) -> i32 {
    const PIO2: [f64; 8] = [
        1.57079625129699707031e+00,
        7.54978941586159635335e-08,
        5.39030252995776476554e-15,
        3.28200341580791294123e-22,
        1.27065575308067607349e-29,
        1.22933308981111328932e-36,
        2.73370053816464559624e-44,
        2.16741683877804819444e-51,
    ];
    const TWO24: f64 = 1.67772160000000000000e+07;
    const TWON24: f64 = 5.96046447753906250000e-08;
    let ipio2 = &TWO_OVER_PI;
    let nx = x.len() as i32;
    let jk: i32 = 4; // init_jk[prec = 2]
    let jp = jk;
    let jx = nx - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);
    let mut f = [0.0f64; 20];
    let mut q = [0.0f64; 20];
    let mut fq = [0.0f64; 20];
    let mut iq = [0i32; 20];
    let mut j = jv - jx;
    let m = jx + jk;
    for fi in f.iter_mut().take((m + 1) as usize) {
        *fi = if j < 0 { 0.0 } else { ipio2[j as usize] as f64 };
        j += 1;
    }
    for i in 0..=jk {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw += x[j as usize] * f[(jx + i - j) as usize];
        }
        q[i as usize] = fw;
    }
    let mut jz = jk;
    let mut n;
    let mut ih;
    let mut z;
    loop {
        // distill q[] into iq[] reversingly
        let mut i = 0usize;
        let mut j = jz;
        z = q[jz as usize];
        while j > 0 {
            let fw = ((TWON24 * z) as i32) as f64;
            iq[i] = (z - TWO24 * fw) as i32;
            z = q[(j - 1) as usize] + fw;
            i += 1;
            j -= 1;
        }
        z = scalbn(z, q0);
        z -= 8.0 * (z * 0.125).floor();
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            let i = iq[(jz - 1) as usize] >> (24 - q0);
            n += i;
            iq[(jz - 1) as usize] -= i << (24 - q0);
            ih = iq[(jz - 1) as usize] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[(jz - 1) as usize] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }
        if ih > 0 {
            n += 1;
            let mut carry = 0;
            for v in iq.iter_mut().take(jz as usize) {
                let jv_ = *v;
                if carry == 0 {
                    if jv_ != 0 {
                        carry = 1;
                        *v = 0x100_0000 - jv_;
                    }
                } else {
                    *v = 0xff_ffff - jv_;
                }
            }
            if q0 > 0 {
                match q0 {
                    1 => iq[(jz - 1) as usize] &= 0x7f_ffff,
                    2 => iq[(jz - 1) as usize] &= 0x3f_ffff,
                    _ => {}
                }
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= scalbn(1.0, q0);
                }
            }
        }
        if z == 0.0 {
            let mut jj = 0;
            let mut i = jz - 1;
            while i >= jk {
                jj |= iq[i as usize];
                i -= 1;
            }
            if jj == 0 {
                let mut k = 1;
                while iq[(jk - k) as usize] == 0 {
                    k += 1;
                }
                for i in (jz + 1)..=(jz + k) {
                    f[(jx + i) as usize] = ipio2[(jv + i) as usize] as f64;
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw += x[j as usize] * f[(jx + i - j) as usize];
                    }
                    q[i as usize] = fw;
                }
                jz += k;
                continue;
            }
        }
        break;
    }
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz as usize] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        z = scalbn(z, -q0);
        if z >= TWO24 {
            let fw = ((TWON24 * z) as i32) as f64;
            iq[jz as usize] = (z - TWO24 * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz as usize] = fw as i32;
        } else {
            iq[jz as usize] = z as i32;
        }
    }
    let mut fw = scalbn(1.0, q0);
    let mut i = jz;
    while i >= 0 {
        q[i as usize] = fw * iq[i as usize] as f64;
        fw *= TWON24;
        i -= 1;
    }
    let mut i = jz;
    while i >= 0 {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw += PIO2[k as usize] * q[(i + k) as usize];
            k += 1;
        }
        fq[(jz - i) as usize] = fw;
        i -= 1;
    }
    // prec = 2
    let mut fw = 0.0;
    let mut i = jz;
    while i >= 0 {
        fw += fq[i as usize];
        i -= 1;
    }
    y[0] = if ih == 0 { fw } else { -fw };
    let mut fw = fq[0] - fw;
    for v in fq.iter().take((jz + 1) as usize).skip(1) {
        fw += *v;
    }
    y[1] = if ih == 0 { fw } else { -fw };
    n & 7
}

/// Arccosine (`e_acos.c`, `__ieee754_acos`).
///
/// QSMART's curvature (`curvatures.m`) takes its triangle angles with MATLAB's `acos`. On the
/// 6 732 `acos` calls of the 36×36×28 fixture of `tests/qsmart_qsmci_parity.rs` (licensed R2023b,
/// astewartau/QSM.rs#320), this gives MATLAB's bits on every one; glibc's `acos` differs on 1 %.
pub fn acos(x: f64) -> f64 {
    const PI: f64 = 3.14159265358979311600e+00;
    const PIO2_HI: f64 = 1.57079632679489655800e+00;
    const PIO2_LO: f64 = 6.12323399573676603587e-17;
    const PS0: f64 = 1.66666666666666657415e-01;
    const PS1: f64 = -3.25565818622400915405e-01;
    const PS2: f64 = 2.01212532134862925881e-01;
    const PS3: f64 = -4.00555345006794114027e-02;
    const PS4: f64 = 7.91534994289814532176e-04;
    const PS5: f64 = 3.47933107596021167570e-05;
    const QS1: f64 = -2.40339491173441421878e+00;
    const QS2: f64 = 2.02094576023350569471e+00;
    const QS3: f64 = -6.88283971605453293030e-01;
    const QS4: f64 = 7.70381505559019352791e-02;
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        // |x| >= 1
        if ((ix - 0x3ff0_0000) as u32 | lo(x)) == 0 {
            return if hx > 0 { 0.0 } else { PI + 2.0 * PIO2_LO };
        }
        return (x - x) / (x - x);
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix <= 0x3c60_0000 {
            return PIO2_HI + PIO2_LO;
        }
        let z = x * x;
        let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
        let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
        let r = p / q;
        PIO2_HI - (x - (PIO2_LO - x * r))
    } else if hx < 0 {
        // x < -0.5
        let z = (1.0 + x) * 0.5;
        let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
        let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
        let s = z.sqrt();
        let r = p / q;
        let w = r * s - PIO2_LO;
        PI - 2.0 * (s + w)
    } else {
        // x > 0.5
        let z = (1.0 - x) * 0.5;
        let s = z.sqrt();
        let df = f64::from_bits(s.to_bits() & 0xffff_ffff_0000_0000);
        let c = (z - df * df) / (s + df);
        let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
        let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
        let r = p / q;
        let w = r * s + c;
        2.0 * (df + w)
    }
}

/// Sine, radians (`s_sin.c`).
pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    let mut y = [0.0; 2];
    let n = rem_pio2(x, &mut y);
    match n & 3 {
        0 => kernel_sin(y[0], y[1], 1),
        1 => kernel_cos(y[0], y[1]),
        2 => -kernel_sin(y[0], y[1], 1),
        _ => -kernel_cos(y[0], y[1]),
    }
}

/// Cosine, radians (`s_cos.c`).
pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    let mut y = [0.0; 2];
    let n = rem_pio2(x, &mut y);
    match n & 3 {
        0 => kernel_cos(y[0], y[1]),
        1 => -kernel_sin(y[0], y[1], 1),
        2 => -kernel_cos(y[0], y[1]),
        _ => kernel_sin(y[0], y[1], 1),
    }
}

/// Arctangent (`s_atan.c`).
pub fn atan(x: f64) -> f64 {
    const ATANHI: [f64; 4] = [
        4.63647609000806093515e-01,
        7.85398163397448278999e-01,
        9.82793723247329054082e-01,
        1.57079632679489655800e+00,
    ];
    const ATANLO: [f64; 4] = [
        2.26987774529616870924e-17,
        3.06161699786838301793e-17,
        1.39033110312309984516e-17,
        6.12323399573676603587e-17,
    ];
    const AT: [f64; 11] = [
        3.33333333333329318027e-01,
        -1.99999999998764832476e-01,
        1.42857142725034663711e-01,
        -1.11111104054623557880e-01,
        9.09088713343650656196e-02,
        -7.69187620504482999495e-02,
        6.66107313738753120669e-02,
        -5.83357013379057348645e-02,
        4.97687799461593236017e-02,
        -3.65315727442169155270e-02,
        1.62858201153657823623e-02,
    ];
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    let mut x = x;
    let id: i32;
    if ix >= 0x4410_0000 {
        if ix > 0x7ff0_0000 || (ix == 0x7ff0_0000 && lo(x) != 0) {
            return x + x;
        }
        return if hx > 0 {
            ATANHI[3] + ATANLO[3]
        } else {
            -ATANHI[3] - ATANLO[3]
        };
    }
    if ix < 0x3fdc_0000 {
        if ix < 0x3e20_0000 && 1.0e300 + x > 1.0 {
            return x;
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3ff3_0000 {
            if ix < 0x3fe6_0000 {
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x4003_8000 {
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            id = 3;
            x = -1.0 / x;
        }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        return x - x * (s1 + s2);
    }
    let id = id as usize;
    let z = ATANHI[id] - ((x * (s1 + s2) - ATANLO[id]) - x);
    if hx < 0 {
        -z
    } else {
        z
    }
}

/// Four-quadrant arctangent of `y / x` (`e_atan2.c`, `__ieee754_atan2`). MATLAB's `atan2` and
/// `angle`.
pub fn atan2(y: f64, x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    const PI_O_4: f64 = 7.8539816339744827900E-01;
    const PI_O_2: f64 = 1.5707963267948965580E+00;
    const PI: f64 = 3.1415926535897931160E+00;
    const PI_LO: f64 = 1.2246467991473531772E-16;
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    let lx = lo(x);
    let hy = hi(y);
    let iy = hy & 0x7fff_ffff;
    let ly = lo(y);
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if (hx.wrapping_sub(0x3ff0_0000) as u32 | lx) == 0 {
        return atan(y);
    }
    let m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if (iy as u32 | ly) == 0 {
        return match m {
            0 | 1 => y,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if (ix as u32 | lx) == 0 {
        return if hy < 0 {
            -PI_O_2 - TINY
        } else {
            PI_O_2 + TINY
        };
    }
    if ix == 0x7ff0_0000 {
        if iy == 0x7ff0_0000 {
            return match m {
                0 => PI_O_4 + TINY,
                1 => -PI_O_4 - TINY,
                2 => 3.0 * PI_O_4 + TINY,
                _ => -3.0 * PI_O_4 - TINY,
            };
        }
        return match m {
            0 => 0.0,
            1 => -0.0,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if iy == 0x7ff0_0000 {
        return if hy < 0 {
            -PI_O_2 - TINY
        } else {
            PI_O_2 + TINY
        };
    }
    let k = (iy - ix) >> 20;
    let z = if k > 60 {
        PI_O_2 + 0.5 * PI_LO
    } else if hx < 0 && k < -60 {
        0.0
    } else {
        atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

/// `sqrt(x² + y²)` (`e_hypot.c`, `__ieee754_hypot`). MATLAB's `hypot` and complex `abs`.
pub fn hypot(x: f64, y: f64) -> f64 {
    let mut ha = hi(x) & 0x7fff_ffff;
    let mut hb = hi(y) & 0x7fff_ffff;
    let (mut a, mut b) = if hb > ha {
        std::mem::swap(&mut ha, &mut hb);
        (y, x)
    } else {
        (x, y)
    };
    a = with_hi(a, ha);
    b = with_hi(b, hb);
    if (ha - hb) > 0x3c0_0000 {
        return a + b;
    }
    let mut k: i32 = 0;
    if ha > 0x5f30_0000 {
        if ha >= 0x7ff0_0000 {
            let mut w = a + b;
            if ((ha & 0xfffff) as u32 | lo(a)) == 0 {
                w = a;
            }
            if ((hb ^ 0x7ff0_0000) as u32 | lo(b)) == 0 {
                w = b;
            }
            return w;
        }
        ha -= 0x2580_0000;
        hb -= 0x2580_0000;
        k += 600;
        a = with_hi(a, ha);
        b = with_hi(b, hb);
    }
    if hb < 0x20b0_0000 {
        if hb <= 0x000f_ffff {
            if (hb as u32 | lo(b)) == 0 {
                return a;
            }
            let t1 = from_words(0x7fd0_0000, 0);
            b *= t1;
            a *= t1;
            k -= 1022;
        } else {
            ha += 0x2580_0000;
            hb += 0x2580_0000;
            k -= 600;
            a = with_hi(a, ha);
            b = with_hi(b, hb);
        }
    }
    let mut w = a - b;
    if w > b {
        let t1 = from_words(ha, 0);
        let t2 = a - t1;
        w = (t1 * t1 - (b * (-b) - t2 * (a + t1))).sqrt();
    } else {
        a = a + a;
        let y1 = from_words(hb, 0);
        let y2 = b - y1;
        let t1 = from_words(ha + 0x0010_0000, 0);
        let t2 = a - t1;
        w = (t1 * y1 - (w * (-w) - (t1 * y2 + t2 * b))).sqrt();
    }
    if k != 0 {
        from_words(0x3ff0_0000 + (k << 20), 0) * w
    } else {
        w
    }
}

/// The real part of MATLAB's complex `log(re + i·im)`, `log|z|`.
///
/// `log(hypot(re, im))`, except for `7/8 < |z| < 9/8`, where it is
/// `½·log1p((a·a − 1) + b·b)` with `a = max(|re|, |im|)`, `b = min(|re|, |im|)`: there
/// `log(hypot)` would lose the low bits of `|z|² − 1`. Measured on MATLAB R2023b
/// (`elem_probe.m`): on 500 000 complex inputs and 800 000 DECOMPOSE model values, every result
/// equals this; the boundaries were read off inputs straddling them by a few ulps.
///
/// An element whose imaginary part is zero takes the real `log(|re|)` instead, whether the array
/// is complex or not (MATLAB stores an array whose imaginary parts are all zero as real).
pub fn log_abs(re: f64, im: f64) -> f64 {
    if im == 0.0 {
        return log(re.abs());
    }
    let h = hypot(re, im);
    if h > 0.875 && h < 1.125 {
        let (a, b) = if re.abs() >= im.abs() {
            (re.abs(), im.abs())
        } else {
            (im.abs(), re.abs())
        };
        0.5 * log1p((a * a - 1.0) + b * b)
    } else {
        log(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from netlib's FDLIBM 5.3 C (gcc, `-ffp-contract=off`), which MATLAB R2023b
    /// reproduces bit for bit; inputs chosen where glibc or musl differ from it.
    #[test]
    fn matches_fdlibm_c() {
        let cases: &[(&str, f64, f64, u64)] = &[
            ("exp", -20.337920770068465, 0.0, 0x3e1941a9afe68dc8),
            ("exp", -21.56552126641968, 0.0, 0x3dfd999f2c0e5c4c),
            ("exp", -19.717373591476335, 0.0, 0x3e277ce3e48f0822),
            ("log", 1.8633921525960495, 0.0, 0x3fe3eab0650d578a),
            ("log", 5.19587097211366e-05, 0.0, 0xc023bae94d2a484a),
            ("log", 0.2933484270095443, 0.0, 0xbff39f4f87ddc66e),
            ("sin", -11.834712007310983, 0.0, 0x3fe5611d0ac9565c),
            ("sin", -8.37854189969444, 0.0, 0xbfebb289df6f074c),
            ("sin", 9213749.512327854, 0.0, 0xbfef759816a5ae7f),
            ("cos", -7.559101327774528, 0.0, 0x3fd2999af64e1c94),
            ("cos", 33.19976940860728, 0.0, 0xbfcb106b0fa08533),
            ("cos", 4285948.854786421, 0.0, 0x3fee2e51a9c8ce60),
            ("log1p", -0.11353890308179407, 0.0, 0xbfbeda45293afb43),
            ("log1p", 0.16167968518550085, 0.0, 0x3fc32ed732c9b65d),
            ("log1p", 0.26013036278006796, 0.0, 0x3fcd98757e37120d),
            (
                "atan2",
                0.2015687867976168,
                0.22977942592155864,
                0x3feb38fc4890d450,
            ),
            (
                "atan2",
                0.30446485613588764,
                -0.02715979914998026,
                0xbfb6c6b5425c5801,
            ),
            (
                "atan2",
                -2.697430357331017,
                1.2834580929693309,
                0x400594734a833d52,
            ),
            (
                "hypot",
                -1.0573861991153275,
                0.8892454389143212,
                0x3ff61b0a4bbd9af2,
            ),
            (
                "hypot",
                -0.5405218274418392,
                1.3817715747541919,
                0x3ff7bd5c4f9270ab,
            ),
            (
                "hypot",
                1.4645536859553416,
                -0.9611939962940466,
                0x3ffc0761e0d4083c,
            ),
        ];
        for &(name, x, y, bits) in cases {
            let v = match name {
                "exp" => exp(x),
                "log" => log(x),
                "sin" => sin(x),
                "cos" => cos(x),
                "atan2" => atan2(y, x),
                "hypot" => hypot(x, y),
                "log1p" => log1p(x),
                _ => unreachable!(),
            };
            assert_eq!(v.to_bits(), bits, "{name}({x:e}, {y:e})");
        }
    }

    /// MATLAB R2023b's `real(log(complex(re, im)))` (`elem_probe.m`), inside and outside the
    /// band `7/8 < |z| < 9/8` where it switches to `log1p`.
    #[test]
    fn complex_log_matches_matlab() {
        let cases: &[(f64, f64, u64)] = &[
            (
                -0.0043370517366579635,
                0.943220487379572,
                0xbfadec749de5016e,
            ),
            (
                -0.004414393526700921,
                -1.014830869206897,
                0x3f8e2b82c96c24d5,
            ),
            (
                -0.06800991333273589,
                -0.9094936534885376,
                0xbfb792807118f2f3,
            ),
            (
                7.824427449121719e-06,
                0.0013751522800540424,
                0xc01a5b50974d5d2c,
            ),
        ];
        for &(re, im, bits) in cases {
            assert_eq!(log_abs(re, im).to_bits(), bits, "log|{re:e} + {im:e}i|");
        }
        // An element with a zero imaginary part takes the real log.
        assert_eq!(log_abs(0.95, 0.0), log(0.95));
        assert_ne!(log(0.95), 0.5 * log1p((0.95 * 0.95 - 1.0) + 0.0));
    }

    #[test]
    fn special_values() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(800.0), f64::INFINITY);
        assert_eq!(exp(-800.0), 0.0);
        assert_eq!(log(1.0), 0.0);
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert!(log(-1.0).is_nan());
        assert_eq!(log1p(0.0), 0.0);
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert_eq!(atan2(0.0, -1.0), std::f64::consts::PI);
        assert_eq!(atan2(1.0, 0.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(0.0, -2.0), 2.0);
        // Large-argument reduction (k_rem_pio2) against the identity sin² + cos² = 1.
        for &x in &[1e6, 1e22, 1.7e308, -3.3e15] {
            let (s, c) = (sin(x), cos(x));
            assert!((s * s + c * c - 1.0).abs() < 1e-15, "{x}");
        }
    }

    /// Within an ulp of the standard library everywhere on a sweep (both are faithful).
    #[test]
    fn faithful_against_std() {
        let ulp = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
        let mut x = -40.0;
        while x < 40.0 {
            assert!(ulp(exp(x), x.exp()) <= 1, "exp {x}");
            assert!(ulp(sin(x), x.sin()) <= 1 || sin(x).abs() < 1e-15, "sin {x}");
            assert!(ulp(cos(x), x.cos()) <= 1 || cos(x).abs() < 1e-15, "cos {x}");
            assert!(ulp(atan2(x, 1.3), x.atan2(1.3)) <= 1, "atan2 {x}");
            assert!(ulp(hypot(x, 0.7), x.hypot(0.7)) <= 1, "hypot {x}");
            let p = x.abs() + 1e-3;
            assert!(ulp(log(p), p.ln()) <= 1, "log {p}");
            assert!(ulp(log1p(p), p.ln_1p()) <= 1, "log1p {p}");
            x += 0.01234567;
        }
    }
}
