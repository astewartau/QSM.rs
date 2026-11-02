//! Conjugate gradients squared (CGS), with MATLAB `cgs`'s stopping rules
//!
//! Sonneveld's CGS (SIAM J. Sci. Stat. Comput. 10(1):36-52, 1989) for a general square `A`,
//! unpreconditioned, as MATLAB (R2023b) `cgs` runs it, because ports of MATLAB code that call
//! `cgs` (RESHARP's `resharp.m`) must stop where it stops and return what it returns:
//!
//! - shadow residual `r̃ = r₀`; each iteration makes two operator calls (`A·p`, `A·(u+q)`);
//! - convergence when the recursively updated residual norm is at most `tol·‖b‖`, confirmed by
//!   the true residual `b − A·x` (if that misses, up to `min(⌊n/50⌋, 5, n − max_iter)` further
//!   "more steps" are taken, each confirmed by a true residual, before giving up);
//! - stagnation after 3 consecutive iterations whose update is below `eps·‖x‖`;
//! - breakdown when `ρ`, `β`, `r̃ᵀv` or `α` is zero or infinite;
//! - when it stops without converging, it returns the iterate with the smallest true residual
//!   seen (if that residual is no larger than the last one), not the last iterate.
//!
//! Vectors are real (the callers' operators are real symmetric); inner products and norms are
//! compensated sums, so the result agrees with MATLAB's to rounding, not bit for bit.

/// Why [`cgs_solve_with_progress`] stopped (MATLAB `cgs`'s `flag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgsFlag {
    /// 0: the true relative residual reached the tolerance.
    Converged,
    /// 1: the iteration cap was reached first.
    MaxIterations,
    /// 3: the iterates stagnated (or the "more steps" after a false convergence ran out).
    Stagnated,
    /// 4: a scalar of the recurrence became zero or infinite.
    Breakdown,
}

/// What [`cgs_solve_with_progress`] returns besides the solution.
#[derive(Clone, Copy, Debug)]
pub struct CgsReport {
    pub flag: CgsFlag,
    /// The iteration of the returned iterate (MATLAB's `iter`).
    pub iterations: usize,
    /// `‖b − A·x‖ / ‖b‖` of the returned iterate (MATLAB's `relres`).
    pub relres: f64,
}

/// Inner product with Neumaier-compensated summation (accurate to about one rounding, as
/// MATLAB's BLAS `ddot` nearly is; a plain sum over millions of voxels is not).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let p = x * y;
        let t = s + p;
        c += if s.abs() >= p.abs() { (s - t) + p } else { (p - t) + s };
        s = t;
    }
    s + c
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn residual<F: Fn(&[f64]) -> Vec<f64>>(a_op: &F, b: &[f64], x: &[f64]) -> Vec<f64> {
    let ax = a_op(x);
    b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect()
}

/// Solve `A·x = b` by CGS from `x0`, stopping as MATLAB `cgs(A, b, tol, max_iter, [], [], x0)`.
///
/// `tol` is clamped to `[eps, 1 − eps]` as MATLAB does. The progress callback receives
/// `(iteration, max_iter)` before each iteration.
pub fn cgs_solve_with_progress<F, P>(
    a_op: F,
    b: &[f64],
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    mut progress: P,
) -> (Vec<f64>, CgsReport)
where
    F: Fn(&[f64]) -> Vec<f64>,
    P: FnMut(usize, usize),
{
    let n = b.len();
    let eps = f64::EPSILON;
    let tol = tol.clamp(eps, 1.0 - eps);

    let n2b = norm(b);
    if n2b == 0.0 {
        return (vec![0.0; n], CgsReport { flag: CgsFlag::Converged, iterations: 0, relres: 0.0 });
    }

    let mut x = x0.to_vec();
    let mut xmin = x.clone();
    let mut imin = 0usize;
    let tolb = tol * n2b;
    let mut r = residual(&a_op, b, &x);
    let mut normr = norm(&r);
    let mut normr_act = normr;
    if normr <= tolb {
        return (x, CgsReport { flag: CgsFlag::Converged, iterations: 0, relres: normr / n2b });
    }

    let rt = r.clone();
    let mut normrmin = normr;
    let mut rho = 1.0f64;
    let mut stag = 0usize;
    let mut moresteps = 0usize;
    let maxmsteps = (n / 50).min(5).min(n.saturating_sub(max_iter));
    let maxstagsteps = 3usize;

    let mut flag = CgsFlag::MaxIterations;
    let mut iter_out = 0usize;
    let mut ii = 0usize;
    let mut u = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];
    let mut uh = vec![0.0; n];

    for it in 1..=max_iter {
        ii = it;
        progress(it - 1, max_iter);
        let rho1 = rho;
        rho = dot(&rt, &r);
        if rho == 0.0 || rho.is_infinite() {
            flag = CgsFlag::Breakdown;
            break;
        }
        if it == 1 {
            u.copy_from_slice(&r);
            p.copy_from_slice(&u);
        } else {
            let beta = rho / rho1;
            if beta == 0.0 || beta.is_infinite() {
                flag = CgsFlag::Breakdown;
                break;
            }
            for i in 0..n {
                u[i] = r[i] + beta * q[i];
                p[i] = u[i] + beta * (q[i] + beta * p[i]);
            }
        }
        let vh = a_op(&p);
        let rtvh = dot(&rt, &vh);
        if rtvh == 0.0 {
            flag = CgsFlag::Breakdown;
            break;
        }
        let alpha = rho / rtvh;
        if alpha.is_infinite() {
            flag = CgsFlag::Breakdown;
            break;
        }
        for i in 0..n {
            q[i] = u[i] - alpha * vh[i];
            uh[i] = u[i] + q[i];
        }
        if alpha.abs() * norm(&uh) < eps * norm(&x) {
            stag += 1;
        } else {
            stag = 0;
        }
        for i in 0..n {
            x[i] += alpha * uh[i];
        }
        let qh = a_op(&uh);
        for i in 0..n {
            r[i] -= alpha * qh[i];
        }
        normr = norm(&r);
        normr_act = normr;

        if normr <= tolb || stag >= maxstagsteps || moresteps > 0 {
            r = residual(&a_op, b, &x);
            normr_act = norm(&r);
            if normr_act <= tolb {
                flag = CgsFlag::Converged;
                iter_out = it;
                break;
            }
            if stag >= maxstagsteps && moresteps == 0 {
                stag = 0;
            }
            moresteps += 1;
            if moresteps >= maxmsteps {
                flag = CgsFlag::Stagnated;
                iter_out = it;
                break;
            }
        }
        if normr_act < normrmin {
            normrmin = normr_act;
            xmin.copy_from_slice(&x);
            imin = it;
        }
        if stag >= maxstagsteps {
            flag = CgsFlag::Stagnated;
            break;
        }
    }

    if flag == CgsFlag::Converged {
        return (x, CgsReport { flag, iterations: iter_out, relres: normr_act / n2b });
    }
    let rmin = norm(&residual(&a_op, b, &xmin));
    if rmin <= normr_act {
        (xmin, CgsReport { flag, iterations: imin, relres: rmin / n2b })
    } else {
        (x, CgsReport { flag, iterations: ii, relres: normr_act / n2b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_nonsymmetric_system() {
        // [[4, 1], [2, 3]] x = [1, 2] -> x = [0.1, 0.6]
        let a = |v: &[f64]| vec![4.0 * v[0] + v[1], 2.0 * v[0] + 3.0 * v[1]];
        let (x, rep) = cgs_solve_with_progress(a, &[1.0, 2.0], &[0.0, 0.0], 1e-12, 20, |_, _| {});
        assert_eq!(rep.flag, CgsFlag::Converged);
        assert!((x[0] - 0.1).abs() < 1e-10 && (x[1] - 0.6).abs() < 1e-10, "{x:?}");
    }

    #[test]
    fn zero_rhs_returns_zero() {
        let (x, rep) = cgs_solve_with_progress(|v: &[f64]| v.to_vec(), &[0.0; 3], &[1.0; 3], 1e-6, 5, |_, _| {});
        assert_eq!(x, vec![0.0; 3]);
        assert_eq!(rep.iterations, 0);
    }

    #[test]
    fn unconverged_returns_the_smallest_residual_iterate() {
        // A diagonal SPD system that one iteration cannot solve.
        let d = [1.0, 10.0, 100.0, 1000.0];
        let a = |v: &[f64]| v.iter().zip(&d).map(|(x, y)| x * y).collect::<Vec<_>>();
        let b = [1.0; 4];
        let (x, rep) = cgs_solve_with_progress(a, &b, &[0.0; 4], 1e-14, 1, |_, _| {});
        assert_eq!(rep.flag, CgsFlag::MaxIterations);
        let r: f64 = x.iter().zip(&d).zip(&b).map(|((x, d), b)| (b - x * d).powi(2)).sum::<f64>().sqrt();
        assert!((r / 2.0 - rep.relres).abs() < 1e-12 || rep.iterations == 0);
    }
}
