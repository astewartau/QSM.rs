//! Conjugate Gradient solver
//!
//! Solves Ax = b for symmetric positive definite A.

/// Conjugate gradient solver
///
/// Solves Ax = b where A is a linear operator represented by a closure.
///
/// # Arguments
/// * `a_op` - Closure that computes A*x
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Solution vector x
pub fn cg_solve<F>(
    a_op: F,
    b: &[f64],
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = x0.to_vec();

    // r = b - A*x
    let ax = a_op(&x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();

    let mut p = r.clone();

    let mut rsold: f64 = r.iter().map(|&ri| ri * ri).sum();
    let b_norm: f64 = b.iter().map(|&bi| bi * bi).sum::<f64>().sqrt();

    for _iter in 0..max_iter {
        let ap = a_op(&p);

        let pap: f64 = p.iter().zip(ap.iter())
            .map(|(&pi, &api)| pi * api)
            .sum();

        if pap.abs() < 1e-20 {
            break;
        }

        let alpha = rsold / pap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r = r - alpha * A*p
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let rsnew: f64 = r.iter().map(|&ri| ri * ri).sum();

        // Check convergence
        if rsnew.sqrt() < tol * b_norm {
            break;
        }

        let beta = rsnew / rsold;

        // p = r + beta * p
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_identity() {
        // Solve Ix = b (identity matrix)
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let x = cg_solve(|v| v.to_vec(), &b, &x0, 1e-10, 100);

        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-8, "x should equal b");
        }
    }

    #[test]
    fn test_cg_diagonal() {
        // Solve diag(2,3,4) * x = [2, 6, 12]
        // Solution: x = [1, 2, 3]
        let b = vec![2.0, 6.0, 12.0];
        let x0 = vec![0.0, 0.0, 0.0];
        let diag = vec![2.0, 3.0, 4.0];

        let x = cg_solve(
            |v| v.iter().zip(diag.iter()).map(|(&vi, &di)| vi * di).collect(),
            &b, &x0, 1e-10, 100
        );

        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < 1e-8, "Expected {}, got {}", ei, xi);
        }
    }
}
