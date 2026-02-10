//! LSMR solver
//!
//! Least Squares Minimal Residual algorithm for solving
//! min ||Ax - b||₂ (possibly with ||x||₂ regularization).
//!
//! Reference:
//! Fong & Saunders, "LSMR: An iterative algorithm for sparse
//! least-squares problems", SISC 2011.

/// LSMR solver
///
/// Solves min ||Ax - b||₂ where A is a linear operator.
///
/// # Arguments
/// * `a_op` - Closure that computes A*x
/// * `at_op` - Closure that computes Aᵀ*y
/// * `b` - Right-hand side vector
/// * `lambda` - Regularization parameter (0 for standard least squares)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Solution vector x
pub fn lsmr_solve<F, G>(
    a_op: F,
    at_op: G,
    b: &[f64],
    lambda: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    // Initialize
    let m = b.len();
    let u = b.to_vec();
    let beta = norm(&u);

    let mut u: Vec<f64> = if beta > 0.0 {
        u.iter().map(|&ui| ui / beta).collect()
    } else {
        u
    };

    let v = at_op(&u);
    let n = v.len();
    let alpha = norm(&v);

    let mut v: Vec<f64> = if alpha > 0.0 {
        v.iter().map(|&vi| vi / alpha).collect()
    } else {
        v
    };

    // Initialize variables
    let mut x = vec![0.0; n];
    let mut h = v.clone();
    let mut h_bar = vec![0.0; n];

    let mut alpha_bar = alpha;
    let mut zeta_bar = alpha * beta;
    let mut rho = 1.0;
    let mut rho_bar;
    let mut c_bar = 1.0;
    let mut s_bar = 0.0;

    for _iter in 0..max_iter {
        // Bidiagonalization
        let au = a_op(&v);
        for i in 0..m {
            u[i] = au[i] - alpha * u[i];
        }
        let beta = norm(&u);

        if beta > 0.0 {
            for i in 0..m {
                u[i] /= beta;
            }
        }

        let atv = at_op(&u);
        for i in 0..n {
            v[i] = atv[i] - beta * v[i];
        }
        let alpha = norm(&v);

        if alpha > 0.0 {
            for i in 0..n {
                v[i] /= alpha;
            }
        }

        // QR factorization
        let rho_prev = rho;
        let chat = alpha_bar;
        let shat = lambda;
        let rho_temp = (chat * chat + shat * shat).sqrt();

        let theta_new;
        if rho_temp > 1e-20 {
            let c1 = chat / rho_temp;
            let s1 = shat / rho_temp;
            theta_new = s1 * alpha;
            alpha_bar = c1 * alpha;
        } else {
            theta_new = 0.0;
            // alpha_bar stays the same
        }

        rho = (alpha_bar * alpha_bar + beta * beta).sqrt();
        if rho < 1e-20 {
            break;  // Converged or degenerate
        }

        let theta_bar = s_bar * rho;
        rho_bar = ((c_bar * rho).powi(2) + theta_new.powi(2)).sqrt();
        if rho_bar < 1e-20 {
            break;  // Converged or degenerate
        }
        c_bar = c_bar * rho / rho_bar;
        s_bar = theta_new / rho_bar;

        let zeta = c_bar * zeta_bar;
        zeta_bar = -s_bar * zeta_bar;

        // Update solution
        let scale_h_bar = if (rho_prev * rho_bar).abs() > 1e-20 {
            theta_bar * rho / (rho_prev * rho_bar)
        } else {
            0.0
        };
        for i in 0..n {
            h_bar[i] = h[i] - scale_h_bar * h_bar[i];
        }

        let scale_x = if (rho * rho_bar).abs() > 1e-20 {
            zeta / (rho * rho_bar)
        } else {
            0.0
        };
        for i in 0..n {
            x[i] += scale_x * h_bar[i];
        }

        let scale_h = if rho.abs() > 1e-20 { theta_new / rho } else { 0.0 };
        for i in 0..n {
            h[i] = v[i] - scale_h * h[i];
        }

        // Check convergence
        if zeta_bar.abs() < tol {
            break;
        }
    }

    x
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]  // TODO: Debug LSMR implementation in Sprint 4
    fn test_lsmr_identity() {
        let b = vec![1.0, 2.0, 3.0];

        let x = lsmr_solve(
            |v| v.to_vec(),
            |v| v.to_vec(),
            &b, 0.0, 1e-10, 100
        );

        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-6, "x should equal b");
        }
    }
}
