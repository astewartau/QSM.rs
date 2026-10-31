//! Echo weighting for the multi-echo iQSM, iQFM and iQSM+ combinations.
//!
//! Not behind the `onnx` feature, so callers can name it (e.g. in configuration) whatever features
//! they build with; the networks themselves need `onnx`.

/// How the multi-echo iQSM and iQFM entry points combine their per-echo reconstructions:
/// `Σ w·χ / Σ w` over the echoes, voxel by voxel, with the weight `w` below. Without magnitudes
/// both are `TE²` (uniform magnitude), as in both of the authors' implementations.
///
/// The authors' two implementations disagree (checked at `sunhongfu/iQSM` 81f5462 and
/// `sunhongfu/iQSM_Plus` 3ba25f5):
///
/// - **MATLAB** (published with the paper): `sunhongfu/iQSM` `legacy/matlab/demo_multi_echo.m:152-160`
///   and `sunhongfu/iQSM_Plus` `legacy/matlab/iQSM.m:260-267` / `iQFM.m:263-270` multiply each
///   echo's output by its TE and fit that against TE with `echofit.m` (`sunhongfu/iQSM`
///   `legacy/matlab/iQSM_fcns/echofit.m`, zero intercept): `Σ mag·TE·(TE·χ) / Σ mag·TE²`, i.e.
///   weights `mag·TE²`.
/// - **Python** (the current drivers): `sunhongfu/iQSM` `run.py:170` and `app.py:416` weight by
///   `(mag·TE)²`, i.e. magnitude squared, although they print "Magnitude × TE²". QSM-CI's iQSM and
///   iQFM originals (bb163ef) run this Python.
///
/// The default is the MATLAB's (the implementation published with the paper);
/// [`MagnitudeSquaredTeSquared`](Self::MagnitudeSquaredTeSquared) reproduces the Python, for
/// parity with it. iQSM+ has no choice: both of its authors' implementations use `(mag·TE)²`
/// (see `iqsm_plus_multi_echo`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IqsmEchoWeighting {
    /// `w = mag·TE²`: the authors' MATLAB echo fit (the default).
    #[default]
    MagnitudeTeSquared,
    /// `w = (mag·TE)² = mag²·TE²`: the authors' Python drivers.
    MagnitudeSquaredTeSquared,
}

impl IqsmEchoWeighting {
    /// The weight of one echo with magnitude `mag` and echo time `te`.
    pub fn weight(self, mag: f64, te: f64) -> f64 {
        match self {
            Self::MagnitudeTeSquared => mag * te * te,
            Self::MagnitudeSquaredTeSquared => (mag * te) * (mag * te),
        }
    }
}

/// Reconstruct each echo with `echo(phase, te)` and combine them with `weighting`'s weights.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) fn combine_echoes<E>(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    tes: &[f64],
    n: usize,
    weighting: IqsmEchoWeighting,
    mut echo: impl FnMut(&[f64], f64) -> Result<Vec<f64>, E>,
) -> Result<Vec<f64>, E> {
    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    for (e, &phase) in phases.iter().enumerate() {
        let chi = echo(phase, tes[e])?;
        for i in 0..n {
            let w = weighting.weight(magnitudes.get(e).map(|m| m[i]).unwrap_or(1.0), tes[e]);
            acc[i] += w * chi[i];
            wsum[i] += w;
        }
    }
    for i in 0..n {
        acc[i] = if wsum[i] > 0.0 { acc[i] / wsum[i] } else { 0.0 };
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The combination's exact weights on two voxels and two echoes, each echo's "reconstruction"
    /// a constant per echo, so the result is the weighted mean of those constants.
    #[test]
    fn echo_weightings_are_the_authors() {
        let tes = [0.01, 0.03];
        let (m1, m2) = ([2.0, 1.0], [1.0, 0.0]);
        let mags: [&[f64]; 2] = [&m1, &m2];
        let phases: [&[f64]; 2] = [&[0.0; 2], &[0.0; 2]];
        let chi = |te: f64| if te < 0.02 { 1.0 } else { 5.0 };
        let run = |w, mags: &[&[f64]]| {
            combine_echoes(&phases, mags, &tes, 2, w, |_, te| Ok::<_, ()>(vec![chi(te); 2])).unwrap()
        };
        let mean = |w1: f64, w2: f64| (w1 * 1.0 + w2 * 5.0) / (w1 + w2);
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);

        // MATLAB echo fit: mag·TE².
        let matlab = run(IqsmEchoWeighting::MagnitudeTeSquared, &mags);
        assert!(close(&matlab, &[mean(2.0 * 1e-4, 1.0 * 9e-4), 1.0]), "{matlab:?}");
        // Python: (mag·TE)².
        let python = run(IqsmEchoWeighting::MagnitudeSquaredTeSquared, &mags);
        assert!(close(&python, &[mean(4.0 * 1e-4, 1.0 * 9e-4), 1.0]), "{python:?}");
        // Without magnitudes both are TE².
        for w in [IqsmEchoWeighting::MagnitudeTeSquared, IqsmEchoWeighting::MagnitudeSquaredTeSquared] {
            let r = run(w, &[]);
            assert!(close(&r, &[mean(1e-4, 9e-4); 2]), "{w:?}: {r:?}");
        }
        // All weights zero: zero.
        let zero = run(IqsmEchoWeighting::MagnitudeTeSquared, &[&[0.0; 2], &[0.0; 2]]);
        assert_eq!(zero, vec![0.0; 2]);
        assert_eq!(IqsmEchoWeighting::default(), IqsmEchoWeighting::MagnitudeTeSquared);
    }
}
