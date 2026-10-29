//! Test support: MATLAB `lsqcurvefit` traces (`scripts/lsq_trr_matlab/probe_lsq_trr.m`) and the
//! comparison of [`super::lsq_trr`] against them, evaluation by evaluation.

use super::lsq_trr::{
    lsq_trust_region_reflective, LsqTrrIterate, LsqTrrOptions, LsqTrrReport, LsqTrrWorkspace,
};
use std::collections::HashMap;

/// One `lsqcurvefit` call as MATLAB ran it.
#[derive(Clone, Debug, Default)]
pub struct Call {
    pub tag: String,
    pub m: usize,
    pub x0: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    /// Every evaluation: point and sum of squares.
    pub evals: Vec<(Vec<f64>, f64)>,
    /// OutputFcn calls: state and the numeric fields.
    pub iters: Vec<(String, HashMap<String, f64>, Vec<f64>)>,
    pub exitflag: i32,
    pub iterations: usize,
    pub func_count: usize,
    pub firstorderopt: f64,
    pub resnorm: f64,
    pub x: Vec<f64>,
    /// DECOMPOSE: (voxel, pass, stage, [chi+, chi-, R0, C+, C-, C0]) of the stage call.
    pub stage: Option<(usize, usize, usize, [f64; 6])>,
}

fn num(s: &str) -> f64 {
    match s {
        "Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        "NaN" => f64::NAN,
        _ => s.parse().unwrap_or_else(|_| panic!("number {s:?}")),
    }
}

/// Parse a probe file into its calls, in order.
pub fn parse(text: &str) -> Vec<Call> {
    let mut calls: Vec<Call> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut pending_stage = None;
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        if w.is_empty() {
            continue;
        }
        match w[0] {
            "S" => {
                let v: Vec<f64> = w[4..10].iter().map(|s| num(s)).collect();
                pending_stage = Some((
                    w[1].parse().unwrap(),
                    w[2].parse().unwrap(),
                    w[3].parse().unwrap(),
                    [v[0], v[1], v[2], v[3], v[4], v[5]],
                ));
            }
            "P" => {
                let (m, n): (usize, usize) = (w[2].parse().unwrap(), w[3].parse().unwrap());
                let v: Vec<f64> = w[4..4 + 3 * n].iter().map(|s| num(s)).collect();
                index.insert(w[1].to_string(), calls.len());
                calls.push(Call {
                    tag: w[1].to_string(),
                    m,
                    x0: v[..n].to_vec(),
                    lb: v[n..2 * n].to_vec(),
                    ub: v[2 * n..].to_vec(),
                    stage: pending_stage.take(),
                    ..Default::default()
                });
            }
            "E" => {
                let c = &mut calls[index[w[1]]];
                let n = c.x0.len();
                let x: Vec<f64> = w[3..3 + n].iter().map(|s| num(s)).collect();
                c.evals.push((x, num(w[3 + n])));
            }
            "I" => {
                let c = &mut calls[index[w[1]]];
                let mut fields = HashMap::new();
                let mut x = Vec::new();
                for kv in &w[3..] {
                    let (k, v) = kv.split_once('=').unwrap();
                    if k == "x" {
                        x = v.split(',').map(num).collect();
                    } else {
                        fields.insert(k.to_string(), num(v));
                    }
                }
                c.iters.push((w[2].to_string(), fields, x));
            }
            "R" => {
                let c = &mut calls[index[w[1]]];
                c.exitflag = w[2].parse().unwrap();
                c.iterations = w[3].parse().unwrap();
                c.func_count = w[4].parse().unwrap();
                c.firstorderopt = num(w[5]);
                c.resnorm = num(w[6]);
                c.x = w[7..].iter().map(|s| num(s)).collect();
            }
            _ => {}
        }
    }
    calls
}

/// Rust's run of the same call.
#[derive(Clone, Debug)]
pub struct Replay {
    pub evals: Vec<(Vec<f64>, f64)>,
    #[allow(clippy::type_complexity)]
    pub iters: Vec<(usize, usize, f64, f64, f64, f64, Vec<f64>)>,
    pub report: LsqTrrReport,
    pub x: Vec<f64>,
}

/// Run [`lsq_trust_region_reflective`] on `call`'s problem (`resid` gives its residual).
pub fn replay(call: &Call, mut resid: impl FnMut(&[f64], &mut [f64])) -> Replay {
    let n = call.x0.len();
    let mut ws = LsqTrrWorkspace::new(call.m, n);
    let mut x = call.x0.clone();
    let mut evals = Vec::new();
    let mut iters = Vec::new();
    let report = lsq_trust_region_reflective(
        |x: &[f64], out: &mut [f64]| {
            resid(x, out);
            evals.push((x.to_vec(), out.iter().map(|r| r * r).sum()));
        },
        &mut x,
        &call.lb,
        &call.ub,
        &LsqTrrOptions::default(),
        &mut ws,
        |it: &LsqTrrIterate| {
            iters.push((
                it.iteration,
                it.func_count,
                it.resnorm,
                it.first_order_optimality,
                it.trust_region_radius,
                it.step_norm,
                it.x.to_vec(),
            ))
        },
    );
    Replay {
        evals,
        iters,
        report,
        x,
    }
}

/// `‖a − b‖∞ / max(‖a‖∞, ‖b‖∞)`.
fn vrel(a: &[f64], b: &[f64]) -> f64 {
    let d = a.iter().zip(b).map(|(p, q)| if p == q { 0.0 } else { (p - q).abs() }).fold(0.0, f64::max);
    let s = a.iter().chain(b).map(|v| v.abs()).fold(0.0, f64::max);
    if d == 0.0 { 0.0 } else { d / s.max(1e-300) }
}

/// How a replay compares with MATLAB.
#[allow(dead_code)] // read through Debug
#[derive(Clone, Debug)]
pub struct Diff {
    pub tag: String,
    pub same_counts: bool,
    pub same_exit: bool,
    /// First evaluation whose point differs by more than `tol` (relative), if any.
    pub first_divergence: Option<usize>,
    /// Largest relative difference over the evaluation points both have.
    pub max_eval_rel: f64,
    /// Relative difference of the final x.
    pub x_rel: f64,
    pub matlab: (i32, usize, usize),
    pub rust: (i32, usize, usize),
}

pub fn compare(call: &Call, r: &Replay, tol: f64) -> Diff {
    let mut first = None;
    let mut maxr = 0.0_f64;
    for (k, (a, b)) in call.evals.iter().zip(&r.evals).enumerate() {
        let d = vrel(&a.0, &b.0);
        maxr = maxr.max(d);
        if d > tol && first.is_none() {
            first = Some(k);
        }
    }
    if first.is_none() && call.evals.len() != r.evals.len() {
        first = Some(call.evals.len().min(r.evals.len()));
    }
    let x_rel = vrel(&call.x, &r.x);
    let rust = (
        r.report.exit.matlab_exitflag(),
        r.report.iterations,
        r.report.func_count,
    );
    let matlab = (call.exitflag, call.iterations, call.func_count);
    Diff {
        tag: call.tag.clone(),
        same_counts: matlab.1 == rust.1 && matlab.2 == rust.2,
        same_exit: matlab.0 == rust.0,
        first_divergence: first,
        max_eval_rel: maxr,
        x_rel,
        matlab,
        rust,
    }
}

/// Side-by-side listing of one call's evaluations and iterations, for diagnosing a divergence.
pub fn dump(call: &Call, r: &Replay) -> String {
    let mut s = format!(
        "{}: x0 {:?} lb {:?} ub {:?}\n",
        call.tag, call.x0, call.lb, call.ub
    );
    let k = call.evals.len().max(r.evals.len());
    for i in 0..k {
        let a = call
            .evals
            .get(i)
            .map(|e| format!("{:?} {:.17e}", e.0, e.1))
            .unwrap_or_default();
        let b = r
            .evals
            .get(i)
            .map(|e| format!("{:?} {:.17e}", e.0, e.1))
            .unwrap_or_default();
        s += &format!("  E{i:3} M {a}\n       R {b}\n");
    }
    for (st, f, x) in &call.iters {
        let mut keys: Vec<_> = f.keys().collect();
        keys.sort();
        let kv: Vec<String> = keys.iter().map(|k| format!("{k}={:.17e}", f[*k])).collect();
        s += &format!("  M-I {st} {} x={x:?}\n", kv.join(" "));
    }
    for it in &r.iters {
        s += &format!(
            "  R-I iter={} fc={} resnorm={:.17e} opt={:.17e} delta={:.17e} step={:.17e} x={:?}\n",
            it.0, it.1, it.2, it.3, it.4, it.5, it.6
        );
    }
    s += &format!(
        "  M-R {:?} x={:?}\n  R-R {:?} x={:?}\n",
        (
            call.exitflag,
            call.iterations,
            call.func_count,
            call.firstorderopt
        ),
        call.x,
        r.report,
        r.x
    );
    s
}
