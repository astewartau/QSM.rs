//! Which echo time converts a field in ppm to phase in radians.
//!
//! Several methods take the field in ppm but work on the phase in radians inside, converting
//! with `2π·γ·B0·TE`. A field combined from several echoes has no single echo time, so the
//! method has to be told which one to use, and the choice scales the phase: it changes how
//! strongly a nonlinear data term, or a threshold defined in radians, acts, and so it can
//! change the result. Each such method has an [`EchoTime`] parameter next to the scan's echo
//! times, and its default is what that method's reference code uses (documented, with its
//! source, on the parameter). With one echo every choice but [`EchoTime::Seconds`] is that
//! echo's time.

/// Which echo time a method converts the field from ppm to radians at.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EchoTime {
    /// The first echo time, `TE(1)`.
    First,
    /// The last echo time, `TE(end)`.
    Last,
    /// The echo spacing, `TE(2) − TE(1)` as the MEDI toolbox's readers set `delta_TE`; the
    /// echo time for one echo.
    Spacing,
    /// The sum of the echo times, `ΣTE`: the phase summed over the echoes, as STI Suite's
    /// `QSM_GUI` passes its combined multi-echo phase to `QSM_star` (its `sum_TE` field).
    Sum,
    /// An explicit echo time in seconds, whatever the scan's echo times are.
    Seconds(f64),
}

impl EchoTime {
    /// The echo time in seconds this choice picks from `echo_times` (seconds, in acquisition
    /// order).
    ///
    /// Errors if the choice needs the echo times and there are none, or if the result is not a
    /// finite positive time (e.g. echoes given out of order for [`EchoTime::Spacing`]).
    pub fn resolve(self, echo_times: &[f64]) -> Result<f64, String> {
        let te = match self {
            EchoTime::Seconds(te) => te,
            _ if echo_times.is_empty() => {
                return Err(format!("the {} needs the echo times; none were given", self.describe()));
            }
            EchoTime::First => echo_times[0],
            EchoTime::Last => echo_times[echo_times.len() - 1],
            EchoTime::Spacing if echo_times.len() == 1 => echo_times[0],
            EchoTime::Spacing => echo_times[1] - echo_times[0],
            EchoTime::Sum => echo_times.iter().sum(),
        };
        if te.is_finite() && te > 0.0 {
            Ok(te)
        } else {
            Err(format!(
                "the {} must be a finite time > 0 s, got {te} s (echo times {echo_times:?})",
                self.describe()
            ))
        }
    }

    /// [`resolve`](Self::resolve) for a method that has no error path: panics with `method`
    /// and the reason if the choice cannot be resolved. The pipeline checks first.
    pub(crate) fn seconds(self, echo_times: &[f64], method: &str) -> f64 {
        self.resolve(echo_times).unwrap_or_else(|e| panic!("{method}: {e}"))
    }

    fn describe(self) -> &'static str {
        match self {
            EchoTime::First => "first echo time",
            EchoTime::Last => "last echo time",
            EchoTime::Spacing => "echo spacing",
            EchoTime::Sum => "sum of the echo times",
            EchoTime::Seconds(_) => "echo time",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_each_choice() {
        let te = [0.004, 0.012, 0.020];
        assert_eq!(EchoTime::First.resolve(&te), Ok(0.004));
        assert_eq!(EchoTime::Last.resolve(&te), Ok(0.020));
        assert!((EchoTime::Spacing.resolve(&te).unwrap() - 0.008).abs() < 1e-15);
        assert!((EchoTime::Sum.resolve(&te).unwrap() - 0.036).abs() < 1e-15);
        assert_eq!(EchoTime::Seconds(0.03).resolve(&te), Ok(0.03));
    }

    #[test]
    fn one_echo_is_that_echo_for_every_scan_choice() {
        for c in [EchoTime::First, EchoTime::Last, EchoTime::Spacing, EchoTime::Sum] {
            assert_eq!(c.resolve(&[0.02]), Ok(0.02), "{c:?}");
        }
    }

    #[test]
    fn refuses_what_it_cannot_resolve() {
        assert!(EchoTime::First.resolve(&[]).unwrap_err().contains("needs the echo times"));
        assert_eq!(EchoTime::Seconds(0.01).resolve(&[]), Ok(0.01));
        assert!(EchoTime::Spacing.resolve(&[0.02, 0.01]).is_err());
        assert!(EchoTime::Seconds(0.0).resolve(&[0.01]).is_err());
        assert!(EchoTime::Seconds(f64::NAN).resolve(&[0.01]).is_err());
    }
}
