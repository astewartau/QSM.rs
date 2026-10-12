//! The error an algorithm returns when it is handed input it cannot process.

/// Input an algorithm cannot process: too few echoes, echo times it cannot use, buffers that do
/// not match the grid, a singular affine, and so on. The message says what was wrong and with
/// what values, so a host can pass it straight to its user.
///
/// Algorithms return this instead of panicking. A host that wants to reject a run before any
/// work starts should still check its parameters itself (the message here is the last line of
/// defence, not a substitute for validating a config).
///
/// The pipeline stages convert it to [`PipelineError::InvalidInput`](crate::pipeline::PipelineError::InvalidInput).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputError(pub String);

impl InputError {
    /// An error with this message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputError {}

/// `Err(InputError)` unless `len == expected`, naming the buffer.
pub(crate) fn check_len(name: &str, len: usize, expected: usize) -> Result<(), InputError> {
    if len == expected {
        Ok(())
    } else {
        Err(InputError(format!("{name} has {len} values, expected {expected}")))
    }
}
