//! Native weight download and on-disk cache (`download` feature).
//!
//! Fetches a model's weight files over HTTP, verifies their SHA-256, and stores
//! them under [`super::cache_dir`]. This is the "download on use" path for native
//! hosts (e.g. QSMxT). WASM hosts do not use this module — they fetch weights in
//! JavaScript and feed the bytes to [`super::onnx`] directly.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use super::{cache_dir, cache_path, resolve_local, ModelSpec, WeightFile};

/// Error from resolving or downloading model weights.
#[derive(Debug)]
pub enum DownloadError {
    /// The model has no hosted URL yet (Pending) and no local copy was found.
    NotHosted { model: String, file: String },
    /// Network/transport failure.
    Http(String),
    /// Filesystem failure.
    Io(std::io::Error),
    /// Downloaded bytes did not match the expected SHA-256.
    ChecksumMismatch { file: String, expected: String, got: String },
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotHosted { model, file } => write!(
                f,
                "weights for '{model}' are not hosted yet (file '{file}'). \
                 Supply it locally via $QSM_MODEL_DIR."
            ),
            Self::Http(msg) => write!(f, "download failed: {msg}"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::ChecksumMismatch { file, expected, got } => write!(
                f,
                "checksum mismatch for '{file}': expected {expected}, got {got}"
            ),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Lowercase hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Ensure a single weight file is present locally, downloading it if needed, and
/// return its path.
///
/// Resolution: `$QSM_MODEL_DIR` / cache hit (checksum-validated) → return;
/// otherwise download from [`WeightFile::url`], verify, and cache.
pub fn ensure_file(model_id: &str, file: &WeightFile) -> Result<PathBuf, DownloadError> {
    // 1. Local override or a good cache entry.
    if let Some(path) = resolve_local(file) {
        if file.sha256.is_empty() || checksum_ok(&path, file.sha256)? {
            return Ok(path);
        }
        // A cached file with the wrong hash is stale — re-fetch it.
    }

    // 2. Must have a URL to fetch.
    if file.url.is_empty() {
        return Err(DownloadError::NotHosted {
            model: model_id.to_string(),
            file: file.name.to_string(),
        });
    }

    let bytes = http_get(file.url)?;

    // 3. Verify before trusting.
    if !file.sha256.is_empty() {
        let got = sha256_hex(&bytes);
        if got != file.sha256 {
            return Err(DownloadError::ChecksumMismatch {
                file: file.name.to_string(),
                expected: file.sha256.to_string(),
                got,
            });
        }
    }

    // 4. Write atomically into the cache (temp file + rename).
    let dir = cache_dir();
    fs::create_dir_all(&dir)?;
    let final_path = cache_path(file);
    let tmp = dir.join(format!("{}.part", file.name));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &final_path)?;
    Ok(final_path)
}

/// Ensure every weight file for a model is present, returning their paths in
/// [`ModelSpec::files`] order.
pub fn ensure_model(spec: &ModelSpec) -> Result<Vec<PathBuf>, DownloadError> {
    spec.files.iter().map(|f| ensure_file(spec.id, f)).collect()
}

/// Read the primary (first) weight file of a model into memory. Convenience for
/// handing bytes straight to [`super::onnx::OnnxModel::load`].
pub fn primary_bytes(spec: &ModelSpec) -> Result<Vec<u8>, DownloadError> {
    let file = spec.files.first().ok_or_else(|| DownloadError::NotHosted {
        model: spec.id.to_string(),
        file: "<none>".to_string(),
    })?;
    let path = ensure_file(spec.id, file)?;
    Ok(fs::read(path)?)
}

fn checksum_ok(path: &PathBuf, expected: &str) -> Result<bool, DownloadError> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes) == expected)
}

fn http_get(url: &str) -> Result<Vec<u8>, DownloadError> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| DownloadError::Http(e.to_string()))?;
    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| DownloadError::Http(e.to_string()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        // SHA-256 of the empty string.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // SHA-256 of "abc".
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
