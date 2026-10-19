#!/usr/bin/env python3
"""Fetch hosted model weights into a directory, by the URLs and SHA-256s in the model registry.

    fetch_model_weights.py DIR ID [ID ...]     fetch each model's files into DIR
    fetch_model_weights.py --key ID [ID ...]   print a cache key for those files

Each file lands in DIR under its registry name, which is where the wasm harness
(tests/wasm_pipeline.rs, `$QSM_WEIGHTS_DIR`) looks for it. A file already in DIR with the
registry's checksum is kept; anything else is downloaded and checked. The cache key is a
digest of the files' registry checksums, so it changes exactly when a file to fetch does.
`all` stands for every model with hosted weights.
"""

import hashlib
import pathlib
import re
import sys
import time
import urllib.request

REGISTRY = pathlib.Path(__file__).resolve().parent.parent / "src" / "models" / "registry.rs"


def registry():
    """{model id: [(file name, url, sha256), ...]} for every model with hosted weights."""
    text = REGISTRY.read_text()
    models = {}
    for block in text.split("ModelSpec {")[1:]:
        mid = re.search(r'id: "([^"]+)"', block).group(1)
        files = re.findall(r'name: "([^"]+\.onnx)",\s*url: "([^"]+)",\s*sha256: "([0-9a-f]{64})"', block)
        if files:
            models[mid] = files
    return models


def files_for(ids):
    models = registry()
    if ids == ["all"]:
        ids = list(models)
    missing = [i for i in ids if i not in models]
    if missing:
        sys.exit(f"no hosted weights in the registry for: {', '.join(missing)}")
    return [f for i in ids for f in models[i]]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, url, want in files:
        path = directory / name
        if path.exists() and sha256(path) == want:
            print(f"{name}: cached")
            continue
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(url, path)
                break
            except OSError as e:
                if attempt == 4:
                    raise
                print(f"{name}: {e}, retrying")
                time.sleep(2 ** attempt)
        got = sha256(path)
        if got != want:
            sys.exit(f"{name}: SHA-256 {got}, the registry says {want}")
        print(f"{name}: fetched")


def main(argv):
    if len(argv) >= 2 and argv[0] == "--key":
        sums = "".join(sorted(f"{name}:{sha}\n" for name, _, sha in files_for(argv[1:])))
        print(hashlib.sha256(sums.encode()).hexdigest()[:32])
    elif len(argv) >= 2:
        fetch(pathlib.Path(argv[0]).expanduser(), files_for(argv[1:]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
