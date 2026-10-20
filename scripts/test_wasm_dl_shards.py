"""Tests for wasm_dl_shards.py: the selection on real change sets, and the list check on a copy of
src/ with a new network wrapper or a new dependency added.

Run: python3 -m unittest discover -s scripts -p 'test_wasm_dl_shards.py'
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wasm_dl_shards as w  # noqa: E402

ALL = list(w.SHARDS)


class Selection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crate = w.Crate()

    def shards(self, *files):
        return w.select(list(files), self.crate)[0]

    def test_classical_only_change_runs_none(self):
        # QSM.rs #202's files: MEDI, TFI, SDF, QSMART and its validation test.
        self.assertEqual(self.shards(
            "src/inversion/medi.rs", "src/inversion/tfi.rs", "src/bgremove/sdf.rs",
            "src/pipeline/qsmart.rs", "src/utils/qsmart.rs", "tests/qsmart_validation.rs"), [])

    def test_model_registry_runs_all(self):
        self.assertEqual(self.shards("src/models/registry.rs"), ALL)

    def test_shared_fft_runs_all(self):
        self.assertEqual(self.shards("src/fft.rs"), ALL)

    def test_network_wrappers_run_their_shard(self):
        self.assertEqual(self.shards("src/inversion/xqsm.rs"), ["stages"])
        self.assertEqual(self.shards("src/bet/mindgrab.rs"), ["masking"])
        self.assertEqual(self.shards("src/segment/synthseg.rs"), ["segmentation"])

    def test_helper_reached_through_a_network(self):
        # HD-BET and SynthSeg resample; nothing in the stages shard's networks does.
        self.assertEqual(self.shards("src/utils/resample.rs"), ["masking", "segmentation"])

    def test_build_and_ci_inputs_run_all(self):
        for f in ("Cargo.lock", "Cargo.toml", "tests/wasm_pipeline.rs", "scripts/fetch_model_weights.py",
                  ".github/workflows/unit-tests.yml", ".cargo/config.toml"):
            self.assertEqual(self.shards(f), ALL, f)

    def test_declarations_only_module_roots_run_none(self):
        self.assertEqual(self.shards("src/lib.rs", "src/inversion/mod.rs", "src/utils/mod.rs"), [])

    def test_unclassified_src_file_runs_all(self):
        self.assertEqual(self.shards("src/inversion/brand_new.rs"), ALL)

    def test_outside_the_build_runs_none(self):
        self.assertEqual(self.shards("README.md", "tests/integration_tests.rs", "docs/index.md"), [])


class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(w.ROOT / "src", self.tmp / "src")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def errors(self):
        return w.check(w.Crate(self.tmp))[0]

    def test_tree_as_committed_passes(self):
        self.assertEqual(self.errors(), [])

    def test_unlisted_network_wrapper_fails(self):
        (self.tmp / "src/inversion/foonet.rs").write_text(
            "use crate::models::onnx::{OnnxModel, Tensor};\n"
            "pub fn foonet(bytes: &[u8]) { let _ = OnnxModel::from_bytes(bytes); }\n")
        mod_rs = self.tmp / "src/inversion/mod.rs"
        mod_rs.write_text(mod_rs.read_text() + "\npub mod foonet;\n")
        errors = self.errors()
        self.assertTrue(any("src/inversion/foonet.rs uses the model machinery" in e for e in errors), errors)

    def test_classical_file_a_network_starts_using_fails(self):
        xqsm = self.tmp / "src/inversion/xqsm.rs"
        xqsm.write_text(xqsm.read_text() + "\nuse crate::utils::padding::pad3d;\n")
        errors = self.errors()
        self.assertTrue(any("src/utils/padding.rs is CLASSICAL but the stages shard reaches it" in e
                            for e in errors), errors)

    def test_neutral_file_with_code_fails(self):
        lib = self.tmp / "src/lib.rs"
        lib.write_text(lib.read_text() + "\npub fn version() -> u32 { 1 }\n")
        errors = self.errors()
        self.assertTrue(any("src/lib.rs is NEUTRAL" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
