//! The registry's memory figures against the peaks they were fitted to.
//!
//! [`NATIVE`] holds, per model and network input shape, the measured peak heap of one inference
//! above the resident model, from `examples/inference_memory.rs` (native, single-threaded,
//! counting allocator): the larger of a per-call run ([`super::onnx::OnnxModel::run`]) and a plan
//! built once and run with its input tensor held ([`super::onnx::OnnxModel::plan_for`]). For
//! NeXtQSM, which loads its BFR network, runs it, then loads and runs its VJP network, the larger
//! of the two above both models. [`WASM`] holds the growth of the `wasm32` heap over loading the
//! model and running it once, from the `browser_memory::net_*` cases in `tests/wasm_pipeline.rs`
//! (Node, single-threaded). Re-measure and replace the rows when `tract` or a graph changes; the
//! tests then say which figures no longer fit.

use super::{all_models, find_model, Heap, WeightStatus};

/// `(model id, network input shape, measured peak bytes of one inference above the resident model)`.
const NATIVE: &[(&str, [usize; 3], u64)] = &[
    ("bfrnet", [32, 32, 32], 160_772_954),
    ("bfrnet", [64, 64, 64], 569_337_689),
    ("bfrnet", [96, 96, 96], 1_655_505_133),
    ("bfrnet", [128, 128, 64], 1_946_048_749),
    ("bfrnet", [128, 128, 128], 3_768_453_372),
    ("bfrnet", [160, 160, 96], 4_399_083_773),
    ("bfrnet", [192, 192, 128], 8_316_850_445),
    ("bfrnet", [192, 256, 128], 11_042_558_221),
    ("xqsm", [32, 32, 32], 88_558_314),
    ("xqsm", [64, 64, 64], 286_366_495),
    ("xqsm", [96, 96, 96], 856_913_719),
    ("xqsm", [128, 128, 64], 1_011_070_101),
    ("xqsm", [128, 128, 128], 1_962_740_641),
    ("xqsm", [160, 160, 96], 2_294_211_270),
    ("xqsm", [192, 192, 128], 4_358_101_702),
    ("xqsm", [192, 256, 128], 5_793_651_398),
    ("qsmnet", [32, 32, 32], 443_264_420),
    ("qsmnet", [64, 64, 64], 646_344_879),
    ("qsmnet", [96, 96, 96], 1_181_219_844),
    ("qsmnet", [128, 128, 64], 1_324_574_337),
    ("qsmnet", [128, 128, 128], 2_208_790_148),
    ("qsmnet", [160, 160, 96], 2_510_987_895),
    ("qsmnet", [192, 192, 128], 4_400_753_911),
    ("qsmnet", [192, 256, 128], 5_714_986_231),
    ("qsmgan", [64, 64, 64], 135_572_872),
    ("lpcnn", [32, 32, 32], 25_641_065),
    ("lpcnn", [64, 64, 64], 180_011_113),
    ("lpcnn", [96, 96, 96], 593_903_721),
    ("lpcnn", [128, 128, 64], 703_381_609),
    ("lpcnn", [128, 128, 128], 1_395_507_305),
    ("lpcnn", [160, 160, 96], 1_634_222_185),
    ("lpcnn", [192, 192, 128], 3_123_691_625),
    ("lpcnn", [192, 256, 128], 4_159_750_249),
    ("ir2qsm", [32, 32, 32], 80_551_524),
    ("ir2qsm", [64, 64, 64], 255_743_144),
    ("ir2qsm", [96, 96, 96], 691_497_316),
    ("ir2qsm", [128, 128, 64], 811_204_964),
    ("ir2qsm", [128, 128, 128], 1_535_258_592),
    ("ir2qsm", [160, 160, 96], 1_790_711_776),
    ("ir2qsm", [192, 192, 128], 3_385_176_032),
    ("ir2qsm", [192, 256, 128], 4_494_274_528),
    ("autoqsm", [64, 64, 64], 129_688_028),
    ("iqsm", [32, 32, 32], 38_834_162),
    ("iqsm", [64, 64, 64], 153_933_652),
    ("iqsm", [96, 96, 96], 452_571_124),
    ("iqsm", [128, 128, 64], 532_725_024),
    ("iqsm", [128, 128, 128], 1_029_454_000),
    ("iqsm", [160, 160, 96], 1_202_411_056),
    ("iqsm", [192, 192, 128], 2_274_559_644),
    ("iqsm", [192, 256, 128], 3_022_102_172),
    ("iqsm-plus", [32, 32, 32], 60_912_247),
    ("iqsm-plus", [64, 64, 64], 306_316_393),
    ("iqsm-plus", [96, 96, 96], 962_772_585),
    ("iqsm-plus", [128, 128, 64], 1_136_670_261),
    ("iqsm-plus", [128, 128, 128], 2_239_758_789),
    ("iqsm-plus", [160, 160, 96], 2_620_554_405),
    ("iqsm-plus", [192, 192, 128], 5_002_360_753),
    ("iqsm-plus", [192, 256, 128], 6_661_040_049),
    ("iqfm", [64, 64, 64], 153_933_652),
    ("iqfm", [128, 128, 128], 1_029_454_000),
    ("nextqsm", [64, 64, 64], 261_866_120),
    ("nextqsm", [128, 128, 64], 738_163_696),
    ("nextqsm", [128, 128, 128], 1_362_044_124),
    ("nextqsm", [192, 192, 128], 2_933_238_472),
    ("nextqsm", [192, 256, 128], 3_877_045_704),
    ("modl-qsm", [32, 32, 32], 25_779_369),
    ("modl-qsm", [64, 64, 64], 181_067_052),
    ("modl-qsm", [96, 96, 96], 597_450_028),
    ("modl-qsm", [128, 128, 64], 707_583_276),
    ("modl-qsm", [128, 128, 128], 1_403_903_276),
    ("modl-qsm", [160, 160, 96], 1_644_059_948),
    ("modl-qsm", [192, 192, 128], 3_142_573_356),
    ("modl-qsm", [192, 256, 128], 4_184_923_436),
    ("susep-net", [32, 32, 32], 285_609_884),
    ("susep-net", [64, 64, 64], 750_511_934),
    ("susep-net", [96, 96, 96], 1_974_346_318),
    ("susep-net", [128, 128, 64], 2_300_857_422),
    ("susep-net", [128, 128, 128], 4_355_772_526),
    ("susep-net", [160, 160, 96], 5_065_517_678),
    ("susep-net", [192, 192, 128], 9_485_736_046),
    ("susep-net", [192, 256, 128], 12_560_292_974),
    ("r2primenet", [32, 32, 32], 128_191_738),
    ("r2primenet", [64, 64, 64], 320_442_371),
    ("r2primenet", [96, 96, 96], 838_998_871),
    ("r2primenet", [128, 128, 64], 976_913_876),
    ("r2primenet", [128, 128, 128], 1_843_959_255),
    ("r2primenet", [160, 160, 96], 2_138_620_361),
    ("r2primenet", [192, 192, 128], 3_997_387_849),
    ("r2primenet", [192, 256, 128], 5_290_255_433),
    ("chi-sepnet", [32, 32, 32], 128_462_687),
    ("chi-sepnet", [64, 64, 64], 322_548_458),
    ("chi-sepnet", [96, 96, 96], 846_085_694),
    ("chi-sepnet", [128, 128, 64], 985_311_419),
    ("chi-sepnet", [128, 128, 128], 1_860_745_406),
    ("chi-sepnet", [160, 160, 96], 2_158_290_096),
    ("chi-sepnet", [192, 192, 128], 4_035_145_520),
    ("chi-sepnet", [192, 256, 128], 5_340_596_016),
    ("hd-bet", [64, 128, 128], 1_011_603_505),
    ("hd-bet", [96, 160, 160], 2_173_939_484),
    ("hd-bet", [96, 192, 192], 3_066_700_863),
    ("hd-bet", [128, 192, 192], 4_033_547_455),
    ("hd-bet", [160, 224, 192], 5_807_159_039),
    ("hd-bet", [176, 256, 256], 9_618_107_087),
    ("rs2-net", [128, 96, 128], 2_660_730_412),
    ("mindgrab", [128, 128, 128], 1_048_211_023),
    ("mindgrab", [192, 192, 192], 3_220_336_207),
    ("mindgrab", [256, 256, 256], 7_298_772_559),
    ("synthseg", [64, 64, 64], 307_800_568),
    ("synthseg", [96, 96, 96], 888_941_434),
    ("synthseg", [128, 128, 128], 2_010_646_010),
    ("synthseg", [160, 160, 160], 3_850_759_288),
    ("synthseg", [192, 224, 160], 6_408_249_082),
    ("synthseg", [256, 256, 192], 11_628_570_106),
    ("qsmnet-plus", [64, 64, 64], 646_344_879),
    ("qsmnet-plus", [128, 128, 128], 2_208_790_148),
];

/// `(model id, network input shape, measured growth of the WASM heap over loading and one run)`.
const WASM: &[(&str, [usize; 3], u64)] = &[
    ("bfrnet", [64, 64, 64], 915_000_000),
    ("bfrnet", [96, 96, 96], 2_506_000_000),
    ("xqsm", [64, 64, 64], 499_000_000),
    ("xqsm", [128, 128, 128], 2_297_000_000),
    ("qsmnet", [64, 64, 64], 1_206_000_000),
    ("qsmnet", [96, 96, 96], 1_911_000_000),
    ("qsmgan", [64, 64, 64], 183_000_000),
    ("lpcnn", [64, 64, 64], 171_000_000),
    ("lpcnn", [128, 128, 128], 1_576_000_000),
    ("ir2qsm", [64, 64, 64], 348_000_000),
    ("ir2qsm", [128, 128, 128], 1_759_000_000),
    ("autoqsm", [64, 64, 64], 167_000_000),
    ("iqsm", [64, 64, 64], 204_000_000),
    ("iqsm", [128, 128, 128], 1_251_000_000),
    ("iqsm-plus", [64, 64, 64], 336_000_000),
    ("iqsm-plus", [128, 128, 128], 2_238_000_000),
    ("iqfm", [64, 64, 64], 204_000_000),
    ("nextqsm", [64, 64, 64], 513_373_333),
    ("nextqsm", [128, 128, 128], 1_894_373_333),
    ("modl-qsm", [64, 64, 64], 176_000_000),
    ("modl-qsm", [128, 128, 128], 1_329_000_000),
    ("susep-net", [64, 64, 64], 1_120_000_000),
    ("susep-net", [96, 96, 96], 2_467_000_000),
    ("r2primenet", [64, 64, 64], 490_000_000),
    ("r2primenet", [128, 128, 64], 1_398_000_000),
    ("chi-sepnet", [64, 64, 64], 490_000_000),
    ("chi-sepnet", [128, 128, 64], 1_398_000_000),
    ("hd-bet", [64, 128, 128], 1_461_000_000),
    ("hd-bet", [96, 160, 160], 3_012_000_000),
    ("rs2-net", [128, 96, 128], 2_669_000_000),
    ("mindgrab", [128, 128, 128], 1_015_000_000),
    ("mindgrab", [192, 192, 192], 2_560_000_000),
    ("synthseg", [96, 96, 96], 1_182_000_000),
    ("synthseg", [128, 128, 128], 2_620_000_000),
    ("qsmnet-plus", [64, 64, 64], 1_287_000_000),
];

/// A figure may sit this far above a measured peak: 10 %, or 32 MB for small shapes, where the
/// fixed part (raised so that the line covers every point) dominates.
fn close_above(figure: u64, measured: u64) -> bool {
    figure >= measured && figure - measured <= (measured / 10).max(32_000_000)
}

#[test]
fn every_model_has_memory_figures() {
    for m in all_models() {
        let mem = &m.memory;
        assert!(mem.resident_bytes > 0, "{} has no resident figure", m.id);
        assert!(mem.load_peak_bytes >= mem.resident_bytes, "{}: load peak below resident", m.id);
        assert!(mem.inference_bytes_per_voxel > 0 && mem.wasm_bytes_per_voxel > 0, "{} has no per-voxel figure", m.id);
    }
}

#[test]
fn every_hosted_model_was_measured() {
    for m in all_models().iter().filter(|m| m.status == WeightStatus::Available) {
        assert!(NATIVE.iter().any(|(id, ..)| *id == m.id), "{}: no native measurement", m.id);
        assert!(WASM.iter().any(|(id, ..)| *id == m.id), "{}: no WASM measurement", m.id);
    }
}

#[test]
fn figures_cover_measured_peaks() {
    let mut off = Vec::new();
    for &(id, shape, peak) in NATIVE {
        let figure = find_model(id).unwrap().inference_bytes(shape);
        if !close_above(figure, peak) {
            off.push(format!("native {id} {shape:?}: figure {figure}, measured {peak}"));
        }
    }
    for &(id, shape, peak) in WASM {
        let figure = find_model(id).unwrap().peak_bytes(shape, Heap::Wasm32);
        if !close_above(figure, peak) {
            off.push(format!("wasm32 {id} {shape:?}: figure {figure}, measured {peak}"));
        }
    }
    assert!(off.is_empty(), "figures off their measured peaks:\n{}", off.join("\n"));
}

#[test]
fn peak_includes_the_load_and_the_resident_model() {
    let m = find_model("qsmnet").unwrap();
    let s = [192, 256, 128];
    assert_eq!(m.peak_bytes(s, Heap::Native), m.resident_bytes() + m.inference_bytes(s));
    // LPCNN's WASM line starts at zero: for a tiny input, parsing the model sets the peak.
    let m = find_model("lpcnn").unwrap();
    assert_eq!(m.peak_bytes([2, 2, 2], Heap::Wasm32), m.memory.load_peak_bytes);
}
