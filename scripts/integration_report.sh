#!/usr/bin/env bash
# Build the integration-test results report (markdown) from the per-test metrics the jobs in
# .github/workflows/integration-tests.yml upload.
#
# Usage: scripts/integration_report.sh [--figures]
#   Reads metrics/<test>.csv (RESULT: lines), slices/*.bin (to decide which sections ran) and,
#   with --figures, figures/*.png (the montages scripts/render_slices.py draws). Writes the
#   report to stdout. COVERAGE and TESTS_PASSED come from the environment (unit-test job).
#
# The job summary and the PR comment are both this report — the comment with --figures — so a
# new result is added in one place. Every test the workflow can run has a row (or, for pass/fail
# characterisation tests, a section) here: adding a test means adding it to the workflow's
# `all_*` list, the path map and this report.
set -uo pipefail

WITH_FIGURES=false
[ "${1:-}" = "--figures" ] && WITH_FIGURES=true

# First RESULT payload of a metrics file (without the "RESULT:" prefix).
first_result() { awk '/^RESULT:/{print substr($0,8); exit}' "$1"; }
# Every RESULT payload of a metrics file, one per line.
all_results() { { grep "^RESULT:" "$1" || true; } | sed 's/^RESULT://'; }
fmt_time() { printf '%.2fs' "${1:-0}"; }

# Field-map table (RMSE / NRMSE / Corr / XSIM) over full test names. Every RESULT row of each
# file becomes a table row.
emit_table() {
  local has_any=false
  for test in "$@"; do [ -f "metrics/${test}.csv" ] && has_any=true; done
  [ "$has_any" = false ] && return 1
  echo "| Method | RMSE | NRMSE (range) | NRMSE% (χ) | Correlation | XSIM | Time |"
  echo "|--------|------|---------------|------------|-------------|------|------|"
  for test in "$@"; do
    file="metrics/${test}.csv"
    [ -f "$file" ] || continue
    all_results "$file" | while IFS=',' read -r name rmse nrmse nrmse_ch corr xsim time; do
      if [ "$rmse" = "-" ] || [ -z "$rmse" ]; then
        echo "| ${name} | :x: Failed | - | - | - | - | - |"
      else
        echo "| ${name} | ${rmse} | ${nrmse} | ${nrmse_ch} | ${corr} | ${xsim} | $(fmt_time "$time") |"
      fi
    done
  done
  echo ""
}

# Susceptibility-map rows, tag appended to the method name. RESULT format:
# name,nrmse%,detrend%,tissue%,blood%,dgm%,dgm_lin,calc_dev,streak,corr,xsim,time
emit_chi_rows() {
  local tag="$1"; shift
  for test in "$@"; do
    file="metrics/${test}.csv"
    [ -f "$file" ] || continue
    all_results "$file" | while IFS=',' read -r name nrmse dt tissue blood dgm dgm_lin calc_dev streak corr xsim time; do
      if [ "$nrmse" = "-" ] || [ -z "$nrmse" ]; then
        echo "| ${name}${tag} | :x: Failed | - | - | - | - | - | - | - | - | - | - |"
      else
        echo "| ${name}${tag} | ${nrmse} | ${dt} | ${tissue} | ${blood} | ${dgm} | ${dgm_lin} | ${calc_dev} | ${streak} | ${corr} | ${xsim} | $(fmt_time "$time") |"
      fi
    done
  done
}
chi_header() {
  echo "| Method | NRMSE% | Detrend% | Tissue% | Blood% | DGM% | DGM Lin | Calc Dev | Streak | Corr | XSIM | Time |"
  echo "|--------|--------|----------|---------|--------|------|---------|----------|--------|------|------|------|"
}
emit_chi_table() {
  local rows
  rows=$(emit_chi_rows "" "$@")
  [ -z "$rows" ] && return 1
  chi_header
  echo "$rows"
  echo ""
}

# Brain-mask rows. RESULT format: name,dice,-,-,time
emit_mask_table() {
  local has_any=false
  for test in "$@"; do [ -f "metrics/${test}.csv" ] && has_any=true; done
  [ "$has_any" = false ] && return 1
  echo "| Method | Dice | Time |"
  echo "|--------|------|------|"
  for test in "$@"; do
    file="metrics/${test}.csv"
    [ -f "$file" ] || continue
    all_results "$file" | while IFS=',' read -r name dice _ _ time; do
      if [ "$dice" = "-" ] || [ -z "$dice" ]; then
        echo "| ${name} | :x: Failed | - |"
      else
        echo "| ${name} | ${dice} | $(fmt_time "$time") |"
      fi
    done
  done
  echo ""
}

# χ-separation rows. RESULT format: label,corr,xsim,nrmse%,time
emit_chisep_rows() {
  for f in "$@"; do
    [ -f "$f" ] || continue
    all_results "$f" | while IFS=',' read -r name corr xsim nrmse time; do
      if [ "$corr" = "-" ] || [ -z "$corr" ]; then
        echo "| ${name} | :x: Failed | - | - | - |"
      else
        echo "| ${name} | ${corr} | ${xsim} | ${nrmse}% | $(fmt_time "$time") |"
      fi
    done
  done
}

# A montage can be referenced by more than one section (stage_dipole carries the single-
# orientation, deep-learning and multi-orientation rows), and which sections run depends on what
# the change touched. Emitting once, from whichever section gets there first, means the figure
# still appears when only one of them ran without being repeated when several did.
EMITTED_FIGURES=""
emit_figures() {
  [ "$WITH_FIGURES" = true ] || return 0
  for slug in "$@"; do
    case " $EMITTED_FIGURES " in *" $slug "*) continue ;; esac
    if [ -f "figures/${slug}.png" ]; then
      echo "![${slug}](figures/${slug}.png)"
      echo ""
      EMITTED_FIGURES="$EMITTED_FIGURES $slug"
    fi
  done
}

# Utilities-style "name ... time" row, time being the last field.
util_row() {
  local f="$1"
  [ -f "$f" ] || return 0
  line=$(first_result "$f")
  name=${line%%,*}
  time=${line##*,}
  if [ "$time" = "-" ] || [ -z "$time" ]; then
    echo "| ${name} | :x: Failed |"
  else
    echo "| ${name} | $(fmt_time "$time") |"
  fi
}

echo "## QSM-Core Test Results"
echo ""
echo "### Unit Tests"
echo "- **Tests passed:** ${TESTS_PASSED:-N/A}"
echo "- **Code coverage:** ${COVERAGE:-N/A}%"
echo ""

if TABLE=$(emit_table test_bgremove_sharp test_bgremove_resharp test_bgremove_vsharp test_bgremove_vsharp_no_deconvolve test_bgremove_pdf test_bgremove_ismv test_bgremove_lbv test_bgremove_msmv); then
  echo "### Background Field Removal"
  echo "*Compared against ground truth local field map*"
  echo ""
  echo "$TABLE"
  echo ""
  emit_figures stage_bfr
fi

if TABLE=$(emit_table test_pipeline_harperella test_pipeline_iharperella); then
  echo "### Combined Phase Unwrapping + Background Removal"
  echo "*Compared against ground truth local field map (input: wrapped phase)*"
  echo ""
  echo "$TABLE"
  echo ""
fi

rows=$(
  emit_chi_rows "" test_inversion_tkd test_inversion_tkd_cropped test_inversion_tsvd test_inversion_tikhonov test_inversion_tv test_inversion_rts test_inversion_ilsqr test_inversion_medi test_inversion_nltv test_inversion_ndi test_inversion_fansi test_inversion_fansi_tgv test_inversion_l1qsm test_inversion_whqsm test_inversion_hdqsm test_inversion_amp_pe test_inversion_lsqr test_inversion_heidi
  emit_chi_rows " — BFR + inversion" test_combined_tgv test_combined_tfi
  emit_chi_rows " — full pipeline" test_pipeline_tgv test_pipeline_tgv_single_echo test_pipeline_qsmart test_pipeline_qsmart_tikhonov test_pipeline_two_pass
)
if [ -n "$rows" ]; then
  echo "### Dipole Inversion"
  echo "*Every method that produces a susceptibility map, against ground truth. Plain rows take the local field; **— BFR + inversion** rows take the total field and remove background themselves; **— full pipeline** rows start from wrapped phase.*"
  echo ""
  chi_header
  echo "$rows"
  echo ""
  emit_figures stage_dipole
fi

if [ -f "slices/register_aligned.bin" ] || [ -f "metrics/test_multiorient_register_capture_range.csv" ]; then
  echo "### Orientation Co-registration"
  echo "*Orientations that are not already on one grid are rigidly co-registered before they are combined, and each one's B0 direction is read off the recovered rotation. The difference panels are the check: **as acquired** should be bright with rotational mismatch and **after registration** near-flat, both on one scale. Equally bright means the transform went on the wrong way round.*"
  echo ""
  if [ -f "metrics/test_multiorient_register_capture_range.csv" ]; then
    line=$(first_result "metrics/test_multiorient_register_capture_range.csv")
    IFS=',' read -r name converged failed worst time <<< "$line"
    echo "| Measure | Value |"
    echo "|---------|-------|"
    if [ "$converged" = "-" ]; then
      echo "| Capture range | :x: Failed |"
    else
      echo "| Capture range (swept to 70°) | ${converged}° |"
      if [ "$failed" = "none" ]; then
        echo "| First failure | none in the swept range |"
      else
        echo "| First failure | ${failed}° |"
      fi
      echo "| Worst rotation error inside the range | ${worst}° |"
      echo "| Time | $(fmt_time "$time") |"
    fi
    echo ""
  fi
  echo ""
  emit_figures stage_registration
fi

if [ -f "metrics/test_motion_correction.csv" ]; then
  echo "### Motion Correction"
  echo "*A four-echo series in which the head drifts between echoes, registered back to the first echo in the complex domain and re-fitted. Scored against the field map the same pipeline produces from the same echoes with no motion — not against ground truth, which a total field map only matches up to the harmonic field unwrapping leaves behind.*"
  echo ""
  recovery=$(awk '/^RESULT:Motion recovery,/{print substr($0,8); exit}' "metrics/test_motion_correction.csv")
  if [ -n "$recovery" ]; then
    IFS=',' read -r _ rot trans bdir disp rtime <<< "$recovery"
    echo "| Recovered motion | Worst over the series |"
    echo "|------------------|-----------------------|"
    echo "| Rotation error | ${rot}° |"
    echo "| Translation error | ${trans} mm |"
    echo "| B0 direction error | ${bdir}° |"
    echo "| Simulated tissue displacement | ${disp} mm |"
    echo "| Time | $(fmt_time "$rtime") |"
    echo ""
  fi
  have_rows=false
  for variant in Uncorrected Corrected; do
    grep -q "^RESULT:${variant}," "metrics/test_motion_correction.csv" && have_rows=true
  done
  echo "| Field map | NRMSE (range) | Correlation | XSIM | Time |"
  echo "|-----------|---------------|-------------|------|------|"
  if [ "$have_rows" = true ]; then
    for variant in Uncorrected Corrected; do
      line=$(awk -v p="^RESULT:${variant}," '$0 ~ p {print substr($0,8); exit}' "metrics/test_motion_correction.csv")
      [ -n "$line" ] || continue
      IFS=',' read -r name nrmse corr xsim mtime <<< "$line"
      echo "| ${name} | ${nrmse} | ${corr} | ${xsim} | $(fmt_time "$mtime") |"
    done
  else
    echo "| Motion correction | :x: Failed | - | - | - |"
  fi
  echo ""
  emit_figures stage_motion
fi

if [ -f "metrics/test_robust_echo_combination.csv" ]; then
  echo "### Motion-Robust Echo Combination"
  echo "*One echo corrupted the way spin history or a mistimed shot does — a slab losing most of*"
  echo "*its magnitude and gaining a smooth phase error — which is deliberately not a rigid*"
  echo "*transform, so registration cannot reach it. Scored against the ordinary fit of the*"
  echo "*uncorrupted echoes. The last row is the cost of running this when nothing is wrong.*"
  echo ""
  eq=$(awk '/^RESULT:Echo quality,/{print substr($0,8); exit}' "metrics/test_robust_echo_combination.csv")
  if [ -n "$eq" ]; then
    IFS=',' read -r _ spread clean_spread echo_no frac <<< "$eq"
    echo "| Detection | Value |"
    echo "|-----------|-------|"
    echo "| Score spread, corrupted series | ${spread}x |"
    echo "| Score spread, clean series | ${clean_spread}x |"
    echo "| Echo flagged | ${echo_no} (the corrupted one) |"
    echo "| Fraction of it downweighted | ${frac} |"
    echo ""
  fi
  have_rows=false
  for variant in "Plain fit" "Robust fit" "Robust on clean"; do
    grep -q "^RESULT:${variant}," "metrics/test_robust_echo_combination.csv" && have_rows=true
  done
  echo "| Field map | NRMSE (range) | Correlation | XSIM | Time |"
  echo "|-----------|---------------|-------------|------|------|"
  if [ "$have_rows" = true ]; then
    for variant in "Plain fit" "Robust fit" "Robust on clean"; do
      line=$(awk -v p="^RESULT:${variant}," '$0 ~ p {print substr($0,8); exit}' "metrics/test_robust_echo_combination.csv")
      [ -n "$line" ] || continue
      IFS=',' read -r name nrmse corr xsim rtime <<< "$line"
      echo "| ${name} | ${nrmse} | ${corr} | ${xsim} | $(fmt_time "$rtime") |"
    done
  else
    echo "| Robust echo combination | :x: Failed | - | - | - |"
  fi
  echo ""
  emit_figures stage_echo_quality
fi

if TABLE=$(emit_chi_table test_multiorient_cosmos test_multiorient_register test_multiorient_sti); then
  echo "### Multi-Orientation Inversion"
  echo "*COSMOS and STI consume several orientations, so these are not like-for-like with the single-orientation methods above. Compared against ground truth susceptibility map; STI reports mean magnetic susceptibility.*"
  echo ""
  echo "$TABLE"
  echo ""
  # Its own montage, next to its own table. COSMOS and STI also appear in stage_dipole above,
  # where they sit against every other method on a shared scale; that answers "how does it
  # compare", this answers "did it work". Not sti_msa: on this isotropic phantom the anisotropy
  # is ~1e-17, so its figure would be three blank panels.
  emit_figures stage_multiorient
fi


if [ -f "figures/oblique_sagittal_chi.png" ] || [ -f "slices/oblique_sagittal_chi.bin" ]; then
  echo "### Geometry — Sagittal Anisotropic Oblique"
  echo "*A 1x1x4 mm sagittal stack tilted 18 deg about S-I and 12 deg about A-P, synthesised from the phantom, reconstructed through the cardinal-resample pathway and mapped back to its own voxel space. No fixed-matrix phantom has this geometry, and it is the only shape of data that exercises how axial_grid_for assigns its spacings. Pass/fail only; see the job log for the correlation and S-I detail numbers. The panel labels are the renderer's generic voxel-axis names; because this stack is stored sagittally, its 'Axial' panel is the sagittal anatomical plane.*"
  echo ""
  emit_figures oblique_sagittal_truth oblique_sagittal_chi
fi

if [ -f "metrics/multislice_2d.csv" ]; then
  echo "### 2D Multi-Slice Unwrapping"
  echo "*Slice-wise unwrapping and the slice-gap refusal on the 2D multi-slice sessions (correlation with the ground-truth field, or counts of slice-to-slice phase jumps).*"
  echo ""
  echo "| Measure | Value |"
  echo "|---------|-------|"
  all_results "metrics/multislice_2d.csv" | tr ' ' '\n' | grep '=' | while IFS='=' read -r k v; do
    echo "| \`${k}\` | ${v} |"
  done
  echo ""
fi

if TABLE=$(emit_chi_table test_dl_qsmgan test_dl_lpcnn test_dl_ir2qsm test_dl_modl_qsm test_dl_xqsm test_dl_qsmnet test_dl_qsmnet_plus test_dl_nextqsm test_dl_autoqsm test_dl_iqsm test_dl_iqsm_plus); then
  echo "### Deep Learning — Dipole Inversion"
  echo "*Compared against ground truth susceptibility map. Local field in, except NeXtQSM and AutoQSM (total field) and iQSM / iQSM+ (wrapped phase). The phantom is out of distribution for every network, so these are regression numbers, not accuracy claims.*"
  echo ""
  echo "$TABLE"
  echo ""
  emit_figures stage_dipole
fi

if TABLE=$(emit_table test_dl_iqfm test_dl_bfrnet); then
  echo "### Deep Learning — Field Mapping and Background Removal"
  echo "*iQFM: wrapped phase → local field; BFRnet: total field → local field. Both against the ground truth local field.*"
  echo ""
  echo "$TABLE"
  echo ""
  echo "_Maps shown in the Local field montage, alongside the background-field-removal methods they are scored against._"
  echo ""
  emit_figures stage_bfr
fi

if TABLE=$(emit_mask_table test_bet test_dl_hdbet test_dl_mindgrab test_dl_rs2net test_pipeline_masking); then
  echo "### Brain Extraction and Masking"
  echo "*Compared against ground truth mask. RS2-Net is a rodent model, run on the phantom at a tenth of its voxel size; its Dice is informational. The pipeline masking rows run \`run_masking\` with each generator and refinement.*"
  echo ""
  echo "$TABLE"
  emit_figures bet
fi

if [ -f "metrics/test_dl_synthseg.csv" ] || [ -f "metrics/test_dl_synthseg_window.csv" ]; then
  echo "### Segmentation (SynthSeg)"
  echo "*Brain Dice: every non-background label against the ground-truth mask (the phantom's own segmentation uses a different label set). Sliding window is the memory-bounded inference the browser runs; the last row is its agreement with the whole-volume segmentation.*"
  echo ""
  echo "| Method | Brain Dice | Structures found | Time |"
  echo "|--------|------------|------------------|------|"
  for t in test_dl_synthseg test_dl_synthseg_window; do
    f="metrics/${t}.csv"
    [ -f "$f" ] || continue
    line=$(first_result "$f")
    IFS=',' read -r name dice found total time <<< "$line"
    if [ "$dice" = "-" ] || [ -z "$dice" ]; then
      echo "| ${name:-$t} | :x: Failed | - | - |"
    else
      echo "| ${name} | ${dice} | ${found}/${total} | $(fmt_time "$time") |"
    fi
  done
  agree=$(awk '/^RESULT:SynthSeg window vs whole,/{print substr($0,8); exit}' metrics/test_dl_synthseg_window.csv 2>/dev/null || true)
  if [ -n "$agree" ]; then
    IFS=',' read -r _ mean min <<< "$agree"
    echo "| Sliding window vs whole volume | mean structure Dice ${mean} (lowest ${min}) | - | - |"
  fi
  echo ""
  emit_figures segmentation
fi

if [ -f "slices/unwrap_raw_romeo.bin" ] || [ -f "slices/unwrap_raw_laplacian_neumann.bin" ] || [ -f "slices/unwrap_raw_bestpath.bin" ] || [ -f "slices/unwrap_raw_prelude.bin" ] || [ -f "metrics/test_pipeline_romeo_b0.csv" ]; then
  echo "### Phase Unwrapping + B0 Estimation"
  echo "*MCPC-3D-S, then each unwrapper → total field map, beside the ground truth. These differ by an arbitrary harmonic field, which is expected of unwrappers; the like-for-like comparison is below.*"
  echo ""
  if TABLE=$(emit_table test_pipeline_romeo_b0); then
    echo "$TABLE"
  fi
  emit_figures stage_unwrap
  echo ""
fi

if TABLE=$(emit_table test_pipeline_unwrap_bfr_romeo test_pipeline_unwrap_bfr_laplacian_neumann test_pipeline_unwrap_bfr_bestpath test_pipeline_unwrap_bfr_prelude test_pipeline_unwrap_bfr_stage_laplacian); then
  echo "### Unwrapper Comparison (local field, after background removal)"
  echo "*Each unwrapper → field map → V-SHARP → local field, vs ground truth.*"
  echo ""
  echo "Total field maps differ between unwrappers by an arbitrary harmonic field, so they are compared after background removal. ROMEO, Laplacian, best path and PRELUDE should land close. The last row drives the shared \`run_field_mapping\` stage, and is what catches a config selecting the wrong variant."
  echo ""
  echo "$TABLE"
fi

util_files=(metrics/test_bias_correction.csv metrics/test_swi.csv metrics/test_smwi.csv metrics/test_r2star_t2star.csv metrics/test_dl_r2primenet.csv)
have_util=false
for f in "${util_files[@]}"; do [ -f "$f" ] && have_util=true; done
if [ "$have_util" = true ]; then
  echo "### Utilities"
  echo ""
  echo "| Method | Time |"
  echo "|--------|------|"
  for f in "${util_files[@]}"; do util_row "$f"; done
  echo ""
  emit_figures bias_correction stage_supplementary
fi

chisep_files=(metrics/chisep_qsmci.csv metrics/chisep_relaxometry_qsmci.csv metrics/chisep_xsepnet.csv metrics/chisep_r2primenet.csv)
rows=$(emit_chisep_rows "${chisep_files[@]}")
if [ -n "$rows" ]; then
  echo "### χ-Separation"
  echo "*Paramagnetic (χ+) / diamagnetic (χ−) source maps vs ground truth (QSM-CI chisep phantom). The R2PRIMEnet row is its R2′ against the phantom's reference R2′ (a 3 T network on a 7 T phantom: off-distribution by construction).*"
  echo ""
  echo "| Map | Correlation | XSIM | NRMSE% | Time |"
  echo "|-----|-------------|------|--------|------|"
  echo "$rows"
  echo ""
  emit_figures stage_chisep stage_supplementary
fi

rows=$(emit_chisep_rows metrics/chisep_orientation.csv)
if [ -n "$rows" ]; then
  echo "### χ-sepnet Orientation Sensitivity"
  echo "*χ-sepnet run on the phantom with its axes relabelled, relabelled back, and scored against the untouched run (not ground truth).*"
  echo ""
  echo "| Relabelling | Correlation | XSIM | NRMSE% | Time |"
  echo "|-------------|-------------|------|--------|------|"
  echo "$rows"
  echo ""
fi

# Every test that ran, pass or fail — including the pass/fail-only ones (COSMOS coverage and
# noise, χ-sepnet orientation, geometry, 2D multi-slice) that have no metrics row above. Each
# job's "Save metrics" step records its Run step's outcome as a STATUS: line.
status_rows=""
for f in metrics/*.csv; do
  [ -f "$f" ] || continue
  t=$(basename "$f" .csv)
  st=$(awk -F: '/^STATUS:/{print $2; exit}' "$f")
  case "$st" in
    success) icon=":white_check_mark:" ;;
    failure) icon=":x:" ;;
    "") icon=":grey_question:" ;;
    *) icon=":warning: ${st}" ;;
  esac
  status_rows="${status_rows}| \`${t}\` | ${icon} |"$'\n'
done
if [ -n "$status_rows" ]; then
  n_total=$(printf '%s' "$status_rows" | grep -c '^|' || true)
  n_fail=$(printf '%s' "$status_rows" | grep -c ':x:' || true)
  echo "<details><summary><b>All integration tests that ran</b> (${n_total}; ${n_fail} failed)</summary>"
  echo ""
  echo "| Test | Status |"
  echo "|------|--------|"
  printf '%s' "$status_rows"
  echo ""
  echo "</details>"
  echo ""
fi
