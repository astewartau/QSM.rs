# Missing Algorithms

Algorithms present in reference toolboxes but not yet in QSM.rs.

## From sunhongfu-qsm

### ~~RESHARP (Regularized SHARP)~~ — DONE
Implemented in `src/bgremove/resharp.rs`.

### ~~iHARPERELLA~~ — DONE
Implemented in `src/bgremove/iharperella.rs` (STI Suite 3.0's `iHARPERELLA`).

### Polynomial Background Fitting — LOW PRIORITY
- **Source**: `background_field_removal/poly2d.m`, `poly3d.m`, `poly3d_nonlinear.m`
- **Description**: 2nd-4th order polynomial fit for low-order background field variations.
- **Why low priority**: Less sophisticated than existing methods (LBV, V-SHARP, PDF). Mainly useful for quick-and-dirty removal or as a preprocessing step.

## Dipole Inversion

### TVDI (NLCG-based TV) — LOW PRIORITY
- **Source**: `dipole_inversion/tvdi.m`, `nlcg.m`
- **Description**: Total Variation dipole inversion using non-linear conjugate gradient with backtracking line search.
- **Why low priority**: QSM.rs already has TV-ADMM which solves the same L1-regularized problem with a different optimizer.

## Coil Combination

### POEM (Phase-Offsets Estimation from Multi-echoes) — LOW PRIORITY
- **Source**: `coil_combination/poem.m`, `poem_bi3.m`, `poem_lr.m`
- **Description**: Multi-channel coil combination using phase offsets estimated from multiple echoes.
- **Why low priority**: Preprocessing step, different category from core QSM algorithms.

### Adaptive Coil Combination — LOW PRIORITY
- **Source**: `coil_combination/adaptive_cmb.m`
- **Description**: Eigenvalue-based adaptive sensitivity estimation (Walsh method, MRM 2000).
- **Why low priority**: Same reasoning as POEM.

## Phase Unwrapping

### Quality-Guided Unwrapping — LOW PRIORITY
- **Source**: `phase_unwrapping/qualityGuidedUnwrapping.m`
- **Description**: 3D phase unwrapping guided by second-difference quality map with region-growing.
- **Why low priority**: ROMEO (already implemented) is generally considered superior.

### 3D SRNCP — LOW PRIORITY
- **Source**: `phase_unwrapping/3DSRNCP.m`
- **Description**: Statistical region-growing based phase unwrapping.
- **Why low priority**: Same reasoning as quality-guided.

## From STISuite V3.0

### ~~QSM_STAR (Streaking Artifact Reduction)~~ — DONE
Implemented in `src/inversion/star.rs` (STI Suite 3.0's `QSM_star`).

### ~~STI (Susceptibility Tensor Imaging)~~ — DONE
Implemented in `src/inversion/sti.rs`.
