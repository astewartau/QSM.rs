mod common;
use common::{TestData, TestResult};
use qsm_core::bgremove;
use qsm_core::Grid;

fn dump(name: &str, v: &[f64]) {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(format!("/scratch/user/uqaste15/src/harp_cmp/{name}.bin"), b).unwrap();
}

#[test]
#[ignore]
fn harp_baseline() {
    std::fs::create_dir_all("/scratch/user/uqaste15/src/harp_cmp").unwrap();
    let d = TestData::load().unwrap();
    let (nx, ny, nz) = d.dims;
    let (vx, vy, vz) = d.voxel_size;
    let grid = Grid::new(nx, ny, nz, vx, vy, vz);
    let p = bgremove::HarperellaParams::default();
    println!("TE={:?} B0={}", d.echo_times, d.field_strength);
    // What run_bg_removal did on main: total field in ppm through the wrapped Laplacian.
    let (h, _) = bgremove::harperella(&d.fieldmap, &d.mask, &grid, &p, |_, _| {});
    TestResult::new("old-pipe-HARP", &h, &d.fieldmap_local, &d.mask, d.dims).print();
    dump("old_pipe_harp", &h);
    let (ih, _) = bgremove::iharperella(&d.fieldmap, &d.mask, &grid, &p, |_, _| {});
    TestResult::new("old-pipe-iHARP", &ih, &d.fieldmap_local, &d.mask, d.dims).print();
    dump("old_pipe_iharp", &ih);
    // Echo-1 wrapped phase in rad, output rad -> ppm afterwards.
    let c = 2.0 * std::f64::consts::PI * 42.5781 * d.field_strength * d.echo_times[0];
    let (h1, _) = bgremove::harperella(&d.phase_echoes[0], &d.mask, &grid, &p, |_, _| {});
    let h1: Vec<f64> = h1.iter().map(|v| v / c).collect();
    TestResult::new("old-ph1-HARP", &h1, &d.fieldmap_local, &d.mask, d.dims).print();
    dump("old_ph1_harp", &h1);
    let (ih1, _) = bgremove::iharperella(&d.phase_echoes[0], &d.mask, &grid, &p, |_, _| {});
    let ih1: Vec<f64> = ih1.iter().map(|v| v / c).collect();
    TestResult::new("old-ph1-iHARP", &ih1, &d.fieldmap_local, &d.mask, d.dims).print();
    dump("old_ph1_iharp", &ih1);
}
