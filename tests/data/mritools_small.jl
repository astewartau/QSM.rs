# Generates tests/data/mritools_small.bin: MriResearchTools.jl (korbinian90, 771eadf) on small
# phantoms, for tests/mritools_parity.rs: `makehomogeneous`, `robustmask`, `fill_holes`,
# `calculateB0_unwrapped` (every weighting QSM.rs has) and `bipolar_correction!`.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -e JULIA_NUM_THREADS=4 -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/mritools.so --startup-file=no /data/mritools_small.jl /data/mritools_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# The phantoms are built from + - * /, round and integer xor only, so the Rust test rebuilds them
# bit for bit. Layout (little-endian, column-major over the 20x18x14 grid, voxel 1x1x1.5 mm):
# - makehomogeneous(mag; sigma = 7 mm in voxels, nbox = 15) over the grid (f64);
# - robustmask(mag), then fill_holes(holey mask; max_hole_size = 30), as bits over the grid
#   (least significant bit first, ceil(n/8) bytes each);
# - the number of mask voxels m (u32), then at the mask voxels calculateB0_unwrapped(unwrapped
#   phase, mag, TEs, type) for type in :phase_snr, :phase_var, :average, :TEs, :mag (f64, Hz);
# - bipolar_correction!(PhaseMag(phase, mag); TEs, sigma = [7, 7, 7/1.5], mask): the corrected
#   phase of each echo at the mask voxels (f64, rad).
using MriResearchTools
const M = MriResearchTools

const N = (20, 18, 14)
const VS = (1.0, 1.0, 1.5)
const TES = [4.0, 8.0, 12.0, 16.0]       # ms

function phantom()
    ne = length(TES)
    phase = zeros(Float64, N..., ne)     # unwrapped
    wrapped = zeros(Float64, N..., ne)
    mag = zeros(Float64, N..., ne)
    mask = falses(N)
    holey = falses(N)
    a, b, c = 0.46 * N[1], 0.44 * N[2], 0.42 * N[3]
    for e in 1:ne, k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x = (i - 1) - (N[1] - 1) / 2 + 0.3
        y = (j - 1) - (N[2] - 1) / 2 + 0.3
        z = (k - 1) - (N[3] - 1) / 2 + 0.3
        inside = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0
        mask[i, j, k] = inside
        # holes: two small cavities (filled) and one large one (kept)
        cav = (x - 4) * (x - 4) + y * y + z * z < 4.0 || (x + 5) * (x + 5) + (y - 3) * (y - 3) + z * z < 2.5
        big = (x + 1) * (x + 1) + (y + 5) * (y + 5) + (z - 1) * (z - 1) < 12.0
        holey[i, j, k] = inside && !cav && !big
        omega = 0.08 * x + 0.05 * y - 0.03 * z + 0.004 * x * y + 0.002 * (x * x - z * z)   # rad/ms
        phi0 = 0.5 + 0.02 * x - 0.03 * y
        h = ((i - 1) * 73856093) ⊻ ((j - 1) * 19349663) ⊻ ((k - 1) * 83492791) ⊻ ((e - 1) * 2654435761)
        noise = ((h % 1000) / 1000 - 0.5) * (inside ? 0.1 : 2.0)
        # a bipolar artefact: alternating sign by echo, linear in x
        bip = (iseven(e) ? 1.0 : -1.0) * 0.03 * x
        raw = TES[e] * omega + phi0 + noise + bip
        phase[i, j, k, e] = raw
        wrapped[i, j, k, e] = raw - 2π * round(raw / 2π)
        bias = 1.0 + 0.02 * x - 0.015 * y + 0.0005 * x * z
        tissue = inside ? (100.0 - 0.1 * (x * x + y * y + z * z)) : 4.0 + (h % 7)
        mag[i, j, k, e] = tissue * bias / (1.0 + 0.1 * (e - 1))
    end
    return phase, wrapped, mag, mask, holey
end

bits(v) = begin
    v = vec(v)
    out = zeros(UInt8, cld(length(v), 8))
    for i in eachindex(v)
        v[i] && (out[(i - 1) ÷ 8 + 1] |= UInt8(1) << ((i - 1) % 8))
    end
    out
end

phase, wrapped, mag, mask, holey = phantom()
open(ARGS[1], "w") do io
    mag1 = mag[:, :, :, 1]
    write(io, Float64.(makehomogeneous(mag1; sigma=7.0 ./ VS, nbox=15)))
    write(io, bits(robustmask(mag1)))
    write(io, bits(M.fill_holes(holey; max_hole_size=30)))
    write(io, UInt32(count(mask)))
    for type in (:phase_snr, :phase_var, :average, :TEs, :mag)
        write(io, Float64.(calculateB0_unwrapped(phase, mag, TES, type)[mask]))
    end
    img = M.PhaseMag(copy(wrapped), copy(mag))
    M.bipolar_correction!(img; TEs=TES, sigma=[7.0, 7.0, 7.0 / 1.5], mask)
    for e in axes(wrapped, 4)
        write(io, Float64.(img.phase[:, :, :, e][mask]))
    end
    println(count(mask), " mask voxels; robustmask ", count(robustmask(mag1)), ", filled ", count(M.fill_holes(holey; max_hole_size=30)), " of holey ", count(holey))
end
