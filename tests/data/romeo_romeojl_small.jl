# Generates tests/data/romeo_romeojl_small.bin: ROMEO.jl's multi-echo `unwrap` (08776d4, v1.6.0)
# and MriResearchTools.jl's `mcpc3ds` (771eadf, v3.8.0) on a small multi-echo phantom, in Float64
# (the API follows the data's precision), for tests/romeo_romeojl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -e JULIA_NUM_THREADS=4 -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/mritools.so --startup-file=no /data/romeo_romeojl_small.jl /data/romeo_romeojl_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# The phantom is built from + - * /, round and integer xor only, so the Rust test rebuilds it bit
# for bit. Layout of the output (little-endian):
# - per unwrapping case in CASES order, per echo, per mask voxel (column-major): the number of
#   turns ROMEO.jl added, round((unwrapped - wrapped) / 2π), as i8;
# - for mcpc3ds (sigma 7 voxels), per echo, the corrected phase at the mask voxels (f64).
using ROMEO, MriResearchTools

const N = (28, 24, 18)
const TES = [4.0, 8.0, 12.0]

function phantom()
    ne = length(TES)
    phase = zeros(Float64, N..., ne)
    mag = zeros(Float64, N..., ne)
    mask = falses(N)
    a, b, c = 0.46 * N[1], 0.44 * N[2], 0.42 * N[3]
    for e in 1:ne, k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x = (i - 1) - (N[1] - 1) / 2 + 0.3
        y = (j - 1) - (N[2] - 1) / 2 + 0.3
        z = (k - 1) - (N[3] - 1) / 2 + 0.3
        inside = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0
        mask[i, j, k] = inside
        omega = 0.08 * x + 0.05 * y - 0.03 * z + 0.004 * x * y + 0.002 * (x * x - z * z)   # rad/ms
        phi0 = 0.5 + 0.02 * x - 0.03 * y
        h = ((i - 1) * 73856093) ⊻ ((j - 1) * 19349663) ⊻ ((k - 1) * 83492791) ⊻ ((e - 1) * 2654435761)
        noise = ((h % 1000) / 1000 - 0.5) * (inside ? 0.4 : 4.0)
        raw = TES[e] * omega + phi0 + noise
        phase[i, j, k, e] = raw - 2π * round(raw / 2π)
        mag[i, j, k, e] = inside ? (100.0 - 0.1 * (x * x + y * y + z * z)) / (1.0 + 0.1 * (e - 1)) : 5.0 + (h % 7)
    end
    return phase, mag, mask
end

phase, mag, mask = phantom()
const CASES = [
    ("template, API defaults", (; )),
    ("template, the CLI's weights and temporal setting", (; weights=:romeo3, temporal_uncertain_unwrapping=0.0)),
    ("individual, global correction", (; individual=true, correctglobal=true)),
]

open(ARGS[1], "w") do io
    for (name, kw) in CASES
        uw = unwrap(phase; TEs=TES, mag, mask, kw...)
        for e in axes(phase, 4)
            turns = round.(Int, (uw[:, :, :, e] .- phase[:, :, :, e]) ./ 2π)
            write(io, Int8.(turns[mask]))
        end
        println(name, ": ", count(mask), " mask voxels")
    end
    corrected, _ = mcpc3ds(phase, mag; TEs=TES, sigma=[7, 7, 7])
    for e in axes(corrected, 4)
        write(io, Float64.(corrected[:, :, :, e][mask]))
    end
    println("mcpc3ds: ", count(isfinite, corrected[mask, :]), " finite of ", 3count(mask))
end
