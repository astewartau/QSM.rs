# Generates tests/data/romeo_validity_romeojl_small.bin: MriResearchTools.jl's `mcpc3ds` (v3.8.0)
# followed by ROMEO.jl's multi-echo `unwrap` (v1.6.0) and `calculateB0_unwrapped`, on a small
# three-echo phantom whose brain mask reaches well past MCPC-3D-S's own `robustmask`, so the
# smoothed phase offset (and the corrected phase) is NaN on part of the brain mask; plus ROMEO.jl's
# `unwrap` on the raw phase (no offset removal). For tests/romeo_validity_romeojl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299), with one thread so ROMEO.jl's
# threaded individual unwrapping is deterministic:
#
#   podman run --rm -e JULIA_NUM_THREADS=1 -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/mritools.so --startup-file=no /data/romeo_validity_romeojl_small.jl /data/romeo_validity_romeojl_small.bin
#
# The phantom is built from + - * /, round and integer xor only, so the Rust test rebuilds it bit
# for bit. Layout of the output (little-endian), "brain voxels" = the brain mask in column-major
# order:
# - mcpc3ds (sigma 2 voxels): per brain voxel, 1 if the corrected phase is finite (u8); then per
#   echo, the corrected phase at the finite brain voxels (f64);
# - per case in CASES order: per echo, the turns ROMEO.jl added, round((unwrapped - input) / 2π),
#   at the finite brain voxels (i8); then calculateB0_unwrapped (:phase_snr, Hz) at the finite
#   brain voxels (f64), for the cases on the corrected phase.
using ROMEO, MriResearchTools

const N = (30, 26, 16)
const TES = [4.0, 8.0, 12.0]   # ms

function phantom()
    ne = length(TES)
    phase = zeros(Float64, N..., ne)
    mag = zeros(Float64, N..., ne)
    brain = falses(N)
    a, b, c = 0.30 * N[1], 0.40 * N[2], 0.40 * N[3]
    for e in 1:ne, k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x = (i - 1) - 0.40 * (N[1] - 1) + 0.3
        y = (j - 1) - (N[2] - 1) / 2 + 0.3
        z = (k - 1) - (N[3] - 1) / 2 + 0.3
        head = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0
        # the brain mask: the head plus a slab beyond it in +x, where there is no signal
        slab = x > 0 && x < 1.75 * a && y * y / (b * b) + z * z / (c * c) < 0.5
        brain[i, j, k] = head || slab
        omega = 0.08 * x + 0.05 * y - 0.03 * z + 0.004 * x * y + 0.002 * (x * x - z * z)   # rad/ms
        # a smooth coil-combination offset in the head, a steep one outside it
        phi0 = head ? 0.5 + 0.02 * x - 0.03 * y : 0.5 + 0.9 * (x - a) * (y + 0.5) - 0.4 * z
        h = ((i - 1) * 73856093) ⊻ ((j - 1) * 19349663) ⊻ ((k - 1) * 83492791) ⊻ ((e - 1) * 2654435761)
        noise = ((h % 1000) / 1000 - 0.5) * (head ? 0.4 : 1.5)
        raw = TES[e] * omega + phi0 + noise
        phase[i, j, k, e] = raw - 2π * round(raw / 2π)
        mag[i, j, k, e] = head ? (100.0 - 0.1 * (x * x + y * y + z * z)) / (1.0 + 0.1 * (e - 1)) : 0.5 + (h % 7) / 10
    end
    return phase, mag, brain
end

phase, mag, brain = phantom()
# Template mode without the temporal re-unwrapping of uncertain voxels: QSM.rs's port of that
# fallback (`temporal_uncertain_unwrapping!`) is not yet ROMEO.jl's (its voxel quality and seeding
# differ), so it is left out here; see the ROMEO module docs in QSM.rs.
const CASES = [
    ("MCPC-3D-S, individual, global correction", true, (; individual=true, correctglobal=true)),
    ("MCPC-3D-S, individual", true, (; individual=true)),
    ("MCPC-3D-S, template", true, (; temporal_uncertain_unwrapping=0.0)),
    ("MCPC-3D-S, template, global correction", true, (; correctglobal=true, temporal_uncertain_unwrapping=0.0)),
    ("raw phase, individual, global correction", false, (; individual=true, correctglobal=true)),
    ("raw phase, template", false, (; temporal_uncertain_unwrapping=0.0)),
]

open(ARGS[1], "w") do io
    corrected, _ = mcpc3ds(phase, mag; TEs=TES, sigma=[2, 2, 2])
    fin = isfinite.(corrected[:, :, :, 1])
    @assert all(isfinite.(corrected[:, :, :, 2]) .== fin) && all(isfinite.(corrected[:, :, :, 3]) .== fin)
    write(io, UInt8.(fin[brain]))
    keep = brain .& fin
    for e in axes(corrected, 4)
        write(io, Float64.(corrected[:, :, :, e][keep]))
    end
    println("mcpc3ds: ", count(keep), " of ", count(brain), " brain voxels finite")
    for (name, offsetremoval, kw) in CASES
        input = offsetremoval ? corrected : phase
        uw = unwrap(input; TEs=TES, mag, mask=brain, kw...)
        sel = offsetremoval ? keep : brain
        for e in axes(input, 4)
            turns = round.(Int, (uw[:, :, :, e][sel] .- input[:, :, :, e][sel]) ./ 2π)
            write(io, Int8.(turns))
        end
        if offsetremoval
            B0 = calculateB0_unwrapped(uw, mag, TES)
            write(io, Float64.(B0[sel]))
            println(name, ": B0 finite outside? ", count(!iszero, B0[brain .& .!fin]))
        else
            println(name)
        end
    end
end
