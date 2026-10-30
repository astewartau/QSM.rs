# Generates tests/data/clearswi_small.bin: CLEARSWI.jl (korbinian90, 0949bc5 = 1.7.0) on a small
# multi-echo phantom, for tests/clearswi_parity.rs: the SWI magnitude (`getswimag`), the phase mask
# (`getswiphase`) and the combined high-pass-filtered phase they come from, in three cases:
#   1. a phase without wraps, the default Options() (Laplacian unwrapping, RSS magnitude, CLEAR
#      sensitivity, softplus, tanh mask of strength 4, high-pass sigma [4, 4, 0]);
#   2. the same with mag_sens = [1] (no sensitivity correction);
#   3. the phase with wraps (its frequency map times 6), the default Options();
# and then, as case 4, CLEARSWI.jl's own steps after unwrapping (`laplacian_combine` without its
# `laplacianunwrap`) on case 1's true unwrapped phase: each echo less its masked
# `gaussiansmooth3d` (sigma [4, 4, 0], dims 1:3), `combine_phase` (calculateB0_unwrapped, Hz),
# and `createphasemask!` (tanh, strength 4), within case 1's mask.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -e JULIA_NUM_THREADS=4 -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/mritools.so --startup-file=no /data/clearswi_small.jl /data/clearswi_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# machine-dependent: Julia's tanh rounds differently between CPUs, so a GitHub-hosted runner gives
# 7 phase-mask values (cases 1-3) 1 ulp (1.1e-16) from the committed .bin, made on an AMD EPYC.
# Everything else is identical. Far inside tests/clearswi_parity.rs's tolerances.
#
# The phantom is built from + - * /, round and integer xor only, so the Rust test rebuilds it bit
# for bit. Layout (little-endian, column-major over the 20x20x14 grid, voxel 1x1x1.5 mm), per case
# 1-3: the phase mask's mask (robustmask of the first echo's magnitude) as bits (least significant
# bit first, ceil(n/8) bytes); the number of mask voxels m (u32); the SWI magnitude over the grid
# (f64); the phase mask over the grid (f64); the combined filtered phase at the mask voxels (f64,
# Hz). Then case 4: the combined filtered phase at case 1's mask voxels (f64, Hz) and the phase
# mask over the grid (f64).
using CLEARSWI, MriResearchTools

const N = (20, 20, 14)
const VS = (1.0, 1.0, 1.5)
const TES = [4.0, 8.0, 12.0]       # ms

function phantom(scale)
    ne = length(TES)
    wrapped = zeros(Float64, N..., ne)
    unwrapped = zeros(Float64, N..., ne)
    rawmax = 0.0
    mag = zeros(Float64, N..., ne)
    a, b, c = 0.46 * N[1], 0.44 * N[2], 0.42 * N[3]
    for e in 1:ne, k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x = (i - 1) - (N[1] - 1) / 2 + 0.3
        y = (j - 1) - (N[2] - 1) / 2 + 0.3
        z = (k - 1) - (N[3] - 1) / 2 + 0.3
        inside = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0
        # a smooth background and three small dipole-like sources (veins, iron): rad/ms
        bg = 0.004 * x + 0.003 * y - 0.002 * z + 0.0002 * x * y
        s1 = (x - 3) * (x - 3) + (y + 2) * (y + 2)
        s2 = (x + 4) * (x + 4) + (y - 3) * (y - 3) + (z - 1) * (z - 1)
        s3 = (y + 5) * (y + 5) + (z + 2) * (z + 2)
        src = 0.06 / (1.0 + s1 / 2.0) - 0.05 / (1.0 + s2 / 3.0) + 0.04 / (1.0 + s3 / 1.5)
        omega = scale * (bg + (inside ? src : 0.0))
        phi0 = 0.1 + 0.01 * x - 0.02 * y
        h = ((i - 1) * 73856093) ⊻ ((j - 1) * 19349663) ⊻ ((k - 1) * 83492791) ⊻ ((e - 1) * 2654435761)
        noise = ((h % 1000) / 1000 - 0.5) * (inside ? 0.04 : 0.6)
        raw = TES[e] * omega + phi0 + noise
        wrapped[i, j, k, e] = raw - 2π * round(raw / 2π)
        unwrapped[i, j, k, e] = raw
        rawmax = max(rawmax, abs(raw))
        bias = 1.0 + 0.02 * x - 0.015 * y + 0.0005 * x * z
        vein = inside ? 30.0 / (1.0 + s1 / 2.0) : 0.0
        tissue = inside ? (100.0 - 0.1 * (x * x + y * y + z * z) - vein) : 4.0 + (h % 7)
        mag[i, j, k, e] = tissue * bias / (1.0 + 0.1 * (e - 1))
    end
    return wrapped, mag, rawmax, unwrapped
end

bits(v) = begin
    v = vec(v)
    out = zeros(UInt8, cld(length(v), 8))
    for i in eachindex(v)
        v[i] && (out[(i - 1) ÷ 8 + 1] |= UInt8(1) << ((i - 1) % 8))
    end
    out
end

header = (pixdim = Float64[1, VS..., 1, 1, 1, 1],)
cases = [(1.0, Options()), (1.0, Options(mag_sens = [1])), (6.0, Options())]
open(ARGS[1], "w") do io
    for (scale, opts) in cases
        phase, mag, rawmax, _ = phantom(scale)
        @assert (scale == 1.0) == (rawmax < π)   # cases 1 and 2 have no wraps, case 3 does
        data = CLEARSWI.Data(mag, phase, header, TES)
        mask = robustmask(view(data.mag, :, :, :, 1))
        swimag = CLEARSWI.getswimag(data, opts)
        combined = CLEARSWI.getcombinedphase(data, opts, mask)
        swiphase = CLEARSWI.createphasemask!(copy(combined), mask, opts.phase_scaling_type, opts.phase_scaling_strength)
        @assert swiphase == CLEARSWI.getswiphase(data, opts)
        write(io, bits(mask))
        write(io, UInt32(count(mask)))
        write(io, Float64.(vec(swimag)))
        write(io, Float64.(vec(swiphase)))
        write(io, Float64.(combined[mask]))
        println("scale $scale (max |phase| before wrapping $(round(rawmax; digits=2))), mag_sens $(opts.mag_sens): $(count(mask)) mask voxels, phase mask in ",
                extrema(swiphase[mask]), ", magnitude in ", extrema(swimag))
    end
    # Case 4: CLEARSWI.jl's laplacian_combine after its unwrapping, on the true unwrapped phase.
    phase, mag, rawmax, unwrapped = phantom(1.0)
    opts = Options()
    mask = robustmask(view(mag, :, :, :, 1))
    filtered = copy(unwrapped)
    for iEco in axes(filtered, 4)
        filtered[:, :, :, iEco] .-= gaussiansmooth3d(unwrapped[:, :, :, iEco], opts.phase_hp_sigma; mask, dims=1:3)
    end
    combined = CLEARSWI.combine_phase(filtered, Float64.(mag), CLEARSWI.to_dim(TES, 4))
    swiphase = CLEARSWI.createphasemask!(copy(combined), mask, opts.phase_scaling_type, opts.phase_scaling_strength)
    write(io, Float64.(combined[mask]))
    write(io, Float64.(vec(swiphase)))
    println("case 4 (after unwrapping): phase mask in ", extrema(swiphase[mask]))
end
