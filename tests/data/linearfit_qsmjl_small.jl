# Generates tests/data/linearfit_qsmjl_small.bin: QSM.jl's `multi_echo_linear_fit`
# (kamesy/QSM.jl fde4b92) on a small four-echo phantom, with and without the phase offset, for
# tests/linearfit_qsmjl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/qsmjl.so --startup-file=no /data/linearfit_qsmjl_small.jl /data/linearfit_qsmjl_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# The phantom (unwrapped phases, magnitudes, mask) is built from + - * / only, so the Rust test
# rebuilds it bit for bit. QSM.jl's `W` is the square root of the weights, and QSM.rs weights each
# echo by its magnitude, so `W` is the square root of the magnitude. Layout (little-endian, column-major), per case (offset, then no
# offset): the number of mask voxels (u32), then at the mask voxels QSM.jl's slope (f64, rad/s)
# and, with the offset, the offset (f64, rad).
using QSM

const N = (14, 12, 10)
const TES = [0.004, 0.009, 0.014, 0.019]

function phantom()
    phas = zeros(Float64, N..., length(TES))
    mag = zeros(Float64, N..., length(TES))
    m = falses(N)
    for k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x, y, z = i - 7.3, j - 6.1, k - 5.4
        m[i, j, k] = x * x / 36.0 + y * y / 30.0 + z * z / 20.0 < 1.0
        f = 300.0 + 40.0 * x - 25.0 * y + 3.0 * x * z          # rad/s
        o = 0.4 - 0.05 * y + 0.01 * x * x                       # rad
        r2 = 25.0 + 2.0 * x + 1.5 * z
        for (e, t) in enumerate(TES)
            # a deterministic, echo- and voxel-dependent deviation from the line
            d = 0.02 * ((i * 7 + j * 3 + k * 5 + e * 11) % 13 - 6) / 6
            phas[i, j, k, e] = o + f * t + d
            mag[i, j, k, e] = (1.0 + 0.1 * z / 5.0) / (1.0 + r2 * t)
        end
    end
    return phas, mag, m
end

phas, mag, m = phantom()
open(ARGS[1], "w") do io
    p, p0 = multi_echo_linear_fit(phas, TES; W=sqrt.(mag), mask=m, phase_offset=true)
    write(io, UInt32(count(m))); write(io, Float64.(p[m])); write(io, Float64.(p0[m]))
    q = multi_echo_linear_fit(phas, TES; W=sqrt.(mag), mask=m, phase_offset=false)
    write(io, UInt32(count(m))); write(io, Float64.(q[m]))
    println(count(m), " mask voxels; slope ", extrema(p[m]), " offset ", extrema(p0[m]))
end
