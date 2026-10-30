# Generates tests/data/laplacian_qsmjl_small.bin: QSM.jl's `unwrap_laplacian` with its `:dct`
# solver (kamesy/QSM.jl fde4b92) on small wrapped phantoms, for tests/laplacian_qsmjl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/qsmjl.so --startup-file=no /data/laplacian_qsmjl_small.jl /data/laplacian_qsmjl_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# QSM.rs ports `:dct` only (QSM.jl's default `:mgpcg` also removes the background; `:fft` is not
# STI's `MRPhaseUnwrap`, which QSM.rs's FFT solver is). The wrapped phase is written to the
# fixture, as wrapping goes through `rem2pi`. Layout (little-endian, column-major), per case in
# CASES order: the wrapped phase over the grid (f64, rad), then QSM.jl's unwrapped phase there
# (f64, rad). The mask is not used by `:dct`.
using QSM

const CASES = [
    # (name, size, voxel size)
    ("1 mm", (20, 18, 16), (1.0, 1.0, 1.0)),
    ("anisotropic", (18, 20, 14), (0.9, 1.0, 1.3)),
]

# A smooth phase spanning several wraps, plus a sharp sphere.
function phantom(n, v)
    p = zeros(Float64, n)
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        x = ((i - 1) - (n[1] - 1) / 2 + 0.3) * v[1]
        y = ((j - 1) - (n[2] - 1) / 2 + 0.3) * v[2]
        z = ((k - 1) - (n[3] - 1) / 2 + 0.3) * v[3]
        val = 0.6 * z + 0.06 * (x * x - y * y) + 0.004 * x * y * z
        if x * x + y * y + z * z < 9.0
            val += 2.0
        end
        p[i, j, k] = rem2pi(val, RoundNearest)
    end
    return p
end

open(ARGS[1], "w") do io
    for (name, n, v) in CASES
        p = phantom(n, v)
        u = unwrap_laplacian(p, trues(n), v; solver=:dct)
        write(io, p)
        write(io, Float64.(u))
        println(name, ": unwrapped range ", extrema(u))
    end
end
