# Generates tests/data/laplacian_mgpcg_qsmjl_small.bin: QSM.jl's `unwrap_laplacian` with its
# default `:mgpcg` solver (kamesy/QSM.jl fde4b92) on small wrapped phantoms, for
# tests/laplacian_qsmjl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest):
#
#   podman run --rm -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/qsmjl.so --startup-file=no /data/laplacian_mgpcg_qsmjl_small.jl /data/laplacian_mgpcg_qsmjl_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# The cases cover the solver's paths: a mask whose bounding box is not cropped (1 mm), one that
# is (the array is more than 10 voxels larger along an axis; anisotropic voxels), and a mask that
# fills the array (mask voxels on the array edge). The phase is wrapped and left raw outside the
# mask, as QSM.jl's README passes it. Layout (little-endian, column-major), per case in CASES
# order: the mask (u8), the wrapped phase (f64, rad), QSM.jl's unwrapped phase (f64, rad).
using QSM

const CASES = [
    # (name, size, voxel size, mask semi-axes in mm; nothing = the whole array)
    ("1 mm, no crop", (20, 18, 16), (1.0, 1.0, 1.0), (7.5, 6.5, 5.5)),
    ("anisotropic, cropped", (40, 34, 26), (0.9, 1.0, 1.3), (9.0, 8.0, 7.0)),
    ("whole array", (18, 20, 14), (1.0, 1.0, 1.0), nothing),
]

# A smooth phase spanning several wraps, plus a sharp sphere. The first three terms are harmonic
# (`:mgpcg` removes them with the background); the last two and the sphere are not.
function phantom(n, v)
    p = zeros(Float64, n)
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        x = ((i - 1) - (n[1] - 1) / 2 + 0.3) * v[1]
        y = ((j - 1) - (n[2] - 1) / 2 + 0.3) * v[2]
        z = ((k - 1) - (n[3] - 1) / 2 + 0.3) * v[3]
        val = 0.6 * z + 0.06 * (x * x - y * y) + 0.004 * x * y * z
        val += 0.02 * (x * x + y * y + z * z) + 0.3 * sin(0.4 * x)
        if x * x + y * y + z * z < 9.0
            val += 2.0
        end
        p[i, j, k] = rem2pi(val, RoundNearest)
    end
    return p
end

function ellipsoid(n, v, r)
    r === nothing && return trues(n)
    m = falses(n)
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        x = ((i - 1) - (n[1] - 1) / 2) * v[1] / r[1]
        y = ((j - 1) - (n[2] - 1) / 2) * v[2] / r[2]
        z = ((k - 1) - (n[3] - 1) / 2) * v[3] / r[3]
        m[i, j, k] = x * x + y * y + z * z <= 1.0
    end
    return m
end

open(ARGS[1], "w") do io
    for (name, n, v, r) in CASES
        p = phantom(n, v)
        m = ellipsoid(n, v, r)
        u = unwrap_laplacian(p, m, v; solver=:mgpcg)
        write(io, UInt8.(m))
        write(io, p)
        write(io, Float64.(u))
        println(name, ": ", sum(m), " mask voxels, unwrapped range ", extrema(u))
    end
end
