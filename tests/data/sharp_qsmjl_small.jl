# Generates tests/data/sharp_qsmjl_small.bin: QSM.jl's `sharp` and `vsharp` (kamesy/QSM.jl fde4b92,
# v0.5.4) on small polynomial phantoms, for tests/sharp_qsmjl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/qsmjl.so --startup-file=no /data/sharp_qsmjl_small.jl /data/sharp_qsmjl_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# The phantom is built from + - * / only, so the Rust test rebuilds it bit for bit. Layout of the
# output (little-endian), per case in CASES order: the number of voxels in QSM.jl's eroded mask
# (u32); that mask as bits over the whole volume (column-major, least significant bit first,
# ceil(n/8) bytes); QSM.jl's local field at those voxels (f64, column-major order).
using QSM

const CASES = [
    # (name, dims, voxel size, method, radii (mm; `nothing` = QSM.jl's default), thr)
    ("sharp, QSM.jl defaults", (56, 52, 48), (1.0, 1.0, 1.0), :sharp, nothing, nothing),
    ("sharp r 4 mm, anisotropic", (32, 28, 24), (0.9, 1.0, 1.3), :sharp, [4.0], 0.05),
    ("vsharp, QSM.jl defaults", (48, 44, 40), (1.0, 1.0, 1.0), :vsharp, nothing, nothing),
    ("vsharp 6:-1.5:1.5 mm, thr 0.1, anisotropic", (32, 28, 24), (0.9, 1.0, 1.3), :vsharp, [6.0, 4.5, 3.0, 1.5], 0.1),
]

# Ellipsoid mask (semi-axes 0.46, 0.44, 0.42 of the extent) holding a harmonic background and two
# spheres of local field, zero outside the mask.
function phantom(n, v)
    f = zeros(Float64, n)
    m = falses(n)
    a, b, c = 0.46 * n[1] * v[1], 0.44 * n[2] * v[2], 0.42 * n[3] * v[3]
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        x = ((i - 1) - (n[1] - 1) / 2 + 0.3) * v[1]
        y = ((j - 1) - (n[2] - 1) / 2 + 0.3) * v[2]
        z = ((k - 1) - (n[3] - 1) / 2 + 0.3) * v[3]
        if x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0
            m[i, j, k] = true
            val = 0.02 * z + 0.003 * (x * x - y * y) + 1.0e-4 * x * y * z
            if (x - 0.2 * a)^2 + y * y + z * z < (0.25 * a)^2
                val += 0.05
            end
            if (x + 0.3 * a)^2 + (y - 0.2 * b)^2 + (z + 0.1 * c)^2 < (0.2 * a)^2
                val -= 0.03
            end
            f[i, j, k] = val
        end
    end
    return f, m
end

open(ARGS[1], "w") do io
    for (name, n, v, method, r, thr) in CASES
        f, m = phantom(n, v)
        kw = Dict{Symbol, Any}()
        r === nothing || (kw[:r] = method == :sharp ? r[1] : r)
        thr === nothing || (kw[:thr] = thr)
        fl, sm = method == :sharp ? sharp(f, m, v; kw...) : vsharp(f, m, v; kw...)
        write(io, UInt32(count(sm)))
        bits = zeros(UInt8, cld(length(sm), 8))
        for (l, b) in enumerate(vec(sm))
            b && (bits[(l - 1) >> 3 + 1] |= UInt8(1) << ((l - 1) & 7))
        end
        write(io, bits)
        write(io, Float64.(fl[sm]))
        println(name, ": ", count(m), " mask voxels, ", count(sm), " eroded")
    end
end
