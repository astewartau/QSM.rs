# Generates tests/data/tgv_tgvjl_small.bin: QuantitativeSusceptibilityMappingTGV.jl's `qsm_tgv`
# (korbinian90, 6c0886a, v0.5.2) on small polynomial phantoms, for tests/tgv_tgvjl_parity.rs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -e JULIA_NUM_THREADS=4 -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/mritools.so --startup-file=no /data/tgv_tgvjl_small.jl /data/tgv_tgvjl_small.bin
#
# .github/workflows/regenerate-fixtures.yml runs this command (with docker) in Actions.
#
# The phantom is built from + - * / only, so the Rust test rebuilds it bit for bit. The mask
# touches the array's first and last z planes, where the package's erosion does not erode. Layout
# of the output (little-endian), per case in CASES order: the number of voxels in the package's
# eroded mask (`erosions` = 3 erosions, u32); that mask as bits over the volume (column-major,
# least significant bit first, ceil(n/8) bytes); the package's chi there (f64 of its Float32, ppm).
using QuantitativeSusceptibilityMappingTGV
const T = QuantitativeSusceptibilityMappingTGV

const B0, TE = 3.0, 0.02
const OBLIQUE = [0.1, -0.2, 0.9746794344808963]   # (0.1, -0.2, sqrt(1 - 0.05))

# Ellipsoid mask (semi-axes 0.46, 0.44 of the extent in x, y; 0.6 in z, so it is cut by the first
# and last z planes) holding a smooth field (ppm) and two spheres, zero outside the mask.
function phantom(n, v)
    f = zeros(Float64, n)
    m = falses(n)
    a, b, c = 0.46 * n[1] * v[1], 0.44 * n[2] * v[2], 0.6 * n[3] * v[3]
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        x = ((i - 1) - (n[1] - 1) / 2 + 0.3) * v[1]
        y = ((j - 1) - (n[2] - 1) / 2 + 0.3) * v[2]
        z = ((k - 1) - (n[3] - 1) / 2 + 0.3) * v[3]
        if x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0
            m[i, j, k] = true
            val = 0.002 * z + 0.0003 * (x * x - y * y) + 1.0e-5 * x * y * z
            if (x - 0.2 * a)^2 + y * y + z * z < (0.25 * a)^2
                val += 0.005
            end
            if (x + 0.3 * a)^2 + (y - 0.2 * b)^2 + (z + 0.1 * c)^2 < (0.2 * a)^2
                val -= 0.003
            end
            f[i, j, k] = val
        end
    end
    return f, m
end

# (name, size, voxel size, B0 direction, iterations, laplacian, phase scale)
# - "del": the package's default Laplacian on the phase (QSM.rs TgvInput::WrappedPhase), the
#   phase being the field times `scale` rad/ppm.
# - "field": QSM.rs TgvInput::FieldPpm, the field as phase at TE: 2π·42.5781·B0·TE rad/ppm (the
#   package's constant), through `get_laplace_phase_romeo`. The phase stays well below π, so its
#   ROMEO unwrap changes nothing and this is the discrete Laplacian QSM.rs uses for a field.
const CASES = [
    ("del, package defaults", (32, 28, 24), (1.0, 1.0, 1.0), [0.0, 0.0, 1.0], nothing, get_laplace_phase_del, 40.0),
    ("field, anisotropic, oblique B0, 300 iterations", (30, 26, 22), (0.9, 1.0, 1.3), OBLIQUE, 300, get_laplace_phase_romeo, nothing),
    ("field, 300 iterations", (30, 26, 22), (1.0, 1.0, 1.0), [0.0, 0.0, 1.0], 300, get_laplace_phase_romeo, nothing),
    ("del, anisotropic, 300 iterations", (30, 26, 22), (0.9, 1.0, 1.3), [0.0, 0.0, 1.0], 300, get_laplace_phase_del, 40.0),
    ("del, oblique B0, 300 iterations", (30, 26, 22), (1.0, 1.0, 1.0), OBLIQUE, 300, get_laplace_phase_del, 40.0),
]

open(ARGS[1], "w") do io
    for (name, n, v, bdir, iters, lap, scale) in CASES
        f, m = phantom(n, v)
        rpp = 2pi * 42.5781 * B0 * TE
        phase = scale === nothing ? f .* rpp : f .* scale
        kw = iters === nothing ? (;) : (; iterations=iters)
        chi = qsm_tgv(phase, m, collect(v); TE=TE, fieldstrength=B0, B0_dir=bdir, laplacian=lap, kw...)
        if scale !== nothing
            chi .*= Float32(rpp / scale)   # chi as if the phase had been f·rpp (the package's scale)
        end
        em = m
        for _ in 1:3
            em = T.erode_mask(em)
        end
        write(io, UInt32(count(em)))
        bits = zeros(UInt8, cld(length(em), 8))
        for (l, b) in enumerate(vec(em))
            b && (bits[(l - 1) >> 3 + 1] |= UInt8(1) << ((l - 1) & 7))
        end
        write(io, bits)
        write(io, Float64.(chi[em]))
        println(name, ": ", count(m), " mask voxels, ", count(em), " eroded, max |phase| ", maximum(abs.(phase)))
    end
end
