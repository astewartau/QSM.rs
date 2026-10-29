# Generates tests/data/inversion_qsmjl_small.bin: QSM.jl's FFT dipole inversions `tkd`, `tsvd`,
# `tikh`, `tv` and `rts` (kamesy/QSM.jl fde4b92, v0.5.4) on small polynomial phantoms, for
# tests/inversion_qsmjl_parity.rs.
#
# Not `nltv`: it allocates its data-term multiplier `ue` with `similar` and never initialises
# it, so on arrays this small (which reuse freed memory) its output is NaN and differs from run
# to run. On the in-vivo scan (fresh, zeroed allocations) QSM.rs matches it to rounding; see
# `inversion::nltv`'s docs.
#
# Run in the verified QSM-CI Julia image (QSMxT/QSM-CI#299: Julia 1.10.11, the committed Manifest,
# output bit-identical across runs and against a host Julia):
#
#   podman run --rm -v "$PWD/tests/data":/data \
#     ghcr.io/astewartau/qsm-ci/julia-qsm@sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185 \
#     julia --sysimage /opt/qsmci-julia/qsmjl.so --startup-file=no /data/inversion_qsmjl_small.jl /data/inversion_qsmjl_small.bin
#
# The phantom is built from + - * / only, so the Rust test rebuilds it bit for bit. Layout of the
# output (little-endian), per case in CASES order: the number of mask voxels (u32) and QSM.jl's
# chi at the mask voxels (f64, ppm, column-major order), on that case's phantom.
using QSM

const N = (24, 22, 20)              # 22 is not a fast FFT size: QSM.jl's default pads it to 24
# Odd sizes, unpadded, for the first oblique cases. The cases after them are oblique on even
# sizes, where QSM.jl's real-to-complex FFT resolves the Nyquist planes of the tilted dipole
# kernel differently from a plain complex FFT; QSM.rs reproduces that with the kernel `irfft`
# implies (astewartau/QSM.rs#329). They are appended so the earlier cases' bytes do not change.
const N_ODD = (23, 21, 19)
const ANISO = (0.9, 1.0, 1.3)
# The iterative methods (tv, rts) run on 1 mm voxels: on anisotropic voxels the FFT of
# QSM.jl's Laplacian kernel is not exactly 0 at DC (4.4e-16 at ANISO), and QSM.jl inverts that
# rounding, which puts an arbitrary constant into chi. QSM.rs zeroes it.
const ISO = (1.0, 1.0, 1.0)
const OBLIQUE = (0.1, -0.2, 0.9746794344808963)   # (0.1, -0.2, sqrt(1 - 0.05))

# Ellipsoid mask (semi-axes 0.46, 0.44, 0.42 of the extent) holding a smooth field and two
# spheres, zero outside the mask.
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

nopad = (-1, -1, -1)
# (name, size, voxel size, method on (f, m, vsz))
const CASES = [
    ("tkd, QSM.jl defaults", N, ANISO, (f, m, v) -> tkd(f, m, v)),
    ("tkd, oblique B0, odd sizes, no padding", N_ODD, ANISO, (f, m, v) -> tkd(f, m, v; bdir=OBLIQUE, pad=nopad)),
    ("tsvd, QSM.jl defaults", N, ANISO, (f, m, v) -> tsvd(f, m, v)),
    ("tikh, QSM.jl defaults", N, ANISO, (f, m, v) -> tikh(f, m, v)),
    ("tikh, gradient, lambda 1e-3, pad 4", N, ANISO, (f, m, v) -> tikh(f, m, v; reg=:gradient, lambda=1e-3, pad=(4, 4, 4))),
    ("tv, QSM.jl defaults", N, ISO, (f, m, v) -> tv(f, m, v)),
    ("rts, QSM.jl defaults", N, ISO, (f, m, v) -> rts(f, m, v)),
    ("rts, oblique B0, odd sizes, no padding", N_ODD, ISO, (f, m, v) -> rts(f, m, v; bdir=OBLIQUE, pad=nopad)),
    ("tkd, oblique B0, even sizes, QSM.jl defaults", N, ANISO, (f, m, v) -> tkd(f, m, v; bdir=OBLIQUE)),
    ("tsvd, oblique B0, even sizes, QSM.jl defaults", N, ANISO, (f, m, v) -> tsvd(f, m, v; bdir=OBLIQUE)),
    ("tikh, oblique B0, even sizes, QSM.jl defaults", N, ANISO, (f, m, v) -> tikh(f, m, v; bdir=OBLIQUE)),
    ("tikh, gradient, oblique B0, even sizes, lambda 1e-3, pad 4", N, ANISO, (f, m, v) -> tikh(f, m, v; bdir=OBLIQUE, reg=:gradient, lambda=1e-3, pad=(4, 4, 4))),
    ("tv, oblique B0, even sizes, QSM.jl defaults", N, ISO, (f, m, v) -> tv(f, m, v; bdir=OBLIQUE)),
    ("rts, oblique B0, even sizes, QSM.jl defaults", N, ISO, (f, m, v) -> rts(f, m, v; bdir=OBLIQUE)),
]

open(ARGS[1], "w") do io
    for (name, n, v, run) in CASES
        f, m = phantom(n, v)
        chi = run(f, m, v)
        write(io, UInt32(count(m)))
        write(io, Float64.(chi[m]))
        println(name, ": ", count(m), " mask voxels, ", count(!isfinite, chi), " non-finite")
    end
end
