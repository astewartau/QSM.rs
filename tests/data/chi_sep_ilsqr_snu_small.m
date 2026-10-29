function chi_sep_ilsqr_snu_small(out_file)
% Writes tests/data/chi_sep_ilsqr_snu_small.bin: the SNU chi-separation toolbox's chi_sep_iLSQR on
% three small phantoms, for tests/chi_sep_ilsqr_snu_parity.rs.
%
% Reference: chi-separation toolbox v1.1.3 (SNU-LIST, obtained through its Google Form; files dated
% 2025-07-02), functions/corefunc/chi_sep_iLSQR.p and its three step functions (sha256 below), run
% in licensed MATLAB R2023b Update 5 (23.2.0.2459199) on a Bunya Slurm node. The toolbox must be
% first on the path (its utils/ shadow other toolboxes), with no other chi-separation code on it:
%   addpath(genpath('<toolbox>/functions')); addpath(genpath('<toolbox>/utils'));
%   chi_sep_ilsqr_snu_small('chi_sep_ilsqr_snu_small.bin')
%   sha256 chi_sep_iLSQR.p            65dd2580f66c4bbdec3994d1d112c5f20d97854d411d5ae68f4cde7923de3cf4
%   sha256 chi_sep_iLSQR_step1_func.p 7eb811e5ec80b3b352622984db59a1c33d96e082ad7c723db36a700ea0881687
%   sha256 chi_sep_iLSQR_step2_func.p 4bab8e19239ee218f1f59ab5d8a7f236d14a2279f1ff1cfcccaafc65b57c07e4
%   sha256 chi_sep_iLSQR_step3_func.p 569e02f8c49c28be4b35742f65052ec81284a5338f3d3dacd783fda8a9d30a77
% The call is Chisep_script.m's: params b0_dir, CF, voxel_size (plus Dr_pos/Dr_neg in case 3) and
% option_data.qsm, option_data.N_std. The script also checks that the result does not depend on the
% magnitude or N_std (chi_sep_iLSQR takes both but never uses them).
%
% The phantom is the one of chi_sep_medi_snu_small.m, built only from operations Rust reproduces
% exactly (integer LCG noise, + - * /), except the field, which goes through one FFT convolution
% (rebuilt with QSM.rs's FFT, equal to rounding). Layout (little-endian), per case in order: the
% number of mask voxels m (u32), the NNLS iteration count (u32), then chi+ (x_para) and |chi-|
% (x_dia) at the mask voxels in column-major order (2m f64, ppm).
fid = fopen(out_file, 'w');
for c = 1:3
    [S, params, od] = mkcase(c);
    log = evalc('[xp, xn] = chi_sep_iLSQR(S.mag, S.f_hz, S.r2p, S.mask, params, od);');
    iters = numel(regexp(log, 'NNLS iter\. #(\d+)', 'tokens'));
    m = S.mask(:) > 0;
    fprintf('case %d: %d mask voxels, %d NNLS iterations, chi+ [%g %g], |chi-| [%g %g]\n', c, nnz(m), iters, ...
        min(xp(:)), max(xp(:)), min(xn(:)), max(xn(:)));
    if c == 1
        % the magnitude and N_std are not used
        od2 = od; od2.N_std = od.N_std .* (1 + 5 * S.mag / max(S.mag(:)));
        evalc('[xp2, xn2] = chi_sep_iLSQR(S.mag .^ 2 + 7, S.f_hz, S.r2p, S.mask, params, od2);');
        assert(isequal(xp, xp2) && isequal(xn, xn2), 'chi_sep_iLSQR depends on mag or N_std');
        fprintf('case 1: identical with another magnitude and N_std\n');
    end
    fwrite(fid, nnz(m), 'uint32'); fwrite(fid, iters, 'uint32');
    fwrite(fid, xp(m), 'double'); fwrite(fid, xn(m), 'double');
end
fclose(fid);
end

function [S, params, od] = mkcase(c)
switch c
    case 1  % 3 T, 1 mm, even grid
        n = [40 36 28]; vs = [1 1 1]; B0 = 3; h = [0 0 1];
        params = struct();
    case 2  % 7 T, odd grid, anisotropic voxels, oblique B0
        n = [33 29 21]; vs = [0.8 0.9 1.5]; B0 = 7; h = [0.1 -0.4 0.9] / norm([0.1 -0.4 0.9]);
        params = struct();
    case 3  % 3 T, negative R2' in places, unequal Dr
        n = [40 36 28]; vs = [1 1 1]; B0 = 3; h = [0 0 1];
        params = struct('Dr_pos', 120, 'Dr_neg', 150);
end
S = phantom(n, vs, B0, h, c == 3);
params.b0_dir = h; params.CF = 42.58e6 * B0; params.voxel_size = vs;
od.qsm = S.qsm; od.N_std = S.nstd;
end

function S = phantom(n, vs, B0, h, negative_r2p)
% Ellipsoid brain with chi+ and chi- blocks, a CSF ellipsoid, LCG noise.
N = prod(n);
u = @(seed) reshape(mod(1103515245 * ((0:N-1)' + seed * 7919) + 12345, 2^31) / 2^31, n);
[i, j, k] = ndgrid(1:n(1), 1:n(2), 1:n(3));
x = (i - (n(1) + 1) / 2) * vs(1); y = (j - (n(2) + 1) / 2) * vs(2); z = (k - (n(3) + 1) / 2) * vs(3);
R = 0.4 * n .* vs;
S.mask = (x / R(1)).^2 + (y / R(2)).^2 + (z / R(3)).^2 <= 1;
S.csf = S.mask & ((x / (0.25 * R(1))).^2 + ((y - 0.2 * R(2)) / (0.15 * R(2))).^2 + (z / (0.3 * R(3))).^2 <= 1);
blk = @(cx, cy, cz, r) abs(x - cx * R(1)) <= r * R(1) & abs(y - cy * R(2)) <= r * R(2) & abs(z - cz * R(3)) <= r * R(3);
cp = 0.08 * blk(0.4, -0.3, 0.1, 0.18) + 0.05 * blk(-0.35, -0.35, -0.2, 0.15) + 0.02 * (y < -0.5 * R(2));
cn = 0.04 * blk(-0.3, 0.45, 0.25, 0.2) + 0.03 * blk(0.35, 0.3, -0.3, 0.16) + 0.015 * (abs(x) < 0.3 * R(1));
cp = cp .* S.mask .* ~S.csf; cn = cn .* S.mask .* ~S.csf;
CF = 42.58e6 * B0;
D = kernel(n, vs, h);
S.f_hz = (real(ifftn(D .* fftn(cp - cn))) * (CF * 1e-6) + 0.4 * (u(1) - 0.5)) .* S.mask;
Dr = 137 / 2.9 * B0;
S.r2p = (Dr * (cp + cn) + 3 + 2 * (u(2) - 0.5)) .* S.mask;
if negative_r2p, S.r2p = S.r2p - 4 * S.mask .* (x > 0); end
S.qsm = (cp - cn + 0.004 * (u(3) - 0.5)) .* S.mask;
S.nstd = (0.01 + 0.004 * u(4)) .* S.mask;
S.mag = 200 + 60 * x / R(1) + 30 * u(5) + 150 * S.csf;
end

function D = kernel(n, vs, h)
% Unshifted dipole kernel on k = f/(n*vs), f = 0..n/2-1, -n/2..-1 (floor), D(0) = 0.
f = @(a) [0:ceil(n(a)/2)-1, -floor(n(a)/2):-1] / (n(a) * vs(a));
[kx, ky, kz] = ndgrid(f(1), f(2), f(3));
k2 = kx.^2 + ky.^2 + kz.^2;
D = 1/3 - (kx * h(1) + ky * h(2) + kz * h(3)).^2 ./ k2;
D(k2 == 0) = 0;
end
