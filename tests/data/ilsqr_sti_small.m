function ilsqr_sti_small(out_file)
% Writes tests/data/ilsqr_sti_small.bin: STI Suite 3.0's QSM_iLSQR and its double-precision
% re-implementation (ilsqr_sti_clone.m, next to this file) on one small synthetic case, for
% tests/ilsqr_sti_parity.rs.
%
% Recovered 2026-10-30 from the session that made the fixture (2026-10-09, Bunya; its two steps,
% merged here). It was run in licensed MATLAB R2023b with STI Suite 3.0's Core_Functions_P on the
% path:
%   addpath('<STISuite_V3.0>/Core_Functions_P'); ilsqr_sti_small('ilsqr_sti_small.bin')
% mksyn seeds MATLAB's generator (rng(31)), so the phase, including the noise added after it, is
% repeatable; the committed .bin holds the phase it was made from all the same. The QSM-CI ilsqr-sti container runs the
% same QSM_iLSQR (QSMxT/QSM-CI#304) but returns neither the initial LSQR solution nor runs the
% re-implementation, which is why this fixture is not made with it (ilsqr_sti_qsmci_small.py is).
%
% Layout (little-endian, column-major): phase (f32, rad, the whole grid), mask (u8), then inside
% the mask: STI's chi and chi_lsqr (f32, ppm), the re-implementation's chi and chi_lsqr at 5
% iterations per LSQR solve in double precision (f64, ppm).
N = [20 20 14]; vs = [1 1 1.5]; H = [0.1 -0.2 sqrt(0.95)]; TE = 20; B0 = 3; pad = [8 8 8];
[ph, mk] = mksyn(N, vs, H, TE, B0, 31);
ph = ph + 0.2*smooth3(randn(N), 'box', 3).*~mk;      % field outside the mask too (STI uses it)
ph = double(single(ph));                              % the fixture stores the phase as f32
evalc('[S, S1] = QSM_iLSQR(ph, mk, ''TE'', TE, ''B0'', B0, ''H'', H, ''padsize'', pad, ''voxelsize'', vs);');
[Xd, Xd1] = ilsqr_sti_clone(ph, mk, vs, H, TE, B0, pad, 5, 'double');
m = mk > 0;
fid = fopen(out_file, 'w');
fwrite(fid, ph(:), 'float32', 'ieee-le'); fwrite(fid, uint8(mk(:)), 'uint8');
fwrite(fid, S(m), 'float32', 'ieee-le'); fwrite(fid, S1(m), 'float32', 'ieee-le');
fwrite(fid, Xd(m), 'float64', 'ieee-le'); fwrite(fid, Xd1(m), 'float64', 'ieee-le');
fclose(fid);
end

function [phs, mask, chi] = mksyn(N, vs, H, TE, B0, seed)
% Random spheres of susceptibility in an ellipsoid mask; phase via the dipole kernel (rad).
rng(seed); [x,y,z] = ndgrid(1:N(1),1:N(2),1:N(3)); c = (N+1)/2;
X = (x-c(1))*vs(1); Y = (y-c(2))*vs(2); Z = (z-c(3))*vs(3); R = min(N.*vs)/2 - 6;
mask = double((X/R).^2 + (Y/(0.9*R)).^2 + (Z/(0.85*R)).^2 < 1);
chi = 0.02*randn(N);
for k = 1:12, p = (rand(1,3)-0.5)*R; r = 2 + 6*rand; chi = chi + (0.2*randn) * ((X-p(1)).^2+(Y-p(2)).^2+(Z-p(3)).^2 < r^2); end
chi = chi .* mask;
P = 2*N; [kx,ky,kz] = ndgrid(ifftshift((-P(1)/2:P(1)/2-1)/(P(1)*vs(1))), ifftshift((-P(2)/2:P(2)/2-1)/(P(2)*vs(2))), ifftshift((-P(3)/2:P(3)/2-1)/(P(3)*vs(3))));
k2 = kx.^2+ky.^2+kz.^2; D = 1/3 - (kx*H(1)+ky*H(2)+kz*H(3)).^2 ./ k2; D(1) = 0;
cp = zeros(P); cp(1:N(1),1:N(2),1:N(3)) = chi; f = real(ifftn(fftn(cp).*D)); f = f(1:N(1),1:N(2),1:N(3));
phs = f * 2*pi*42.576*B0*TE*1e-3 .* mask;   % rad
phs = phs + 0.01*randn(N).*mask;
end
