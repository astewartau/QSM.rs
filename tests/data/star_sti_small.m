function star_sti_small(out_file)
% Writes tests/data/star_sti_small.bin: STI Suite 3.0's QSM_star in double precision on four
% 48x56x40 phantoms, for tests/star_sti_parity.rs.
%
% Recovered 2026-10-30 from the sessions that made the fixture (2026-10-20: the first three cases;
% 2026-10-22: the GUI-padding case, appended), merged into one script; not re-run since. Run in
% licensed MATLAB R2023b with STI Suite 3.0 on the path:
%   st = '<STISuite_V3.0>';
%   addpath(fullfile(st, 'Core_Functions_P')); addpath(fullfile(st, 'Support_Functions', 'SpaRSA'));
%   addpath(fullfile(st, 'Support_Functions', 'qsm_kiwi_1')); star_sti_small('star_sti_small.bin')
% calcD2Matrix, fftnc and ifftnc are STI's own (P-code). The QSM-CI star-sti container
% (star_sti_qsmci_small.py) writes chi as float32 and takes one padsize for all axes, so the
% double-precision comparison and the GUI's [8 8 20] padding need this licensed-MATLAB fixture.
%
% Layout (little-endian), per case: the number of mask voxels (u32), then STI's chi at the mask
% voxels (f64, ppm, column-major).
% {voxel size, H, B0 (T), TE (ms), padsize}
cases = { {[1 1 1],       [0 0 1],       3, 20, [12 12 12]}, ...
          {[0.8 0.9 1.5], [0 0 1],       3, 20, [12 12 12]}, ...
          {[1 1 1],       [0.1 -0.4 0.9], 7, 5, [8 8 8]}, ...
          {[0.8 0.9 1.5], [0 0 1],       3, 20, [8 8 20]} };    % QSM_GUI's padding
sz = [48 56 40];
fid = fopen(out_file, 'w');
for c = 1:numel(cases)
    a = cases{c}; vox = a{1}; H = a{2} / norm(a{2}); B0 = a{3}; TE = a{4}; pad = a{5};
    [chi0, mask] = mk_phantom(sz, vox);
    D = calcD2Matrix(sz, vox, H);
    field = real(ifftnc(D .* fftnc(chi0))) .* mask;
    TP = field * 1e-6 * 2*pi * 42.576e6 * B0 * TE / 1000;
    chi = QSM_star(TP, double(mask), 'TE', TE, 'B0', B0, 'H', H, 'padsize', pad, 'voxelsize', vox);
    m = mask > 0;
    fwrite(fid, uint32(nnz(m)), 'uint32'); fwrite(fid, chi(m), 'double');
end
fclose(fid);
end

function [chi, mask] = mk_phantom(sz, vox)
% Spheres of various chi inside an ellipsoid brain mask.
[x,y,z] = ndgrid((1:sz(1))-sz(1)/2-0.5, (1:sz(2))-sz(2)/2-0.5, (1:sz(3))-sz(3)/2-0.5);
x=x*vox(1); y=y*vox(2); z=z*vox(3);
R = 0.4*min(sz.*vox);
mask = (x/R).^2 + (y/(0.9*R)).^2 + (z/(0.8*R)).^2 < 1;
chi = zeros(sz);
c = [0.3 0.2 0; -0.3 0.1 0.1; 0 -0.35 -0.2; 0.1 0.1 0.3].*R;
v = [0.1 -0.05 0.2 0.5];
r = [0.15 0.2 0.12 0.08]*R;
for i=1:4, chi(((x-c(i,1)).^2+(y-c(i,2)).^2+(z-c(i,3)).^2) < r(i)^2) = v(i); end
chi = chi.*mask;
end
