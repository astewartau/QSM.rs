function msmv_medi_small(d, out_file)
% Writes tests/data/msmv_medi_small.bin: the mSMV authors' msmv.m alone and MEDI_L1(..., 'msmv', 5)
% on the phantom of tests/support/medi_phantom.rs, for tests/msmv_medi_parity.rs.
%
% Reference: agr78/mSMV 42a1f18, prerelease/MEDI_toolbox-mSMV100125.zip (functions/ on the path),
% licensed MATLAB R2023b with the Image Processing Toolbox (fibermetric). Assembled 2026-10-30
% from the script that was in tests/msmv_medi_parity.rs's docs, with the writing of the fixture
% added (layout there). No QSM-CI container runs the mSMV prerelease.
%
%   MSMV_PARITY_DUMP=<d> cargo test --test msmv_medi_parity -- --ignored dump_inputs
%   matlab -batch "addpath('<mSMV prerelease>/functions'); msmv_medi_small('<d>', 'msmv_medi_small.bin')"
N = [32 36 28]; vs = [1 1 1.2]; TE = [4 8 12 16]*1e-3; B0 = 3; n = prod(N);
rd = @(f,t,k) reshape(fread(fopen(fullfile(d,f)),k,t), [N k/n]);
RDF = rd('rdf.f64','double',n); Mask = rd('mask.u8','uint8',n) > 0;
R2s = rd('r2s.f64','double',n); mag = rd('mag.f64','double',4*n);
for pf = [1 0]                                      % msmv.m alone
  out{pf+1} = msmv(RDF, double(Mask), R2s, vs, 5, B0, pf);
end
t = kernel_lim(RDF, vs, N, double(Mask), B0);       % msmv.m's own lines for t, Me, Mv
SphereK = single(sphere_kernel(N, vs, 5)); Me = double(Mask) - (SMV(double(Mask), SphereK) > 0.999);
Mask_ev = double(Mask) - MaskErode(double(Mask), N, vs, 6);
Mv = imbinarize(fibermetric(Mask_ev.*R2s, 1:15, 'ObjectPolarity', 'bright'), 0);
% MEDI_L1 'msmv' as MEDI's pipeline feeds it (no CSF term)
[~, N_std] = Fit_ppm_complex(mag); iMag = sqrt(sum(mag.^2, 4)); R2s = arlo(TE, mag);
matrix_size = N; voxel_size = vs; delta_TE = TE(2)-TE(1); CF = 42.576e6*B0; B0_dir = [0 0 1];
B0_mag = B0; iFreq = []; prefilter = 1;
save RDF.mat RDF iFreq N_std iMag Mask matrix_size voxel_size delta_TE CF B0_dir R2s B0_mag prefilter
opt = {'lambda', 1000, 'msmv', 5, 'max_iter', 1};
chi_on = MEDI_L1('filename', 'RDF.mat', opt{:});
prefilter = 0; save('RDF.mat', 'prefilter', '-append');
chi_off = MEDI_L1('filename', 'RDF.mat', opt{:});
fid = fopen(out_file, 'w');
fwrite(fid, nnz(Mask), 'uint32'); fwrite(fid, t, 'double');
fwrite(fid, uint8(Me(:) > 0) + uint8(Me(:) > 0 & Mv(:) > 0), 'uint8');
for c = {out{2}, out{1}, chi_on, chi_off}, fwrite(fid, double(c{1}(Mask)), 'double'); end
fclose(fid);
end
