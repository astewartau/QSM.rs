function medi_smv_small(d, release_functions, prerelease_functions, out_file)
% Writes tests/data/medi_smv_small.bin: MEDI_L1's SMV options on the phantom of
% tests/support/medi_phantom.rs, one Gauss-Newton step, no CSF term, for tests/medi_smv_parity.rs.
%
% Reference: MEDI toolbox release 2024.11.26 (functions/) and the mSMV prerelease's toolbox
% (agr78/mSMV 42a1f18, prerelease/MEDI_toolbox-mSMV100125.zip, functions/), each run with only
% its own functions/ on the path, in licensed MATLAB R2023b. Assembled 2026-10-30 from the script
% that was in tests/medi_smv_parity.rs's docs, with the writing of the fixture added (layout
% there). The release's three cases are also made by the verified QSM-CI medi-cornell container
% (tests/data/medi_cornell_small.py, regenerated in Actions); the prerelease has no container.
%
%   MEDI_PARITY_DUMP=<d> cargo test --test medi_smv_parity -- --ignored dump_inputs
%   matlab -batch "medi_smv_small('<d>', '<MEDI 2024.11.26>/functions', '<mSMV prerelease>/functions', 'medi_smv_small.bin')"
N = [32 36 28]; vs = [1 1 1.2]; TE = [4 8 12 16]*1e-3; B0 = 3; n = prod(N);
rd = @(f,t,k) reshape(fread(fopen(fullfile(d,f)),k,t), [N k/n]);
RDF = rd('rdf.f64','double',n); Mask = rd('mask.u8','uint8',n) > 0; mag = rd('mag.f64','double',4*n);
one = {'lambda', 1000, 'max_iter', 1};                    % no Mask_CSF: no CSF term
old = path;
addpath(release_functions);
mkrdf(RDF, Mask, mag, N, vs, TE, B0);
chi_nosmv    = MEDI_L1('filename', 'RDF.mat', one{:}, 'nosmv');
chi_shrink   = MEDI_L1('filename', 'RDF.mat', one{:}, 'smv', 5);
chi_noshrink = MEDI_L1('filename', 'RDF.mat', one{:}, 'smv', 5, 'smv_shrink_mask', 0);
path(old); addpath(prerelease_functions);
mkrdf(RDF, Mask, mag, N, vs, TE, B0);
chi_pre = MEDI_L1('filename', 'RDF.mat', one{:}, 'smv', 5, 'smv_shrink_mask', 0);
path(old);
fid = fopen(out_file, 'w');
fwrite(fid, nnz(Mask), 'uint32');
for c = {chi_nosmv, chi_shrink, chi_noshrink, chi_pre}, fwrite(fid, double(c{1}(Mask)), 'double'); end
fclose(fid);
end

function mkrdf(RDF, Mask, mag, N, vs, TE, B0)
[~, N_std] = Fit_ppm_complex(mag); iMag = sqrt(sum(mag.^2, 4));
matrix_size = N; voxel_size = vs; delta_TE = TE(2)-TE(1); CF = 42.576e6*B0; B0_dir = [0 0 1];
iFreq = [];
save RDF.mat RDF iFreq N_std iMag Mask matrix_size voxel_size delta_TE CF B0_dir
end
