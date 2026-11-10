% Fixture for qsm_core::utils::r2star::chisep_nnls: the χ-separation toolbox v1.1.3's
% r2star_nnls (P-code) on 300 noisy mono-exponential voxels, with what lsqnonlin did on each.
% Writes r2star_nnls_chisep_matlab.txt: per voxel, its five echo magnitudes, then lsqnonlin's
% x(1) x(2) resnorm exitflag iterations funcCount. r2star_nnls returns x(2) as R2*.
%
% Needs licensed MATLAB with the Optimization Toolbox and the χ-separation toolbox:
%   matlab -batch "chisep = '<toolbox dir>'; r2star_nnls_chisep_matlab"
% Last run 2026-11-10, MATLAB R2023b Update 5 (23.2.0.2459199).
%
% r2star_nnls calls lsqnonlin inside a parfor; with no pool the loop runs here, so a logging
% lsqnonlin ahead of MATLAB's on the path sees every call. It runs the real lsqnonlin again
% with all outputs to record the solver's report.
global LSQ_LOG LSQ_DIR
addpath(genpath(chisep));
LSQ_DIR = tempname; mkdir(LSQ_DIR);
fid = fopen(fullfile(LSQ_DIR, 'lsqnonlin.m'), 'w');
fprintf(fid, '%s\n', ...
  'function varargout = lsqnonlin(varargin)', ...
  'global LSQ_LOG LSQ_DIR', ...
  'rmpath(LSQ_DIR); c = onCleanup(@() addpath(LSQ_DIR));', ...
  '[varargout{1:nargout}] = lsqnonlin(varargin{:});', ...
  '[x, resnorm, ~, exitflag, output] = lsqnonlin(varargin{:});', ...
  'fi = functions(varargin{1}); w = fi.workspace{1};', ...
  'e = struct(''ii'', w.ii, ''slice'', w.slice, ''x'', x, ''resnorm'', resnorm, ''exitflag'', exitflag, ...', ...
  '    ''iter'', output.iterations, ''fc'', output.funcCount);', ...
  'LSQ_LOG{end+1} = e;', ...
  'end');
fclose(fid);
addpath(LSQ_DIR);
ps = parallel.Settings; ps.Pool.AutoCreate = false;

TE = [5 11 17 23 29];   % ms, as Chisep_script.m passes TE*1000
rng(7); n = 300; R = [0 0.5 2 5 10 20 40 80 150 300 600 1000 2000];
r2t = reshape(R(randi(numel(R), n, 1)), [], 1) .* (0.8 + 0.4*rand(n, 1));
s0 = 10.^(4*rand(n, 1) - 1); snr = [Inf 1000 100 20 5 2];
sn = reshape(snr(randi(numel(snr), n, 1)), [], 1);
mag = zeros(n, 5);
for e = 1:5, mag(:, e) = s0 .* exp(-r2t*TE(e)/1000) + (s0./sn) .* randn(n, 1); end
v = reshape(mag, 10, 6, 5, 5);
LSQ_LOG = {};
r = r2star_nnls(v, TE, true(10, 6, 5));

fit = zeros(n, 6);
for i = 1:numel(LSQ_LOG)
  e = LSQ_LOG{i}; k = e.ii + (e.slice - 1)*60;
  assert(r(k) == e.x(2));
  fit(k, :) = [e.x(1) e.x(2) e.resnorm e.exitflag e.iter e.fc];
end
fid = fopen('r2star_nnls_chisep_matlab.txt', 'w');
fprintf(fid, '# r2star_nnls (χ-separation toolbox v1.1.3) in MATLAB R2023b Update 5, 2026-11-10: 300 noisy mono-exponential\n');
fprintf(fid, '# voxels at TE = [5 11 17 23 29] ms (tests/data/r2star_nnls_chisep_matlab.m). Per voxel: the five echo\n');
fprintf(fid, '# magnitudes, then lsqnonlin''s x(1) x(2) resnorm exitflag iterations funcCount.\n');
for k = 1:n
  fprintf(fid, '%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %d %d\n', mag(k, :), fit(k, :));
end
fclose(fid);
rmpath(LSQ_DIR);
