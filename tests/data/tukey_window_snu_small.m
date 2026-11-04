function tukey_window_snu_small(out_file)
% Writes tests/data/tukey_window_snu_small.bin: the SNU chi-separation toolbox's tukey_windowing on
% four small complex multi-echo volumes, for tests/tukey_window_snu_parity.rs.
%
% Reference: chi-separation toolbox v1.1.3 (SNU-LIST, obtained through its Google Form; files dated
% 2025-07-02), functions/supportfunc/tukey_windowing.p and the utils/fft3c.p and utils/ifft3c.p it
% calls, run in licensed MATLAB R2023b (Signal Processing Toolbox's tukeywin) on a Bunya Slurm
% node, as Chisep_script.m:71 calls it: tukey_windowing(complex multi-echo data, tukey_fac).
%   addpath(genpath('<toolbox>')); tukey_window_snu_small('tukey_window_snu_small.bin')
%   sha256 tukey_windowing.p e4703efee3397eea8b95a6fcd461f757ec775f9708a4564d50b38fee94e85153
%   sha256 fft3c.p           8a6f493f3ce73e6e85cd69190dc552a3dfb86a61dc882d7e396c2581b69f2d71
%   sha256 ifft3c.p          f5afa8ecd47a952cab6c7337f6855e95637c36f5e73db9e22e32f8591bc7a6d1
%
% The inputs are built from an integer LCG only (real and imaginary parts, no transcendental
% functions), so Rust rebuilds them exactly. Layout (little-endian), per case in order: the
% toolbox's output, real parts then imaginary parts, each over all voxels and echoes in
% column-major order (x fastest, echo slowest), f64.
cases = { [10 9 7 3], 0.4;    % Siemens/GE value, even and odd axes, three echoes
          [16 12 8 1], 0.25;  % even axes, one echo
          [7 8 1 2], 1;       % a Hann window (factor >= 1), a singleton axis
          [6 5 4 2], 0 };     % Philips value: the identity
fid = fopen(out_file, 'w');
for c = 1:size(cases, 1)
    [n, r] = cases{c, :};
    X = input_volume(n, c);
    Y = tukey_windowing(X, r);
    fprintf('case %d: %s, factor %g, max |Y - X| %g\n', c, mat2str(n), r, max(abs(Y(:) - X(:))));
    fwrite(fid, real(Y(:)), 'double'); fwrite(fid, imag(Y(:)), 'double');
end
fclose(fid);
end

function X = input_volume(n, seed)
% Real and imaginary parts in [-1, 1) from an LCG over the linear index (column-major).
N = prod(n);
u = @(s) mod(1103515245 * ((0:N-1)' + s * 7919) + 12345, 2^31) / 2^31;
X = reshape(complex(2 * u(2 * seed) - 1, 2 * u(2 * seed + 1) - 1), [n 1]);
end
