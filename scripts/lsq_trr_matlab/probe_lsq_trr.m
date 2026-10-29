function probe_lsq_trr(outfile, yfile, decompose_utils)
% Black-box record of MATLAB's lsqcurvefit (default trust-region-reflective) for QSM.rs's
% solvers::lsq_trr (tests/lsq_trr_matlab.rs, separation::decompose's replay test).
%
%   probe_lsq_trr(outfile)                          generic problems only
%   probe_lsq_trr(outfile, yfile, decompose_utils)  also every stage call of the DECOMPOSE fit
%
% Only lsqcurvefit's documented interface is used: the objective records every point it is
% evaluated at, and an OutputFcn records every optimValues field that is a numeric scalar.
% yfile (written by `cargo test ... decompose_probe_input`): f64 little-endian
%   [n_voxels, n_echoes, B0, TE(1..ne), then per voxel Re(conj y)(1..ne), Im(conj y)(1..ne)].
%
% Output lines (all numbers %.17g):
%   P <tag> m n x0... lb... ub...                problem header
%   E <tag> k x... ssq                           k-th evaluation (1-based) and its sum of squares
%   I <tag> <state> name=value ... x=x1,x2,..    OutputFcn call
%   R <tag> exitflag iterations funcCount firstorderopt resnorm x...
%   S <voxel> <pass> <stage> chi+ chi- R0 C+ C- C0   DECOMPOSE stage call: the fixed params
fid = fopen(outfile, 'w');
fprintf(fid, 'V %s\n', version);
c = onCleanup(@() fclose(fid));

G = generic_problems();
for p = 1:numel(G)
    run_one(fid, sprintf('G%d', p), @(x, xd) G(p).fun(x), G(p).x0, 0, zeros(G(p).m, 1), G(p).lb, G(p).ub);
end

if nargin < 2 || isempty(yfile); return; end
addpath(decompose_utils);
f = fopen(yfile, 'r'); h = fread(f, 3, 'double'); nv = h(1); ne = h(2); B0 = h(3);
TE = fread(f, ne, 'double')'; Y = fread(f, [2*ne, nv], 'double'); fclose(f);
lb = [0 0 0]; ub = [inf inf inf]; lb2 = 0; ub2 = inf; upper = 0.5; lb3 = [0 -upper]; ub3 = [upper 0];
for v = 1:nv
    ydata = complex(Y(1:ne, v), Y(ne+1:end, v)).';   % already conj(y), a row as theirs is
    if sum(ydata) == 0; continue; end
    chi_0 = [0.05 -0.05]; R0 = 25; C0 = [0.3 0.3 0.4];
    for it = 1:10
        stage(fid, v, it, 1, chi_0, R0, C0);
        m1 = @(x,xd) complex_to_real(objective1(x, xd, chi_0(1), chi_0(2), R0, B0));
        C0 = run_one(fid, sprintf('D%d_%d_1', v-1, it), m1, C0, TE, complex_to_real(ydata), lb, ub);
        stage(fid, v, it, 2, chi_0, R0, C0);
        m2 = @(x,xd) complex_to_real(objective2(x, xd, C0(1), C0(2), C0(3), chi_0(1), chi_0(2), B0));
        R0 = run_one(fid, sprintf('D%d_%d_2', v-1, it), m2, R0, TE, complex_to_real(log(ydata)), lb2, ub2);
        stage(fid, v, it, 3, chi_0, R0, C0);
        m3 = @(x,xd) complex_to_real(objective3(x, xd, C0(1), C0(2), C0(3), R0, B0));
        chi_0 = run_one(fid, sprintf('D%d_%d_3', v-1, it), m3, chi_0, TE, complex_to_real(log(ydata)), lb3, ub3);
    end
end
end

function stage(fid, v, it, s, chi_0, R0, C0)
fprintf(fid, 'S %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g\n', v-1, it, s, chi_0(1), chi_0(2), R0, C0(1), C0(2), C0(3));
end

function x = run_one(fid, tag, fun, x0, xdata, ydata, lb, ub)
global PROBE_FID PROBE_TAG PROBE_K
PROBE_FID = fid; PROBE_TAG = tag; PROBE_K = 0;
fprintf(fid, 'P %s %d %d %s %s %s\n', tag, numel(ydata), numel(x0), nums(x0), nums(lb), nums(ub));
opt = optimoptions('lsqcurvefit', 'Display', 'off', 'UseParallel', false, 'OutputFcn', @outfun);
wrapped = @(x, xd) record(fun, x, xd, ydata);
[x, resnorm, ~, exitflag, output] = lsqcurvefit(wrapped, x0, xdata, ydata, lb, ub, opt);
fprintf(fid, 'R %s %d %d %d %.17g %.17g %s\n', tag, exitflag, output.iterations, output.funcCount, ...
    output.firstorderopt, resnorm, nums(x));
end

function y = record(fun, x, xd, ydata)
global PROBE_FID PROBE_TAG PROBE_K
y = fun(x, xd);
r = y(:) - ydata(:);
PROBE_K = PROBE_K + 1;
fprintf(PROBE_FID, 'E %s %d %s %.17g\n', PROBE_TAG, PROBE_K, nums(x), sum(r.^2));
end

function stop = outfun(x, ov, state)
global PROBE_FID PROBE_TAG
stop = false;
s = sprintf('I %s %s', PROBE_TAG, state);
for fn = fieldnames(ov)'
    v = ov.(fn{1});
    if isnumeric(v) && isscalar(v) || islogical(v) && isscalar(v)
        s = [s sprintf(' %s=%.17g', fn{1}, double(v))]; %#ok<AGROW>
    end
end
fprintf(PROBE_FID, '%s x=%s\n', s, strrep(nums(x), ' ', ','));
end

function s = nums(v)
s = strjoin(arrayfun(@(a) sprintf('%.17g', a), v(:)', 'UniformOutput', false), ' ');
end

function G = generic_problems()
% Keep in step with tests/lsq_trr_matlab.rs::generic().
inf_ = inf;
t4 = [0 1 2 3]'; d4 = [1 3 5 7]';
lin = @(x) x(1) * t4 + x(2) - d4;
rosen = @(x) [10 * (x(2) - x(1)^2); 1 - x(1)];
te = (0:0.5:3)';
e1 = @(x) exp(-x(1) * te) - exp(-2 * te);
t3 = (0:0.25:5)';
d3 = 2 * exp(-0.7 * t3) + 0.3;
d3b = 2 * exp(-0.7 * t3) - 0.2;
e3 = @(x) x(1) * exp(-x(2) * t3) + x(3) - d3;
e3b = @(x) x(1) * exp(-x(2) * t3) + x(3) - d3b;
G = struct('fun', {}, 'x0', {}, 'lb', {}, 'ub', {}, 'm', {});
G(end+1) = struct('fun', lin, 'x0', [0 0], 'lb', [-inf_ -inf_], 'ub', [inf_ inf_], 'm', 4);              % G1
G(end+1) = struct('fun', rosen, 'x0', [-1.2 1], 'lb', [-inf_ -inf_], 'ub', [inf_ inf_], 'm', 2);         % G2
G(end+1) = struct('fun', rosen, 'x0', [-1.2 1], 'lb', [-2 -2], 'ub', [0.5 2], 'm', 2);                   % G3
G(end+1) = struct('fun', rosen, 'x0', [-2 -2], 'lb', [-2 -2], 'ub', [0.5 2], 'm', 2);                    % G4 x0 on lb
G(end+1) = struct('fun', rosen, 'x0', [3 3], 'lb', [-2 -2], 'ub', [0.5 2], 'm', 2);                      % G5 x0 outside
G(end+1) = struct('fun', e1, 'x0', 0, 'lb', 0, 'ub', inf_, 'm', numel(te));                              % G6 x0 on lb
G(end+1) = struct('fun', e1, 'x0', 5, 'lb', 0, 'ub', 2.5, 'm', numel(te));                               % G7 x0 above ub
G(end+1) = struct('fun', e1, 'x0', 2.5 - 1e-9, 'lb', 0, 'ub', 2.5, 'm', numel(te));                      % G8 FD at ub
G(end+1) = struct('fun', lin, 'x0', [2 1], 'lb', [-inf_ -inf_], 'ub', [inf_ inf_], 'm', 4);              % G9 optimal x0
G(end+1) = struct('fun', @(x) 1e4 * rosen(x), 'x0', [-1.2 1], 'lb', [-inf_ -inf_], 'ub', [inf_ inf_], 'm', 2); % G10 scaled
G(end+1) = struct('fun', e3, 'x0', [1 1 1], 'lb', [0 0 0], 'ub', [inf_ inf_ inf_], 'm', numel(t3));      % G11
G(end+1) = struct('fun', rosen, 'x0', [-1.2 1], 'lb', [-inf_ -inf_], 'ub', [0.5 inf_], 'm', 2);          % G12 one-sided
G(end+1) = struct('fun', e3, 'x0', [1e-8 1 1e-8], 'lb', [0 0 0], 'ub', [inf_ inf_ inf_], 'm', numel(t3)); % G13 near lb
G(end+1) = struct('fun', e3b, 'x0', [1 1 1], 'lb', [0 0 0], 'ub', [inf_ inf_ inf_], 'm', numel(t3));     % G14 active lb
G(end+1) = struct('fun', rosen, 'x0', [-1.2 1], 'lb', [-1.5 -inf_], 'ub', [-1 inf_], 'm', 2);           % G15 two-sided active
G(end+1) = struct('fun', @(x) rosen(x) - [0; 0.5], 'x0', [0 0], 'lb', [-inf_ -inf_], 'ub', [inf_ inf_], 'm', 2); % G16 x0 = 0
end
