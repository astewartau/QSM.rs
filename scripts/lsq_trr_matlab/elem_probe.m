function elem_probe(d, utils)
% Bit-level record of MATLAB's elementary functions on the inputs in d (f64 LE files).
% Writes <name>_<mode>.bin: modes vec (whole vector), r8 (8-element rows, as DECOMPOSE calls),
% sc (scalar calls, first 40000 inputs).
addpath(utils);
rd = @(n) fread_all(fullfile(d, [n '.bin']));
wr = @(n, v) fwrite_all(fullfile(d, 'out', [n '.bin']), v);
mkdir(fullfile(d, 'out'));
fprintf('%s\n', version);
x = rd('exp_in'); run3(@exp, x, 'exp', wr);
x = rd('trig_in'); run3(@cos, x, 'cos', wr); run3(@sin, x, 'sin', wr);
x = rd('log_in'); run3(@log, x, 'log', wr); run3(@sqrt, x, 'sqrt', wr);
z = complex(rd('cz_re'), rd('cz_im'));
run3(@abs, z, 'cabs', wr); run3(@angle, z, 'cangle', wr);
lz = run3(@log, z, 'clog', wr);
re = rd('cz_re'); im = rd('cz_im');
run3c(@(a) atan2(imag(a), real(a)), z, 'atan2', wr);
run3c(@(a) hypot(real(a), imag(a)), z, 'hypot', wr);
ze = complex(rd('ce_re'), rd('ce_im'));
run3(@exp, ze, 'cexp', wr);
% The model as DECOMPOSE evaluates it.
P = reshape(rd('model_p'), [], 6);
TE = [0.003 0.009 0.015 0.021 0.027 0.033 0.039 0.045]; TE(2) = 0.009000000000000001; B0 = 7;
n = size(P, 1); S = zeros(n, 8); L = zeros(n, 8);
for r = 1:n
    s = signalModel(TE, P(r,4), P(r,5), P(r,6), P(r,1), P(r,2), P(r,3), B0);
    S(r,:) = s; L(r,:) = log(s);
end
wr('model_re', real(S).'); wr('model_im', imag(S).'); wr('mlog_re', real(L).'); wr('mlog_im', imag(L).');
fprintf('done\n');
end
function y = run3(f, x, name, wr)
    y = f(x); wr([name '_vec_re'], real(y)); wr([name '_vec_im'], imag(y));
    n8 = floor(numel(x)/8)*8; X = reshape(x(1:n8), 8, []); Y = zeros(size(X)) * (1 + 0i*~isreal(x));
    for c = 1:size(X, 2); Y(:, c) = f(X(:, c).'); end
    wr([name '_r8_re'], real(Y(:))); wr([name '_r8_im'], imag(Y(:)));
    ns = min(40000, numel(x)); Ys = zeros(ns, 1);
    for k = 1:ns; Ys(k) = f(x(k)); end
    wr([name '_sc_re'], real(Ys)); wr([name '_sc_im'], imag(Ys));
end
function run3c(f, z, name, wr)
    run3(f, z, name, wr);
end
function v = fread_all(f)
    h = fopen(f, 'r'); v = fread(h, inf, 'double'); fclose(h);
end
function fwrite_all(f, v)
    h = fopen(f, 'w'); fwrite(h, v(:), 'double'); fclose(h);
end
