function lin_probe(infile, outf)
% lsqcurvefit (defaults) on F(x) = F0 + J*(x - x0): records every evaluation point.
fid = fopen(infile); v = fread(fid, Inf, 'double'); fclose(fid);
o = fopen(outf, 'w'); c = onCleanup(@() fclose(o));
p = 2; nc = v(1);
global EV
for k = 1:nc
    m = v(p); n = v(p+1); p = p + 2;
    x0 = v(p:p+n-1)'; lb = v(p+n:p+2*n-1)'; ub = v(p+2*n:p+3*n-1)'; p = p + 3*n;
    F0 = v(p:p+m-1); p = p + m; J = reshape(v(p:p+m*n-1), m, n); p = p + m*n;
    EV = {};
    fun = @(x, xd) rec(F0 + J*(x(:) - x0(:)), x);
    opt = optimoptions('lsqcurvefit', 'Display', 'off');
    [x, resn, ~, flag, out] = lsqcurvefit(fun, x0, [], zeros(m,1), lb, ub, opt);
    fprintf(o, 'C %d %d %d %d\n', k, flag, out.iterations, out.funcCount);
    for e = 1:numel(EV); fprintf(o, 'E %d %d %s\n', k, e-1, sprintf('%.17g ', EV{e})); end
end
end
function y = rec(y, x)
global EV
EV{end+1} = x(:)';
end
