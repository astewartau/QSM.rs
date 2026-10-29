function ismv_wen2014_small(msmv_home, out)
% Generates tests/data/ismv_wen2014_small.bin: the iSMV authors' lab code, iSMV.m with its kernel
% tdcsmv.m (github.com/agr78/mSMV 42a1f18, code/dependencies/MEDI_functions/), on small
% polynomial phantoms, for tests/ismv_wen2014_parity.rs.
%
% iSMV.m is run unmodified for the case at its own tolerance (0.001). For the cases at Wen et al.
% 2014's stopping rule (relative change < 5e-5) it is run with exactly one change: the constant in
% `while (d>=.001)` becomes a `tol` argument (a string replacement on the shipped file, checked
% below). The odd-size case is zero-padded by one slice at the end of each odd axis first and
% cropped back afterwards, as SEPIA's zeropad_odd_dimension does (iSMV.m errors on odd sizes).
%
% Run in GNU Octave (Debian bookworm's octave 7.3; iSMV.m and tdcsmv.m use only core functions):
%
%   podman run --rm -v "$PWD/tests/data":/data -v /path/to/mSMV:/msmv:ro <debian:bookworm-slim
%     with apt's octave> octave --no-gui --eval \
%     "addpath('/data'); ismv_wen2014_small('/msmv', '/data/ismv_wen2014_small.bin')"
%
% The phantom is built from + - * / only, so the Rust test rebuilds it bit for bit. Layout of the
% output (little-endian), per case in CASES order: the number of voxels in iSMV.m's eroded mask M
% (u32); M as bits over the volume (column-major, least significant bit first, ceil(n/8) bytes);
% real(L) at those voxels (f64, column-major order).

src_dir = fullfile(msmv_home, 'code', 'dependencies', 'MEDI_functions');
addpath(src_dir);
tmp = tempname(); mkdir(tmp);
s = fileread(fullfile(src_dir, 'iSMV.m'));
s2 = strrep(s, 'function [L,M] = iSMV(iFreq,Mask,matrix_size,voxel_size,radius)', ...
    'function [L,M] = iSMV_tol(iFreq,Mask,matrix_size,voxel_size,radius,tol)');
s2 = strrep(s2, 'while (d>=.001)', 'while (d>=tol)');
assert(numel(strfind(s2, 'iSMV_tol(')) == 1 && numel(strfind(s2, 'while (d>=tol)')) == 1);
fid = fopen(fullfile(tmp, 'iSMV_tol.m'), 'w'); fwrite(fid, s2); fclose(fid);
addpath(tmp);

CASES = {
    % name, dims, voxel size, radius (mm), tol (0 = iSMV.m unmodified, i.e. 0.001)
    'Wen 2014 defaults (6 voxels, 5e-5), 1 mm',          [40 36 32], [1 1 1],       6,   5e-5
    'Wen 2014 defaults, anisotropic 0.9 x 0.9 x 1.3 mm', [36 32 26], [0.9 0.9 1.3], 5.4, 5e-5
    'iSMV.m as shipped (tol 0.001), r 4 mm',             [36 32 28], [1 1 1],       4,   0
    'Wen 2014 defaults, odd dims (padded)',              [33 29 25], [1 1 1],       6,   5e-5
};

fo = fopen(out, 'w');
for c = 1:size(CASES, 1)
    [name, n, v, r, tol] = CASES{c, :};
    [f, m] = phantom(n, v);
    np = n + mod(n, 2);
    fp = zeros(np); mp = zeros(np);
    fp(1:n(1), 1:n(2), 1:n(3)) = f;
    mp(1:n(1), 1:n(2), 1:n(3)) = m;
    if tol == 0
        [L, M] = iSMV(fp, mp, np, v, r);
    else
        [L, M] = iSMV_tol(fp, mp, np, v, r, tol);
    end
    L = L(1:n(1), 1:n(2), 1:n(3));
    M = M(1:n(1), 1:n(2), 1:n(3));
    fprintf('%s: max |imag(L)| %g\n', name, max(abs(imag(L(:)))));
    L = real(L);
    assert(all(L(~M) == 0));
    fwrite(fo, uint32(nnz(M)), 'uint32', 0, 'l');
    bits = zeros(1, ceil(numel(M) / 8), 'uint8');
    idx = find(M(:)) - 1;
    for l = idx'
        b = floor(l / 8) + 1;
        bits(b) = bitor(bits(b), bitshift(uint8(1), mod(l, 8)));
    end
    fwrite(fo, bits, 'uint8');
    fwrite(fo, L(M), 'float64', 0, 'l');
    fprintf('%s: %d mask voxels, %d eroded\n', name, nnz(m), nnz(M));
end
fclose(fo);
end

function [f, m] = phantom(n, v)
% tests/data/sharp_qsmjl_small.jl's phantom: an ellipsoid mask holding a harmonic background and
% two spheres of local field, zero outside the mask.
f = zeros(n); m = zeros(n);
a = 0.46*n(1)*v(1); b = 0.44*n(2)*v(2); c = 0.42*n(3)*v(3);
for k = 1:n(3), for j = 1:n(2), for i = 1:n(1)
    x = ((i-1) - (n(1)-1)/2 + 0.3)*v(1);
    y = ((j-1) - (n(2)-1)/2 + 0.3)*v(2);
    z = ((k-1) - (n(3)-1)/2 + 0.3)*v(3);
    if x*x/(a*a) + y*y/(b*b) + z*z/(c*c) < 1.0
        m(i,j,k) = 1;
        val = 0.02*z + 0.003*(x*x - y*y) + 1.0e-4*x*y*z;
        p = x - 0.2*a; q = 0.25*a;
        if p*p + y*y + z*z < q*q, val = val + 0.05; end
        p = x + 0.3*a; s = y - 0.2*b; t = z + 0.1*c; q = 0.2*a;
        if p*p + s*s + t*t < q*q, val = val - 0.03; end
        f(i,j,k) = val;
    end
end, end, end
end
