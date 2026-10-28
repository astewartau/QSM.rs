function sharp_sepia_small(sepia_home, medi_home, out)
% Generates tests/data/sharp_sepia_small.bin: SEPIA's SHARP (Tian Liu's SHARP.m, kschan0214/sepia
% db939cc, with MEDI 2024.11.26's SMV.m / SMV_kernel.m / sphere_kernel.m) on small polynomial
% phantoms, for tests/sharp_sepia_parity.rs.
%
% It runs SEPIA's own background-field-removal path, BackgroundRemovalMacro with
% algorParam.bfr.method = 'SHARP' and the rest of the settings SEPIA's GUI exports for SHARP
% (refine 'None', no extra erosion): SEPIA zero-pads odd dimensions, Wrapper_BFR_SHARP adds MEDI
% to the path and calls SHARP.m, and the padding is cropped off.
%
% Run in licensed MATLAB R2023b (Bunya, a Slurm job, 4 CPUs):
%
%   module load matlab/R2023b
%   matlab -nodisplay -batch "addpath('tests/data'); sharp_sepia_small( ...
%       '/path/to/sepia', '/path/to/MEDI_toolbox-2024.11.26', 'tests/data/sharp_sepia_small.bin')"
%
% The phantom is built from + - * / only, so the Rust test rebuilds it bit for bit. Layout of the
% output (little-endian), per case in CASES order: the number of voxels in SHARP.m's eroded mask
% (u32); that mask (SMV(Mask, ..., radius) > 0.999, the expression SHARP.m uses, on SEPIA's padded
% grid, cropped) as bits over the volume (column-major, least significant bit first, ceil(n/8)
% bytes); SEPIA's local field at those voxels (f64, column-major order).

fid = fopen(fullfile(sepia_home, 'SpecifyToolboxesDirectory.m'), 'w');
fprintf(fid, 'MEDI_HOME = ''%s'';\n', medi_home); fclose(fid);
addpath(sepia_home); sepia_addpath;

CASES = {
    % name, dims, voxel size, radius (mm), threshold
    'SEPIA defaults (4 mm, 0.03), 1 mm',        [40 36 32], [1 1 1],       4,   0.03
    'SEPIA defaults, anisotropic, odd dims',    [33 29 25], [0.9 1.0 1.3], 4,   0.03
    'r 2.5 mm, thr 0.1, anisotropic',           [32 28 24], [1.1 0.8 1.2], 2.5, 0.1
};

fo = fopen(out, 'w');
for c = 1:size(CASES, 1)
    [name, n, v, r, thr] = CASES{c, :};
    [f, m] = phantom(n, v);
    algorParam = struct();
    algorParam.bfr = struct('method', 'SHARP', 'refine_method', 'None', 'refine_order', 4, ...
        'erode_radius', 0, 'erode_before_radius', 0, 'radius', r, 'threshold', thr);
    RDF = BackgroundRemovalMacro(f, m, n, v, algorParam, struct());
    assert(isreal(RDF));
    % SHARP.m's eroded mask, on the grid SEPIA pads to
    mp = zeropad_odd_dimension(m, 'pre');
    me = SMV(mp, size(mp), v, r) > 0.999;
    me = me(1:n(1), 1:n(2), 1:n(3));
    assert(all(RDF(~me) == 0));
    fwrite(fo, uint32(nnz(me)), 'uint32', 0, 'l');
    bits = zeros(1, ceil(numel(me) / 8), 'uint8');
    idx = find(me(:)) - 1;
    for l = idx'
        b = floor(l / 8) + 1;
        bits(b) = bitor(bits(b), bitshift(uint8(1), mod(l, 8)));
    end
    fwrite(fo, bits, 'uint8');
    fwrite(fo, RDF(me), 'float64', 0, 'l');
    fprintf('%s: %d mask voxels, %d eroded\n', name, nnz(m), nnz(me));
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
