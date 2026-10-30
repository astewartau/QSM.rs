% Recovered for QSM.rs's fixtures from astewartau/ukb-qsmxt (exp/v9-upstream 882785d,
% experiments/v9_upstream/ilsqr_sti_clone.m). Used by ilsqr_sti_small.m; see tests/ilsqr_sti_parity.rs.
function [chi, chi_lsqr, info] = ilsqr_sti_clone(phs, mask, vs, H, TE, B0, padsize, maxit, prec, tol1)
% Re-implementation of STI Suite 3.0 QSM_iLSQR (p-coded), reverse-engineered by black-box
% probing (docs/09_ilsqr_sti.md). Call as STI does:
%   QSM_iLSQR(phs, mask, 'TE', TE, 'B0', B0, 'H', H, 'padsize', padsize, 'voxelsize', vs)
%   == ilsqr_sti_clone(phs, mask, vs, H, TE, B0, padsize)
% phs: tissue phase (rad), not re-masked by STI; mask: ROI; vs: voxel size (mm); H: B0 direction
% in voxel axes; TE in ms; B0 in T; padsize as passed to STI (mm; give it explicitly, STI's own
% default is a different, small pad). chi in ppm (masked); chi_lsqr = the initial LSQR solution
% (ppm), STI's second output.
% maxit (default 100, as STI) caps both LSQR solves. prec 'single' (default) evaluates the LSQR
% operators in single precision on STI's ifftshift-ed layout, as STI does, which makes the initial
% solve bitwise identical to STI; 'double' gives the precision-free algorithm (what QSM.rs runs).
if nargin < 7, padsize = [0 0 0]; end
if nargin < 8 || isempty(maxit), maxit = 100; end
if nargin < 9 || isempty(prec), prec = 'single'; end
if nargin < 10 || isempty(tol1), tol1 = 0.01; end
if strcmp(prec, 'single'), prc = @single; else, prc = @double; end
N0 = size(mask); mask = double(mask);
% ---- crop to the mask's bounding box, odd extents made even (STI's MaskBoundingBox) ----
bb = bbox_even(mask);
P = phs(bb(1,1):bb(1,2), bb(2,1):bb(2,2), bb(3,1):bb(3,2));
M = mask(bb(1,1):bb(1,2), bb(2,1):bb(2,2), bb(3,1):bb(3,2));
% ---- zero-pad: total size = next multiple of 16 strictly above crop + 2*padsize/vs ----
c = size(M); c(end+1:3) = 1;
n = 16 * ceil((c + 2 * padsize(:)' ./ vs(:)' + 1) / 16);
pd = (n - c) / 2;                                    % c is even, so pd is an integer
P = padarray(P, pd); M = padarray(M, pd); Nt = prod(n);
m = M > 0;
% ---- k-space (centred, DC at n/2+1), cycles/mm ----
[k1, k2, k3] = ndgrid((-n(1)/2:n(1)/2-1) / (n(1)*vs(1)), (-n(2)/2:n(2)/2-1) / (n(2)*vs(2)), (-n(3)/2:n(3)/2-1) / (n(3)*vs(3)));
kk = k1.^2 + k2.^2 + k3.^2;
D = 1/3 - (k1*H(1) + k2*H(2) + k3*H(3)).^2 ./ kk; D(kk == 0) = 0;
kk(kk == 0) = 1e6;                                   % STI's DC value (see the Laplacian weights)
fc = @(a) fftshift(fftn(fftshift(a))) / sqrt(numel(a));   % STI fftnc / ifftnc (unitary, centred)
ic = @(a) fftshift(ifftn(fftshift(a))) * sqrt(numel(a));
gamma = 42.575; sfX = 1 / (2*pi*gamma*B0*TE*1e-3);   % rad -> ppm (STI ScalingFactor)
% ---- Laplacian weights from |k^2 * phase| (unmasked phase), ranked inside the mask ----
% k^2 has 1e6 at DC, which adds 1e6*mean(P) to every voxel: with |.| this decides which tail of
% the Laplacian is down-weighted (the one with the sign of mean(P)). Kept for parity.
L = abs(real(ic(kk .* fc(P))));
t = sort(L(m)); Nm = numel(t);
lo = t(round(0.6 * Nm)); hi = t(round(0.999 * Nm));   % STI's percentile: sorted(round(p*N))
W = 1 - 0.8 * (L - lo) / (hi - lo); W = min(max(W, 0.2), 1); W(~m) = 0;
% ---- step 1: LSQR on D W D x = D W phi (tol 1e-2, maxit) ----
% STI works on ifftshift-ed images (origin at index 1) and evaluates the operator in single
% precision, the rhs in double; mirroring both makes this step bitwise identical to STI.
Du = ifftshift(D); Dus = prc(Du); Ws = prc(ifftshift(W));
A1 = @(x, tflag) reshape(ifftn(Dus .* fftn(Ws .* ifftn(Dus .* fftn(reshape(x, n))))), [], 1);
b1 = prc(reshape(ifftn(Du .* fftn(ifftshift(W) .* ifftshift(P))), [], 1));
[x1, info.flag1, info.relres1, info.iter1] = lsqr(A1, b1, tol1, maxit);
info.lapl_lo = lo; info.lapl_hi = hi; info.nmask = Nm;
x1 = fftshift(reshape(double(x1), n));   % complex when H is oblique (unpaired Nyquist planes)
% ---- step 2a: FastQSM (gradient-weight prior; only its ranking matters here) ----
xfs = fastqsm(P, M, D, fc, ic);
g = cell(1, 3); for d = 1:3, g{d} = fd(xfs, d); end
pool = sort(abs([g{1}(m); g{2}(m); g{3}(m)])); Np = numel(pool);
t50 = pool(round(0.5 * Np)); t70 = pool(round(0.7 * Np));
Wg = cell(1, 3); for d = 1:3, Wg{d} = min(max((t70 - abs(g{d})) / (t70 - t50), 0), 1) .* M; end
% ---- step 2b: streaking artefacts on the ill-conditioned cone |D| < 0.1 (tol 1e-3, maxit) ----
Mic = ifftshift(abs(D) < 0.1);
x1m = ifftshift(real(x1) .* M); Wgu = cellfun(@ifftshift, Wg, 'uni', 0);
b2 = prc([reshape(Wgu{1} .* fd(x1m, 1), [], 1); reshape(Wgu{2} .* fd(x1m, 2), [], 1); reshape(Wgu{3} .* fd(x1m, 3), [], 1)]);
Wgs = cellfun(prc, Wgu, 'uni', 0);
A2 = @(x, tflag) step2op(x, tflag, Wgs, prc(Mic), n);
[x2, info.flag2, info.relres2, info.iter2] = lsqr(A2, b2, 1e-3, maxit);
xsa = fftshift(real(ifftn(Mic .* reshape(double(x2), n)) * sqrt(Nt)));
% ---- step 3: subtract, scale to ppm, crop back ----
X = real(x1 - xsa) * sfX .* M; X1 = real(x1) * sfX .* M;
ct = @(A) A(pd(1)+1:pd(1)+c(1), pd(2)+1:pd(2)+c(2), pd(3)+1:pd(3)+c(3));
chi = zeros(N0); chi_lsqr = zeros(N0);
chi(bb(1,1):bb(1,2), bb(2,1):bb(2,2), bb(3,1):bb(3,2)) = ct(X);
chi_lsqr(bb(1,1):bb(1,2), bb(2,1):bb(2,2), bb(3,1):bb(3,2)) = ct(X1);
info.bbox = bb; info.pad = pd; info.n = n;
if nargout > 2, info.W = W; info.b1 = b1; info.x1 = x1; info.Wg = Wg; info.b2 = b2; info.x2 = x2; info.xfs = xfs; end
end

function y = step2op(x, tflag, Wg, Mic, n)
s = sqrt(prod(n));                                   % in the class of Wg (single, as STI)
if strcmp(tflag, 'notransp')
    u = ifftn(Mic .* reshape(x, n)) * s;
    y = [reshape(fd(u, 1) .* Wg{1}, [], 1); reshape(fd(u, 2) .* Wg{2}, [], 1); reshape(fd(u, 3) .* Wg{3}, [], 1)];
else
    N = prod(n); u = zeros(n, 'like', Wg{1});
    for d = 1:3, u = u + fdT(reshape(x((d-1)*N+1:d*N), n) .* Wg{d}, d); end
    y = reshape(Mic .* fftn(u) / s, [], 1);
end
end

function xfs = fastqsm(P, M, D, fc, ic)
% STI FastQSM (as called by QSM_iLSQR): sign(D) inverse, blended with a k-space-smoothed copy
% where |D| is small, masked, repeated once. The TKD-regression rescaling STI applies at the end
% is a positive scalar, irrelevant to the percentile-based weights, so it is left out.
n = size(P);
b = ball3d_sti(3); b = b / sum(b(:));
b = imclose_line(b);                                 % STI: imclose(ball/sum, [1 1 1]); not renormalised
K = zeros(n); cc = floor(n/2) + 1; h = (size(b) - 1) / 2;
K(cc(1)-h(1):cc(1)+h(1), cc(2)-h(2):cc(2)+h(2), cc(3)-h(3):cc(3)+h(3)) = b;
K = real(fftshift(fftn(fftshift(K))));               % transfer function (non-unitary)
sm = @(Xk) ic(K .* fc(Xk));                         % smoothing *of k-space* (k-index convolution)
z = abs(D).^0.001; s = sort(z(:)); Nt = numel(s);
lo = s(round(0.01 * Nt)); hi = s(round(0.3 * Nt));
w = min(max((z - lo) / (hi - lo), 0), 1);
X0 = sign(D) .* fc(P);
X1 = w .* X0 + (1 - w) .* sm(X0);
X1 = fc(M .* ic(X1));
X2 = w .* X1 + (1 - w) .* sm(X1);
xfs = real(ic(X2)) .* M;
end

function b = ball3d_sti(r)
% STI ball3D(r): {|x| <= r + 1/6} in voxels, centre excluded
[x, y, z] = ndgrid(-r:r); b = double(x.^2 + y.^2 + z.^2 <= (r + 1/6)^2); b(r+1, r+1, r+1) = 0;
end

function B = imclose_line(A)
% greyscale closing with the 1x3 structuring element [1 1 1] (along dim 2), IPT semantics
% (out-of-array = -Inf for dilation, +Inf for erosion)
p = -inf(size(A) + [0 2 0]); p(:, 2:end-1, :) = A;
Dl = max(max(p(:, 1:end-2, :), p(:, 2:end-1, :)), p(:, 3:end, :));
q = inf(size(A) + [0 2 0]); q(:, 2:end-1, :) = Dl;
B = min(min(q(:, 1:end-2, :), q(:, 2:end-1, :)), q(:, 3:end, :));
end

function g = fd(u, d)
g = circshift(u, -1, d) - u;                         % forward difference, no voxel-size scaling
end

function g = fdT(y, d)
g = circshift(y, 1, d) - y;                          % adjoint of fd
end

function bb = bbox_even(mask)
% STI MaskBoundingBox: tight bbox; an odd extent grows by one at the high end (or at the low end
% if the high end is the array edge; if both are edges it shrinks by one at the high end)
n = size(mask); n(end+1:3) = 1; bb = zeros(3, 2);
for d = 1:3
    o = setdiff(1:3, d); a = squeeze(any(any(permute(mask > 0, [d o]), 2), 3));
    lo = find(a, 1, 'first'); hi = find(a, 1, 'last');
    if mod(hi - lo + 1, 2) == 1
        if hi < n(d), hi = hi + 1; elseif lo > 1, lo = lo - 1; else, hi = hi - 1; end
    end
    bb(d, :) = [lo hi];
end
end
