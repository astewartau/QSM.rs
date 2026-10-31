#!/usr/bin/env bash
# Regenerates vnl_inverse_3x3.txt: VNL's 3x3 inverse (vnl_svd(M).inverse(), LINPACK dsvdc), which
# ITK 5.3's ImageBase takes of direction * diag(spacing) to map physical points to indices, and which
# src/utils/itk_geometry.rs ports. Builds the v3p netlib C sources of ITK v5.3.0 (dsvdc and its
# BLAS level 1) with gcc -O2 on x86-64 and runs them on fixed matrices.
#   bash tests/data/vnl_inverse_3x3.sh > tests/data/vnl_inverse_3x3.txt
set -euo pipefail
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
B=https://raw.githubusercontent.com/InsightSoftwareConsortium/ITK/v5.3.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib
for f in linpack/dsvdc blas/dnrm2 blas/drotg blas/drot blas/ddot blas/daxpy blas/dscal blas/dswap; do
  curl -fsSL -o "$T/$(basename $f).c" "$B/$f.c"
done
# The f2c definitions the sources need (v3p_netlib.h, reduced).
cat > "$T/v3p_netlib.h" <<'H'
typedef long integer; typedef double doublereal; typedef long logical;
#define TRUE_ (1)
#define FALSE_ (0)
#define abs(x) ((x) >= 0 ? (x) : -(x))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#include <math.h>
static double d_sign(doublereal *a, doublereal *b){ double x = (*a >= 0 ? *a : - *a); return( *b >= 0 ? x : -x);}
#define v3p_netlib_dsvdc_ dsvdc_
#define v3p_netlib_dnrm2_ dnrm2_
#define v3p_netlib_drotg_ drotg_
#define v3p_netlib_drot_ drot_
#define v3p_netlib_ddot_ ddot_
#define v3p_netlib_daxpy_ daxpy_
#define v3p_netlib_dscal_ dscal_
#define v3p_netlib_dswap_ dswap_
H
# vnl_svd<double>(M) (job 21) then pinverse = V * W^-1 * U^T, products as vnl_matrix computes them.
cat > "$T/main.c" <<'C'
#include <stdio.h>
#include "v3p_netlib.h"
int dsvdc_(doublereal*, integer*, integer*, integer*, doublereal*, doublereal*, doublereal*, integer*,
           doublereal*, integer*, doublereal*, integer*, integer*);
int main(void) {
  unsigned long long b[9];
  while (scanf("%llx %llx %llx %llx %llx %llx %llx %llx %llx", b, b+1, b+2, b+3, b+4, b+5, b+6, b+7, b+8) == 9) {
    double M[3][3]; for (int i = 0; i < 9; i++) ((unsigned long long*)M)[i] = b[i];
    double X[9]; for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) X[i+3*j] = M[i][j];
    integer n = 3, p = 3, job = 21, info = 0; double s[4] = {0}, e[3] = {0}, u[9] = {0}, v[9] = {0}, w[3] = {0};
    dsvdc_(X, &n, &n, &p, s, e, u, &n, v, &p, w, &job, &info);
    double U[3][3], V[3][3], Wi[3][3] = {{0}}, Ut[3][3], VW[3][3], R[3][3];
    for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) { U[i][j] = u[i+3*j]; V[i][j] = v[i+3*j]; }
    for (int j = 0; j < 3; j++) { double wj = fabs(s[j]); Wi[j][j] = (fabs(wj) <= 0) ? 0 : 1.0 / wj; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ut[i][j] = U[j][i];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double a = V[i][0]*Wi[0][j]; for (int k = 1; k < 3; k++) a += V[i][k]*Wi[k][j]; VW[i][j] = a; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double a = VW[i][0]*Ut[0][j]; for (int k = 1; k < 3; k++) a += VW[i][k]*Ut[k][j]; R[i][j] = a; }
    for (int i = 0; i < 9; i++) printf("%016llx%c", ((unsigned long long*)R)[i], i == 8 ? '\n' : ' ');
  }
  return 0;
}
C
( cd "$T" && gcc -O2 -o vnlinv main.c dsvdc.c dnrm2.c drotg.c drot.c ddot.c daxpy.c dscal.c dswap.c -lm 2>/dev/null )
python3 - > "$T/mats.txt" <<'PY'
import struct, random, math
random.seed(327)
def hx(x): return '%016x' % struct.unpack('<Q', struct.pack('<d', x))[0]
def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
mats = [[[1, 4.89699991e-12, 0], [4.89699991e-12, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]
for k in range(40):
    q = [random.gauss(0, 1) for _ in range(4)]; n = math.sqrt(sum(v*v for v in q)); a, b, c, d = [v/n for v in q]
    if k % 3 == 0:
        b, c, d = b*1e-3, c*1e-3, d*1e-2; n = math.sqrt(a*a+b*b+c*c+d*d); a, b, c, d = a/n, b/n, c/n, d/n
    R = [[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
         [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-c*c-b*b]]
    R = [[f32(v) for v in r] for r in R]
    for j in range(3):
        s = math.sqrt(sum(R[i][j]**2 for i in range(3)))
        for i in range(3): R[i][j] /= s
    sp = [f32(random.choice([1.0, 0.9375, 0.43, 1.2, 2.0, 0.6, 0.8571428, 3.0])) for _ in range(3)]
    mats.append([[R[i][j]*sp[j] for j in range(3)] for i in range(3)])
for k in range(10):
    mats.append([[random.uniform(-2, 2) for j in range(3)] for i in range(3)])
for m in mats: print(' '.join(hx(float(m[i][j])) for i in range(3) for j in range(3)))
PY
"$T/vnlinv" < "$T/mats.txt" > "$T/inv.txt"
echo "# VNL's 3x3 inverse (itk::Matrix::GetInverse = vnl_svd(M).inverse(), LINPACK dsvdc) from ITK v5.3.0's"
echo "# v3p netlib C sources, built with gcc -O2 on x86-64 by vnl_inverse_3x3.sh. Each line: M (row-major) | inverse,"
echo "# 9 + 9 IEEE-754 doubles as hex bits. Matrices: the in-vivo Prisma direction (4.9e-12 off-diagonal), make_nii's"
echo "# diag(-1,-1,1), 40 float32-rounded rotations times voxel sizes, 10 random."
paste -d'|' "$T/mats.txt" "$T/inv.txt"
