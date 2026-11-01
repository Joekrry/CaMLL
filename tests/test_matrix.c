#include "test.h"
#include "src/matrix.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);

    Mat m = mat_zeros(&a, 2, 3);
    CHECK(m.rows == 2);
    CHECK(m.cols == 3);
    CHECK(mat_get(&m, 1, 2) == 0.0);

    mat_set(&m, 1, 2, 4.5);
    CHECK(mat_get(&m, 1, 2) == 4.5);

    Vec v = vec_zeros(&a, 4);
    CHECK(v.len == 4);
    CHECK(v.data[3] == 0.0);

    /* add / sub / scale on 2x2 matrices */
    Mat x = mat_alloc(&a, 2, 2);
    mat_set(&x, 0, 0, 1.0); mat_set(&x, 0, 1, 2.0);
    mat_set(&x, 1, 0, 3.0); mat_set(&x, 1, 1, 4.0);

    Mat y = mat_alloc(&a, 2, 2);
    mat_set(&y, 0, 0, 5.0); mat_set(&y, 0, 1, 6.0);
    mat_set(&y, 1, 0, 7.0); mat_set(&y, 1, 1, 8.0);

    Mat sum = mat_add(&a, &x, &y);
    CHECK(mat_get(&sum, 0, 0) == 6.0);
    CHECK(mat_get(&sum, 1, 1) == 12.0);

    Mat diff = mat_sub(&a, &y, &x);
    CHECK(mat_get(&diff, 0, 1) == 4.0);
    CHECK(mat_get(&diff, 1, 0) == 4.0);

    Mat scaled = mat_scale(&a, &x, 2.0);
    CHECK(mat_get(&scaled, 0, 1) == 4.0);
    CHECK(mat_get(&scaled, 1, 1) == 8.0);

    /* matmul: (2x3)(3x2) = (2x2), plus transpose */
    Mat p = mat_alloc(&a, 2, 3);
    for (usize i = 0; i < 6; i++) p.data[i] = (f64)(i + 1); /* 1..6 */
    Mat q = mat_alloc(&a, 3, 2);
    for (usize i = 0; i < 6; i++) q.data[i] = (f64)(i + 1); /* 1..6 */

    Mat prod = mat_matmul(&a, &p, &q);
    CHECK(prod.rows == 2 && prod.cols == 2);
    CHECK(mat_get(&prod, 0, 0) == 22.0); /* 1*1 + 2*3 + 3*5 */
    CHECK(mat_get(&prod, 0, 1) == 28.0); /* 1*2 + 2*4 + 3*6 */
    CHECK(mat_get(&prod, 1, 0) == 49.0); /* 4*1 + 5*3 + 6*5 */
    CHECK(mat_get(&prod, 1, 1) == 64.0); /* 4*2 + 5*4 + 6*6 */

    Mat pt = mat_transpose(&a, &p);
    CHECK(pt.rows == 3 && pt.cols == 2);
    CHECK(mat_get(&pt, 2, 1) == mat_get(&p, 1, 2));
    CHECK(mat_get(&pt, 0, 1) == mat_get(&p, 1, 0));

    /* dot / norm / elementwise mul / elementwise div */
    Vec u = vec_alloc(&a, 3);
    u.data[0] = 1.0; u.data[1] = -2.0; u.data[2] = 2.0;
    Vec w = vec_alloc(&a, 3);
    w.data[0] = 4.0; w.data[1] = 5.0; w.data[2] = 6.0;

    CHECK(vec_dot(&u, &w) == 6.0); /* 1*4 + -2*5 + 2*6 */
    CHECK(vec_norm1(&u) == 5.0);   /* |1| + |-2| + |2| */
    CHECK(vec_norm2(&u) == 3.0);   /* sqrt(1 + 4 + 4) */

    Vec uw = vec_mul(&a, &u, &w);
    CHECK(uw.data[0] == 4.0 && uw.data[1] == -10.0 && uw.data[2] == 12.0);

    Vec wu = vec_div(&a, &w, &u);
    CHECK(wu.data[0] == 4.0 && wu.data[2] == 3.0);

    /* outer product: 2-vec by 3-vec -> 2x3 */
    Vec ov = vec_alloc(&a, 2);
    ov.data[0] = 2.0; ov.data[1] = 3.0;
    Mat outer = mat_outer(&a, &ov, &w);
    CHECK(outer.rows == 2 && outer.cols == 3);
    CHECK(mat_get(&outer, 0, 0) == 8.0);  /* 2*4 */
    CHECK(mat_get(&outer, 1, 2) == 18.0); /* 3*6 */

    /* ones / eye / copy */
    Vec ones = vec_ones(&a, 3);
    CHECK(ones.data[0] == 1.0 && ones.data[2] == 1.0);

    Mat onesm = mat_ones(&a, 2, 2);
    CHECK(mat_get(&onesm, 0, 0) == 1.0 && mat_get(&onesm, 1, 1) == 1.0);

    Mat eye = mat_eye(&a, 3);
    CHECK(mat_get(&eye, 0, 0) == 1.0 && mat_get(&eye, 1, 1) == 1.0);
    CHECK(mat_get(&eye, 0, 1) == 0.0 && mat_get(&eye, 2, 0) == 0.0);

    Vec ucopy = vec_copy(&a, &u);
    CHECK(ucopy.data[0] == u.data[0] && ucopy.data[1] == u.data[1]);
    ucopy.data[0] = 99.0;
    CHECK(u.data[0] == 1.0); /* copy is independent of source */

    Mat xcopy = mat_copy(&a, &x);
    CHECK(mat_get(&xcopy, 0, 0) == 1.0 && mat_get(&xcopy, 1, 1) == 4.0);

    /* deterministic RNG: same seed gives same stream */
    Rng r1 = rng_seed(42);
    Rng r2 = rng_seed(42);
    f64 a1 = rng_uniform(&r1, 0.0, 1.0);
    f64 a2 = rng_uniform(&r2, 0.0, 1.0);
    CHECK(a1 == a2);

    Vec ru = vec_rand_uniform(&a, 100, &r1, -1.0, 1.0);
    for (usize i = 0; i < ru.len; i++) {
        CHECK(ru.data[i] >= -1.0 && ru.data[i] < 1.0);
    }

    Mat rn = mat_rand_normal(&a, 50, 50, &r1, 0.0, 1.0);
    f64 rnsum = 0.0;
    usize rncount = rn.rows * rn.cols;
    for (usize i = 0; i < rncount; i++) rnsum += rn.data[i];
    CHECK_NEAR(rnsum / (f64)rncount, 0.0, 0.3); /* sample mean near 0 */

    /* eigendecomposition of a diagonal matrix: no rotation needed, so
       eigenvalues come back sorted and eigenvectors are the matching
       standard basis columns */
    Mat diag = mat_zeros(&a, 3, 3);
    mat_set(&diag, 0, 0, 5.0);
    mat_set(&diag, 1, 1, 1.0);
    mat_set(&diag, 2, 2, 3.0);
    Eigen eig_diag = mat_eigen_sym(&a, &diag, 100, 1e-12);
    CHECK_NEAR(eig_diag.values.data[0], 5.0, 1e-9);
    CHECK_NEAR(eig_diag.values.data[1], 3.0, 1e-9);
    CHECK_NEAR(eig_diag.values.data[2], 1.0, 1e-9);
    CHECK_NEAR(mat_get(&eig_diag.vectors, 0, 0), 1.0, 1e-9);
    CHECK_NEAR(mat_get(&eig_diag.vectors, 2, 1), 1.0, 1e-9);
    CHECK_NEAR(mat_get(&eig_diag.vectors, 1, 2), 1.0, 1e-9);

    /* 2x2 case with known eigenvalues 3 and 1; check Av = lambda*v
       directly so eigenvector sign doesn't matter */
    Mat sym2 = mat_alloc(&a, 2, 2);
    mat_set(&sym2, 0, 0, 2.0); mat_set(&sym2, 0, 1, 1.0);
    mat_set(&sym2, 1, 0, 1.0); mat_set(&sym2, 1, 1, 2.0);
    Eigen eig2 = mat_eigen_sym(&a, &sym2, 100, 1e-12);
    CHECK_NEAR(eig2.values.data[0], 3.0, 1e-9);
    CHECK_NEAR(eig2.values.data[1], 1.0, 1e-9);
    for (usize i = 0; i < 2; i++) {
        f64 v0 = mat_get(&eig2.vectors, 0, i);
        f64 v1 = mat_get(&eig2.vectors, 1, i);
        f64 lambda = eig2.values.data[i];
        CHECK_NEAR(mat_get(&sym2, 0, 0) * v0 + mat_get(&sym2, 0, 1) * v1, lambda * v0, 1e-9);
        CHECK_NEAR(mat_get(&sym2, 1, 0) * v0 + mat_get(&sym2, 1, 1) * v1, lambda * v1, 1e-9);
        CHECK_NEAR(v0 * v0 + v1 * v1, 1.0, 1e-9);
    }

    /* general 3x3 case: reconstruct A = V diag(values) V^T and check V is
       orthonormal, with no analytic eigenvalues needed */
    Mat sym3 = mat_alloc(&a, 3, 3);
    mat_set(&sym3, 0, 0, 4.0); mat_set(&sym3, 0, 1, 1.0); mat_set(&sym3, 0, 2, -2.0);
    mat_set(&sym3, 1, 0, 1.0); mat_set(&sym3, 1, 1, 3.0); mat_set(&sym3, 1, 2, 0.0);
    mat_set(&sym3, 2, 0, -2.0); mat_set(&sym3, 2, 1, 0.0); mat_set(&sym3, 2, 2, 2.0);
    Eigen eig3 = mat_eigen_sym(&a, &sym3, 100, 1e-12);

    Mat dmat = mat_zeros(&a, 3, 3);
    for (usize i = 0; i < 3; i++) mat_set(&dmat, i, i, eig3.values.data[i]);
    Mat vt = mat_transpose(&a, &eig3.vectors);
    Mat vd = mat_matmul(&a, &eig3.vectors, &dmat);
    Mat recon = mat_matmul(&a, &vd, &vt);
    for (usize i = 0; i < 3; i++) {
        for (usize j = 0; j < 3; j++) {
            CHECK_NEAR(mat_get(&recon, i, j), mat_get(&sym3, i, j), 1e-9);
        }
    }

    Mat vtv = mat_matmul(&a, &vt, &eig3.vectors);
    Mat id3 = mat_eye(&a, 3);
    for (usize i = 0; i < 3; i++) {
        for (usize j = 0; j < 3; j++) {
            CHECK_NEAR(mat_get(&vtv, i, j), mat_get(&id3, i, j), 1e-9);
        }
    }

    arena_destroy(&a);
    return TEST_SUMMARY("matrix");
}
