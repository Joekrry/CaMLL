#include "test.h"
#include "src/dataset.h"
#include "src/matrix.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);

    /* label in the last column */
    Dataset d = dataset_from_csv(&a, "1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n", 2);
    CHECK(d.X.rows == 3);
    CHECK(d.X.cols == 2);
    CHECK_NEAR(mat_get(&d.X, 0, 0), 1.0, 1e-12);
    CHECK_NEAR(mat_get(&d.X, 2, 1), 6.0, 1e-12);
    CHECK_NEAR(d.y.data[1], 1.0, 1e-12);
    CHECK_NEAR(d.y.data[2], 0.0, 1e-12);

    /* label in the middle, trailing blank line skipped */
    Dataset d2 = dataset_from_csv(&a, "10,1,20\n30,0,40\n\n", 1);
    CHECK(d2.X.rows == 2);
    CHECK_NEAR(mat_get(&d2.X, 0, 1), 20.0, 1e-12);
    CHECK_NEAR(d2.y.data[0], 1.0, 1e-12);
    CHECK_NEAR(d2.y.data[1], 0.0, 1e-12);

    /* split: y = 100*x0 + x1 lets us confirm rows stay paired after shuffling */
    usize n = 8;
    Dataset big;
    big.X = mat_alloc(&a, n, 2);
    big.y = vec_alloc(&a, n);
    for (usize i = 0; i < n; i++) {
        f64 x0 = (f64)i, x1 = (f64)(i * 2);
        mat_set(&big.X, i, 0, x0);
        mat_set(&big.X, i, 1, x1);
        big.y.data[i] = 100.0 * x0 + x1;
    }
    Rng rng = rng_seed(1234);
    Split sp = dataset_split(&a, &big, 0.75, &rng);
    CHECK(sp.train.X.rows == 6);
    CHECK(sp.test.X.rows == 2);
    for (usize i = 0; i < sp.train.X.rows; i++) {
        f64 inv = 100.0 * mat_get(&sp.train.X, i, 0) + mat_get(&sp.train.X, i, 1);
        CHECK_NEAR(sp.train.y.data[i], inv, 1e-9);
    }
    for (usize i = 0; i < sp.test.X.rows; i++) {
        f64 inv = 100.0 * mat_get(&sp.test.X, i, 0) + mat_get(&sp.test.X, i, 1);
        CHECK_NEAR(sp.test.y.data[i], inv, 1e-9);
    }

    /* standardize: col0 varies, col1 is constant and must stay finite */
    Mat m = mat_alloc(&a, 3, 2);
    f64 col0[3] = {1.0, 2.0, 3.0};
    for (usize i = 0; i < 3; i++) {
        mat_set(&m, i, 0, col0[i]);
        mat_set(&m, i, 1, 10.0);
    }
    Standardizer st = standardizer_fit(&a, &m);
    CHECK_NEAR(st.mean.data[0], 2.0, 1e-12);
    CHECK_NEAR(st.std.data[1], 1.0, 1e-12);
    Mat z = standardizer_apply(&a, &st, &m);
    f64 sd0 = sqrt(2.0 / 3.0);
    CHECK_NEAR(mat_get(&z, 0, 0), (1.0 - 2.0) / sd0, 1e-9);
    CHECK_NEAR(mat_get(&z, 1, 0), 0.0, 1e-12);
    CHECK_NEAR(mat_get(&z, 0, 1), 0.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("dataset");
}
