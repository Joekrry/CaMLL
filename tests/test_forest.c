#include "test.h"
#include "src/forest.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"
#include <math.h>

static void set_row(Mat *m, usize r, f64 x, f64 y) {
    mat_set(m, r, 0, x);
    mat_set(m, r, 1, y);
}

int main(void) {
    Arena a = arena_create(0);

    usize n = 8;
    Mat X = mat_alloc(&a, n, 2);
    Vec y = vec_alloc(&a, n);
    set_row(&X, 0, -1.0, 0.0);
    set_row(&X, 1, 1.0, 0.0);
    set_row(&X, 2, 0.0, -1.0);
    set_row(&X, 3, 0.0, 1.0);
    set_row(&X, 4, 9.0, 10.0);
    set_row(&X, 5, 11.0, 10.0);
    set_row(&X, 6, 10.0, 9.0);
    set_row(&X, 7, 10.0, 11.0);
    for (usize i = 0; i < n; i++) y.data[i] = i < 4 ? 0.0 : 1.0;

    Rng rng1 = rng_seed(1);
    Forest forest = forest_fit_classify(&a, &X, &y, 2, TREE_GINI, 10, 4, 2, 0, &rng1);
    CHECK(forest.num_trees == 10);
    Vec pred = forest_predict(&a, &forest, &X);
    CHECK_NEAR(metrics_accuracy(&y, &pred), 1.0, 1e-12);

    Rng rng2 = rng_seed(2);
    Forest forest_sub = forest_fit_classify(&a, &X, &y, 2, TREE_GINI, 10, 4, 2, 1, &rng2);
    Vec pred_sub = forest_predict(&a, &forest_sub, &X);
    CHECK_NEAR(metrics_accuracy(&y, &pred_sub), 1.0, 1e-12);

    Rng rng3 = rng_seed(3);
    Forest one_tree = forest_fit_classify(&a, &X, &y, 2, TREE_ENTROPY, 1, 4, 2, 0, &rng3);
    CHECK(one_tree.num_trees == 1);

    usize nk = 9;
    usize k = 3;
    Mat Xk = mat_alloc(&a, nk, 1);
    Vec yk = vec_alloc(&a, nk);
    f64 xs[9] = {-3.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0};
    for (usize i = 0; i < nk; i++) {
        mat_set(&Xk, i, 0, xs[i]);
        yk.data[i] = (f64)(i / 3);
    }
    Rng rng4 = rng_seed(4);
    Forest forest_k = forest_fit_classify(&a, &Xk, &yk, k, TREE_ENTROPY, 15, 4, 2, 0, &rng4);
    Vec pred_k = forest_predict(&a, &forest_k, &Xk);
    CHECK_NEAR(metrics_accuracy(&yk, &pred_k), 1.0, 1e-12);

    usize nr = 6;
    Mat Xr = mat_alloc(&a, nr, 1);
    Vec yr = vec_alloc(&a, nr);
    f64 rxs[6] = {0.0, 1.0, 2.0, 10.0, 11.0, 12.0};
    f64 rys[6] = {1.0, 1.0, 1.0, 5.0, 5.0, 5.0};
    for (usize i = 0; i < nr; i++) {
        mat_set(&Xr, i, 0, rxs[i]);
        yr.data[i] = rys[i];
    }
    Rng rng5 = rng_seed(5);
    Forest reg = forest_fit_regress(&a, &Xr, &yr, 15, 4, 2, 0, &rng5);
    CHECK(!reg.is_classification);
    Vec predr = forest_predict(&a, &reg, &Xr);
    for (usize i = 0; i < nr; i++) CHECK(fabs(predr.data[i] - rys[i]) < 1.0);

    arena_destroy(&a);
    return TEST_SUMMARY("forest");
}
