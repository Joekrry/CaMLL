#include "test.h"
#include "src/tree.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"

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

    Tree gini = tree_fit_classify(&a, &X, &y, 2, TREE_GINI, 4, 2, 0, NULL);
    CHECK(!gini.root->is_leaf);
    Vec pred_gini = tree_predict(&a, &gini, &X);
    CHECK_NEAR(metrics_accuracy(&y, &pred_gini), 1.0, 1e-12);

    Tree entropy = tree_fit_classify(&a, &X, &y, 2, TREE_ENTROPY, 4, 2, 0, NULL);
    Vec pred_entropy = tree_predict(&a, &entropy, &X);
    CHECK_NEAR(metrics_accuracy(&y, &pred_entropy), 1.0, 1e-12);

    Tree stump = tree_fit_classify(&a, &X, &y, 2, TREE_GINI, 0, 2, 0, NULL);
    CHECK(stump.root->is_leaf);
    Vec pred_stump = tree_predict(&a, &stump, &X);
    for (usize i = 0; i < n; i++) CHECK(pred_stump.data[i] == pred_stump.data[0]);

    Mat Xp = mat_alloc(&a, 4, 1);
    Vec yp = vec_alloc(&a, 4);
    for (usize i = 0; i < 4; i++) {
        mat_set(&Xp, i, 0, (f64)i);
        yp.data[i] = 1.0;
    }
    Tree pure = tree_fit_classify(&a, &Xp, &yp, 2, TREE_GINI, 4, 2, 0, NULL);
    CHECK(pure.root->is_leaf);
    CHECK_NEAR(pure.root->value, 1.0, 1e-12);

    usize nk = 9;
    usize k = 3;
    Mat Xk = mat_alloc(&a, nk, 1);
    Vec yk = vec_alloc(&a, nk);
    f64 xs[9] = {-3.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0};
    for (usize i = 0; i < nk; i++) {
        mat_set(&Xk, i, 0, xs[i]);
        yk.data[i] = (f64)(i / 3);
    }
    Tree treek = tree_fit_classify(&a, &Xk, &yk, k, TREE_ENTROPY, 4, 2, 0, NULL);
    Vec predk = tree_predict(&a, &treek, &Xk);
    CHECK_NEAR(metrics_accuracy(&yk, &predk), 1.0, 1e-12);

    usize nr = 6;
    Mat Xr = mat_alloc(&a, nr, 1);
    Vec yr = vec_alloc(&a, nr);
    f64 rxs[6] = {0.0, 1.0, 2.0, 10.0, 11.0, 12.0};
    f64 rys[6] = {1.0, 1.0, 1.0, 5.0, 5.0, 5.0};
    for (usize i = 0; i < nr; i++) {
        mat_set(&Xr, i, 0, rxs[i]);
        yr.data[i] = rys[i];
    }
    Tree reg = tree_fit_regress(&a, &Xr, &yr, 4, 2, 0, NULL);
    CHECK(!reg.root->is_leaf);
    Vec predr = tree_predict(&a, &reg, &Xr);
    for (usize i = 0; i < nr; i++) CHECK_NEAR(predr.data[i], rys[i], 1e-12);
    CHECK_NEAR(metrics_mse(&yr, &predr), 0.0, 1e-12);

    Tree reg_stump = tree_fit_regress(&a, &Xr, &yr, 4, nr + 1, 0, NULL);
    CHECK(reg_stump.root->is_leaf);
    CHECK_NEAR(reg_stump.root->value, 3.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("tree");
}
