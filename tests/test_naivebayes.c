#include "test.h"
#include "src/naivebayes.h"
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

    /* Two 2-D "plus" clusters, centered at (0,0) and (10,10). */
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

    NaiveBayes nb = naivebayes_fit(&a, &X, &y, 2, 1e-6);
    CHECK(nb.num_classes == 2);
    CHECK_NEAR(mat_get(&nb.mean, 0, 0), 0.0, 1e-12);
    CHECK_NEAR(mat_get(&nb.mean, 1, 0), 10.0, 1e-12);

    Vec pred = naivebayes_predict(&a, &nb, &X);
    CHECK(pred.len == n);
    CHECK_NEAR(metrics_accuracy(&y, &pred), 1.0, 1e-12);

    /* Three classes on a line, same setup as test_softmax.c. */
    usize nk = 9;
    usize k = 3;
    Mat Xk = mat_alloc(&a, nk, 1);
    Vec yk = vec_alloc(&a, nk);
    f64 xs[9] = {-3.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0};
    for (usize i = 0; i < nk; i++) {
        mat_set(&Xk, i, 0, xs[i]);
        yk.data[i] = (f64)(i / 3);
    }

    NaiveBayes nbk = naivebayes_fit(&a, &Xk, &yk, k, 1e-9);
    Mat ll = naivebayes_predict_log_proba(&a, &nbk, &Xk);
    CHECK(ll.rows == nk);
    CHECK(ll.cols == k);

    Vec predk = naivebayes_predict(&a, &nbk, &Xk);
    CHECK_NEAR(metrics_accuracy(&yk, &predk), 1.0, 1e-12);

    /* Feature constant within each class: raw variance is 0, var_smoothing keeps the density finite. */
    Mat Xc = mat_alloc(&a, 6, 1);
    Vec yc = vec_alloc(&a, 6);
    for (usize i = 0; i < 3; i++) {
        mat_set(&Xc, i, 0, 5.0);
        yc.data[i] = 0.0;
        mat_set(&Xc, i + 3, 0, -5.0);
        yc.data[i + 3] = 1.0;
    }
    NaiveBayes nbc = naivebayes_fit(&a, &Xc, &yc, 2, 1e-6);
    CHECK_NEAR(mat_get(&nbc.var, 0, 0), 1e-6, 1e-15);
    CHECK_NEAR(mat_get(&nbc.var, 1, 0), 1e-6, 1e-15);

    Mat llc = naivebayes_predict_log_proba(&a, &nbc, &Xc);
    for (usize i = 0; i < llc.rows * llc.cols; i++) {
        CHECK(isfinite(llc.data[i]));
    }
    Vec predc = naivebayes_predict(&a, &nbc, &Xc);
    CHECK_NEAR(metrics_accuracy(&yc, &predc), 1.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("naivebayes");
}
