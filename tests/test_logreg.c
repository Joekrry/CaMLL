#include "test.h"
#include "src/logreg.h"
#include "src/matrix.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);

    /* single feature, linearly separable: y = 1 iff x > 0 */
    usize n = 6;
    Mat X = mat_alloc(&a, n, 1);
    Vec y = vec_alloc(&a, n);
    f64 xs[6] = {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0};
    for (usize i = 0; i < n; i++) {
        mat_set(&X, i, 0, xs[i]);
        y.data[i] = xs[i] > 0.0 ? 1.0 : 0.0;
    }

    LogReg m = logreg_fit_gd(&a, &X, &y, 0.5, 3000);
    CHECK(m.weights.data[0] > 0.0);

    Vec proba = logreg_predict_proba(&a, &m, &X);
    CHECK(proba.len == n);
    for (usize i = 0; i < n; i++) {
        CHECK(proba.data[i] > 0.0 && proba.data[i] < 1.0);
    }

    Vec pred = logreg_predict(&a, &m, &X);
    CHECK_NEAR(logreg_accuracy(&y, &pred), 1.0, 1e-12);
    CHECK(logreg_bce(&y, &proba) < 0.1);

    /* two features, separable by x0 - x1 > 0 */
    usize n2 = 8;
    Mat X2 = mat_alloc(&a, n2, 2);
    Vec y2 = vec_alloc(&a, n2);
    f64 x0s[8] = {0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0};
    f64 x1s[8] = {2.0, 3.0, 4.0, 5.0, -2.0, -1.0, 0.0, 1.0};
    for (usize i = 0; i < n2; i++) {
        mat_set(&X2, i, 0, x0s[i]);
        mat_set(&X2, i, 1, x1s[i]);
        y2.data[i] = (x0s[i] - x1s[i]) > 0.0 ? 1.0 : 0.0;
    }

    LogReg m2 = logreg_fit_gd(&a, &X2, &y2, 0.5, 3000);
    CHECK(m2.weights.data[0] > 0.0);
    CHECK(m2.weights.data[1] < 0.0);

    Vec pred2 = logreg_predict(&a, &m2, &X2);
    CHECK_NEAR(logreg_accuracy(&y2, &pred2), 1.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("logreg");
}
