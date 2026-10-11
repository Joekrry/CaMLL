#include "test.h"
#include "src/linreg.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);

    /* single feature, exact line: y = 2x + 1 */
    usize n = 5;
    Mat X = mat_alloc(&a, n, 1);
    Vec y = vec_alloc(&a, n);
    for (usize i = 0; i < n; i++) {
        f64 xi = (f64)i;
        mat_set(&X, i, 0, xi);
        y.data[i] = 2.0 * xi + 1.0;
    }

    LinReg m = linreg_fit_normal(&a, &X, &y);
    CHECK_NEAR(m.bias, 1.0, 1e-9);
    CHECK_NEAR(m.weights.data[0], 2.0, 1e-9);

    Vec pred = linreg_predict(&a, &m, &X);
    CHECK_NEAR(metrics_mse(&y, &pred), 0.0, 1e-12);

    /* two features, exact plane: y = 3*x0 - x1 + 2 */
    usize n2 = 6;
    Mat X2 = mat_alloc(&a, n2, 2);
    Vec y2 = vec_alloc(&a, n2);
    for (usize i = 0; i < n2; i++) {
        f64 x0 = (f64)i;
        f64 x1 = (f64)(i % 3);
        mat_set(&X2, i, 0, x0);
        mat_set(&X2, i, 1, x1);
        y2.data[i] = 3.0 * x0 - 1.0 * x1 + 2.0;
    }

    LinReg m2 = linreg_fit_normal(&a, &X2, &y2);
    CHECK_NEAR(m2.bias, 2.0, 1e-8);
    CHECK_NEAR(m2.weights.data[0], 3.0, 1e-8);
    CHECK_NEAR(m2.weights.data[1], -1.0, 1e-8);

    /* gradient descent on the single-feature case should converge close */
    LinReg gd = linreg_fit_gd(&a, &X, &y, 0.05, 2000);
    CHECK_NEAR(gd.bias, 1.0, 0.05);
    CHECK_NEAR(gd.weights.data[0], 2.0, 0.05);

    arena_destroy(&a);
    return TEST_SUMMARY("linreg");
}
