#include "test.h"
#include "src/softmax.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"

#include <math.h>

int main(void) {
    Arena a = arena_create(0);

    /* single feature, three classes separable by thresholds on x:
       0 if x < -1, 1 if -1 <= x <= 1, 2 if x > 1 */
    usize n = 9;
    usize k = 3;
    Mat X = mat_alloc(&a, n, 1);
    Vec y = vec_alloc(&a, n);
    f64 xs[9] = {-3.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0};
    for (usize i = 0; i < n; i++) {
        mat_set(&X, i, 0, xs[i]);
        f64 x = xs[i];
        y.data[i] = x < -1.0 ? 0.0 : (x > 1.0 ? 2.0 : 1.0);
    }

    Softmax m = softmax_fit_gd(&a, &X, &y, k, 0.5, 3000, 0.0);
    CHECK(m.weights.rows == k);
    CHECK(m.weights.cols == 1);
    CHECK(m.bias.len == k);

    Mat proba = softmax_predict_proba(&a, &m, &X);
    CHECK(proba.rows == n);
    CHECK(proba.cols == k);
    for (usize i = 0; i < n; i++) {
        f64 sum = 0.0;
        for (usize c = 0; c < k; c++) {
            f64 p = mat_get(&proba, i, c);
            CHECK(p >= 0.0 && p <= 1.0);
            sum += p;
        }
        CHECK_NEAR(sum, 1.0, 1e-9);
    }

    Vec pred = softmax_predict(&a, &m, &X);
    CHECK(pred.len == n);
    CHECK_NEAR(metrics_accuracy(&y, &pred), 1.0, 1e-12);

    /* a strong ridge penalty shrinks the weights toward zero relative to
       the unregularized fit */
    Softmax m_l2 = softmax_fit_gd(&a, &X, &y, k, 0.5, 3000, 1.0);
    f64 norm = 0.0, norm_l2 = 0.0;
    for (usize i = 0; i < m.weights.rows * m.weights.cols; i++) {
        norm += m.weights.data[i] * m.weights.data[i];
        norm_l2 += m_l2.weights.data[i] * m_l2.weights.data[i];
    }
    CHECK(norm_l2 < norm);

    arena_destroy(&a);
    return TEST_SUMMARY("softmax");
}
