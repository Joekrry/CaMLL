#include "test.h"
#include "src/knn.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"

#include <math.h>

int main(void) {
    Arena a = arena_create(0);

    /* Two separated 1-D clusters: label 0 near x=0, label 1 near x=10. */
    usize n = 8;
    Mat Xc = mat_alloc(&a, n, 1);
    Vec yc = vec_alloc(&a, n);
    f64 xs[8] = {-1.0, -0.5, 0.0, 0.5, 9.5, 10.0, 10.5, 11.0};
    for (usize i = 0; i < n; i++) {
        mat_set(&Xc, i, 0, xs[i]);
        yc.data[i] = xs[i] < 5.0 ? 0.0 : 1.0;
    }

    KNN knn = knn_fit(&a, &Xc, &yc, 3);
    CHECK(knn.X.rows == n);
    CHECK(knn.k == 3);

    Vec pred = knn_predict_classify(&a, &knn, &Xc);
    CHECK(pred.len == n);
    CHECK_NEAR(metrics_accuracy(&yc, &pred), 1.0, 1e-12);

    /* Query points land exactly on training points. */
    Mat Xq = mat_alloc(&a, 2, 1);
    mat_set(&Xq, 0, 0, -0.5);
    mat_set(&Xq, 1, 0, 10.5);
    KNN knn1 = knn_fit(&a, &Xc, &yc, 1);
    Vec pred1 = knn_predict_classify(&a, &knn1, &Xq);
    CHECK_NEAR(pred1.data[0], 0.0, 1e-12);
    CHECK_NEAR(pred1.data[1], 1.0, 1e-12);

    /* y = 2x on a small grid. */
    usize nr = 6;
    Mat Xr = mat_alloc(&a, nr, 1);
    Vec yr = vec_alloc(&a, nr);
    for (usize i = 0; i < nr; i++) {
        f64 xi = (f64)i;
        mat_set(&Xr, i, 0, xi);
        yr.data[i] = 2.0 * xi;
    }

    KNN knnr1 = knn_fit(&a, &Xr, &yr, 1);
    Vec predr1 = knn_predict_regress(&a, &knnr1, &Xr);
    CHECK_NEAR(metrics_mse(&yr, &predr1), 0.0, 1e-12);

    /* Midpoint query, k=2: averages its two nearest neighbors. */
    Mat Xmid = mat_alloc(&a, 1, 1);
    mat_set(&Xmid, 0, 0, 1.5);
    KNN knnr2 = knn_fit(&a, &Xr, &yr, 2);
    Vec predr2 = knn_predict_regress(&a, &knnr2, &Xmid);
    CHECK_NEAR(predr2.data[0], 3.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("knn");
}
