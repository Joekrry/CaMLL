#include "test.h"
#include "src/metrics.h"
#include "src/matrix.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);

    /* mse: one prediction off by 2, rest exact */
    usize n = 4;
    Vec y_true = vec_alloc(&a, n);
    Vec y_pred = vec_alloc(&a, n);
    f64 truev[4] = {1.0, 2.0, 3.0, 4.0};
    f64 predv[4] = {1.0, 2.0, 3.0, 6.0};
    for (usize i = 0; i < n; i++) {
        y_true.data[i] = truev[i];
        y_pred.data[i] = predv[i];
    }
    CHECK_NEAR(metrics_mse(&y_true, &y_pred), 1.0, 1e-12);

    /* bce: confident correct predictions give near-zero loss */
    Vec yb = vec_alloc(&a, 2);
    Vec pb = vec_alloc(&a, 2);
    yb.data[0] = 1.0;
    yb.data[1] = 0.0;
    pb.data[0] = 0.999;
    pb.data[1] = 0.001;
    CHECK(metrics_bce(&yb, &pb) < 0.01);

    /* bce: confident wrong predictions give large loss */
    Vec pw = vec_alloc(&a, 2);
    pw.data[0] = 0.001;
    pw.data[1] = 0.999;
    CHECK(metrics_bce(&yb, &pw) > 5.0);

    /* accuracy and confusion matrix on a mixed set */
    usize m = 6;
    Vec yt = vec_alloc(&a, m);
    Vec yp = vec_alloc(&a, m);
    f64 truth[6] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    f64 predicted[6] = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    for (usize i = 0; i < m; i++) {
        yt.data[i] = truth[i];
        yp.data[i] = predicted[i];
    }
    CHECK_NEAR(metrics_accuracy(&yt, &yp), 4.0 / 6.0, 1e-12);

    ConfusionMatrix cm = metrics_confusion(&yt, &yp);
    CHECK(cm.tp == 2);
    CHECK(cm.fp == 1);
    CHECK(cm.tn == 2);
    CHECK(cm.fn == 1);

    CHECK_NEAR(metrics_precision(&cm), 2.0 / 3.0, 1e-12);
    CHECK_NEAR(metrics_recall(&cm), 2.0 / 3.0, 1e-12);
    CHECK_NEAR(metrics_f1(&cm), 2.0 / 3.0, 1e-12);

    /* degenerate: no predicted or actual positives at all */
    ConfusionMatrix cm_none = {0, 0, 4, 0};
    CHECK_NEAR(metrics_precision(&cm_none), 0.0, 1e-12);
    CHECK_NEAR(metrics_recall(&cm_none), 0.0, 1e-12);
    CHECK_NEAR(metrics_f1(&cm_none), 0.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("metrics");
}
