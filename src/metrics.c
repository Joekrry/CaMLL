#include "metrics.h"

#include <assert.h>
#include <math.h>

f64 metrics_mse(const Vec *y_true, const Vec *y_pred) {
    assert(y_true->len == y_pred->len);
    f64 s = 0.0;
    for (usize i = 0; i < y_true->len; i++) {
        f64 d = y_pred->data[i] - y_true->data[i];
        s += d * d;
    }
    return s / (f64)y_true->len;
}

f64 metrics_bce(const Vec *y_true, const Vec *y_proba) {
    assert(y_true->len == y_proba->len);
    const f64 eps = 1e-12;
    f64 s = 0.0;
    for (usize i = 0; i < y_true->len; i++) {
        f64 p = y_proba->data[i];
        if (p < eps) p = eps;
        if (p > 1.0 - eps) p = 1.0 - eps;
        s -= y_true->data[i] * log(p) + (1.0 - y_true->data[i]) * log(1.0 - p);
    }
    return s / (f64)y_true->len;
}

f64 metrics_accuracy(const Vec *y_true, const Vec *y_pred) {
    assert(y_true->len == y_pred->len);
    usize correct = 0;
    for (usize i = 0; i < y_true->len; i++) {
        if (y_pred->data[i] == y_true->data[i]) correct++;
    }
    return (f64)correct / (f64)y_true->len;
}

ConfusionMatrix metrics_confusion(const Vec *y_true, const Vec *y_pred) {
    assert(y_true->len == y_pred->len);
    ConfusionMatrix cm = {0, 0, 0, 0};
    for (usize i = 0; i < y_true->len; i++) {
        int actual_pos = y_true->data[i] == 1.0;
        int pred_pos = y_pred->data[i] == 1.0;
        if (actual_pos && pred_pos) {
            cm.tp++;
        } else if (!actual_pos && pred_pos) {
            cm.fp++;
        } else if (!actual_pos && !pred_pos) {
            cm.tn++;
        } else {
            cm.fn++;
        }
    }
    return cm;
}

f64 metrics_precision(const ConfusionMatrix *cm) {
    usize denom = cm->tp + cm->fp;
    return denom == 0 ? 0.0 : (f64)cm->tp / (f64)denom;
}

f64 metrics_recall(const ConfusionMatrix *cm) {
    usize denom = cm->tp + cm->fn;
    return denom == 0 ? 0.0 : (f64)cm->tp / (f64)denom;
}

f64 metrics_f1(const ConfusionMatrix *cm) {
    f64 p = metrics_precision(cm);
    f64 r = metrics_recall(cm);
    return (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
}
