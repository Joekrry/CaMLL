#include "naivebayes.h"

#include <assert.h>
#include <math.h>

#define CAMLL_LOG_TWO_PI 1.8378770664093454835606594728112

NaiveBayes naivebayes_fit(Arena *a, const Mat *X, const Vec *y, usize num_classes, f64 var_smoothing) {
    assert(X->rows == y->len);
    usize n = X->rows;
    usize d = X->cols;
    usize k = num_classes;

    NaiveBayes model;
    model.num_classes = k;
    model.mean = mat_zeros(a, k, d);
    model.var = mat_zeros(a, k, d);
    model.log_prior = vec_zeros(a, k);

    usize *count = (usize *)arena_alloc(a, k * sizeof(usize));
    for (usize c = 0; c < k; c++) count[c] = 0;

    for (usize i = 0; i < n; i++) {
        usize c = (usize)y->data[i];
        count[c]++;
        const f64 *xrow = mat_row(X, i);
        f64 *mrow = mat_row(&model.mean, c);
        for (usize j = 0; j < d; j++) mrow[j] += xrow[j];
    }
    for (usize c = 0; c < k; c++) {
        f64 *mrow = mat_row(&model.mean, c);
        for (usize j = 0; j < d; j++) mrow[j] /= (f64)count[c];
    }

    for (usize i = 0; i < n; i++) {
        usize c = (usize)y->data[i];
        const f64 *xrow = mat_row(X, i);
        const f64 *mrow = mat_row(&model.mean, c);
        f64 *vrow = mat_row(&model.var, c);
        for (usize j = 0; j < d; j++) {
            f64 diff = xrow[j] - mrow[j];
            vrow[j] += diff * diff;
        }
    }
    for (usize c = 0; c < k; c++) {
        f64 *vrow = mat_row(&model.var, c);
        for (usize j = 0; j < d; j++) vrow[j] = vrow[j] / (f64)count[c] + var_smoothing;
        model.log_prior.data[c] = log((f64)count[c] / (f64)n);
    }
    return model;
}

Mat naivebayes_predict_log_proba(Arena *a, const NaiveBayes *model, const Mat *X) {
    assert(X->cols == model->mean.cols);
    usize n = X->rows;
    usize d = X->cols;
    usize k = model->num_classes;

    Mat ll = mat_alloc(a, n, k);
    for (usize i = 0; i < n; i++) {
        const f64 *xrow = mat_row(X, i);
        f64 *lrow = mat_row(&ll, i);
        for (usize c = 0; c < k; c++) {
            const f64 *mrow = mat_row(&model->mean, c);
            const f64 *vrow = mat_row(&model->var, c);
            f64 s = model->log_prior.data[c];
            for (usize j = 0; j < d; j++) {
                f64 diff = xrow[j] - mrow[j];
                s -= 0.5 * (CAMLL_LOG_TWO_PI + log(vrow[j]) + diff * diff / vrow[j]);
            }
            lrow[c] = s;
        }
    }
    return ll;
}

Vec naivebayes_predict(Arena *a, const NaiveBayes *model, const Mat *X) {
    Mat ll = naivebayes_predict_log_proba(a, model, X);
    Vec pred = vec_alloc(a, ll.rows);
    for (usize i = 0; i < ll.rows; i++) {
        const f64 *lrow = mat_row(&ll, i);
        usize best = 0;
        f64 best_v = lrow[0];
        for (usize c = 1; c < ll.cols; c++) {
            if (lrow[c] > best_v) {
                best_v = lrow[c];
                best = c;
            }
        }
        pred.data[i] = (f64)best;
    }
    return pred;
}
