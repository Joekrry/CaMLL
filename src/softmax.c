#include "softmax.h"
#include "activation.h"
#include <assert.h>

Mat softmax_predict_proba(Arena *a, const Softmax *model, const Mat *X) {
    assert(X->cols == model->weights.cols);
    usize n = X->rows;
    usize k = model->weights.rows;

    Mat wt = mat_transpose(a, &model->weights);
    Mat z = mat_matmul(a, X, &wt);
    Mat p = mat_alloc(a, n, k);

    for (usize i = 0; i < n; i++) {
        Vec row = { .len = k, .data = mat_row(&z, i) };
        for (usize c = 0; c < k; c++) {
            row.data[c] += model->bias.data[c];
        }
        Vec sm = vec_softmax(a, &row);
        f64 *prow = mat_row(&p, i);
        for (usize c = 0; c < k; c++) {
            prow[c] = sm.data[c];
        }
    }
    return p;
}

Softmax softmax_fit_gd(Arena *a, const Mat *X, const Vec *y, usize num_classes, f64 lr, usize epochs) {
    assert(X->rows == y->len);
    usize n = X->rows;
    usize d = X->cols;
    usize k = num_classes;

    Softmax model;
    model.weights = mat_zeros(a, k, d);
    model.bias = vec_zeros(a, k);

    for (usize epoch = 0; epoch < epochs; epoch++) {
        Mat proba = softmax_predict_proba(a, &model, X);
        Mat grad_w = mat_zeros(a, k, d);
        Vec grad_b = vec_zeros(a, k);

        for (usize i = 0; i < n; i++) {
            usize yi = (usize)y->data[i];
            const f64 *prow = mat_row(&proba, i);
            const f64 *xrow = mat_row(X, i);
            for (usize c = 0; c < k; c++) {
                f64 err = prow[c] - (c == yi ? 1.0 : 0.0);
                grad_b.data[c] += err;
                f64 *gwrow = mat_row(&grad_w, c);
                for (usize j = 0; j < d; j++) {
                    gwrow[j] += err * xrow[j];
                }
            }
        }

        for (usize c = 0; c < k; c++) {
            f64 *wrow = mat_row(&model.weights, c);
            const f64 *gwrow = mat_row(&grad_w, c);
            for (usize j = 0; j < d; j++) {
                wrow[j] -= lr * (gwrow[j] / (f64)n);
            }
            model.bias.data[c] -= lr * (grad_b.data[c] / (f64)n);
        }
    }
    return model;
}

Vec softmax_predict(Arena *a, const Softmax *model, const Mat *X) {
    Mat p = softmax_predict_proba(a, model, X);
    Vec pred = vec_alloc(a, p.rows);
    for (usize i = 0; i < p.rows; i++) {
        const f64 *prow = mat_row(&p, i);
        usize best = 0;
        f64 best_v = prow[0];
        for (usize c = 1; c < p.cols; c++) {
            if (prow[c] > best_v) {
                best_v = prow[c];
                best = c;
            }
        }
        pred.data[i] = (f64)best;
    }
    return pred;
}
