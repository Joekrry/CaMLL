#include "logreg.h"
#include "activation.h"

#include <assert.h>
#include <math.h>

static Vec mat_vec_mul(Arena *a, const Mat *m, const Vec *x) {
    assert(m->cols == x->len);
    Vec r = vec_zeros(a, m->rows);
    for (usize i = 0; i < m->rows; i++) {
        const f64 *row = mat_row(m, i);
        f64 s = 0.0;
        for (usize j = 0; j < m->cols; j++) {
            s += row[j] * x->data[j];
        }
        r.data[i] = s;
    }
    return r;
}

/* x^T * m, result length m->cols */
static Vec vec_mat_mul(Arena *a, const Vec *x, const Mat *m) {
    assert(m->rows == x->len);
    Vec r = vec_zeros(a, m->cols);
    for (usize i = 0; i < m->rows; i++) {
        f64 xi = x->data[i];
        const f64 *row = mat_row(m, i);
        for (usize j = 0; j < m->cols; j++) {
            r.data[j] += xi * row[j];
        }
    }
    return r;
}

Vec logreg_predict_proba(Arena *a, const LogReg *model, const Mat *X) {
    assert(X->cols == model->weights.len);
    Vec p = mat_vec_mul(a, X, &model->weights);
    for (usize i = 0; i < p.len; i++) {
        p.data[i] = sigmoid(p.data[i] + model->bias);
    }
    return p;
}

LogReg logreg_fit_gd(Arena *a, const Mat *X, const Vec *y, f64 lr, usize epochs) {
    assert(X->rows == y->len);
    usize n = X->rows;
    usize d = X->cols;

    LogReg model;
    model.weights = vec_zeros(a, d);
    model.bias = 0.0;

    for (usize epoch = 0; epoch < epochs; epoch++) {
        Vec proba = logreg_predict_proba(a, &model, X);
        Vec err = vec_alloc(a, n);
        for (usize i = 0; i < n; i++) {
            err.data[i] = proba.data[i] - y->data[i];
        }

        Vec grad_w = vec_mat_mul(a, &err, X);
        f64 grad_b = 0.0;
        for (usize i = 0; i < n; i++) {
            grad_b += err.data[i];
        }
        grad_b /= (f64)n;

        for (usize j = 0; j < d; j++) {
            model.weights.data[j] -= lr * (grad_w.data[j] / (f64)n);
        }
        model.bias -= lr * grad_b;
    }
    return model;
}

Vec logreg_predict(Arena *a, const LogReg *model, const Mat *X) {
    Vec proba = logreg_predict_proba(a, model, X);
    Vec pred = vec_alloc(a, proba.len);
    for (usize i = 0; i < proba.len; i++) {
        pred.data[i] = proba.data[i] >= 0.5 ? 1.0 : 0.0;
    }
    return pred;
}

f64 logreg_bce(const Vec *y_true, const Vec *y_proba) {
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

f64 logreg_accuracy(const Vec *y_true, const Vec *y_pred) {
    assert(y_true->len == y_pred->len);
    usize correct = 0;
    for (usize i = 0; i < y_true->len; i++) {
        if (y_pred->data[i] == y_true->data[i]) correct++;
    }
    return (f64)correct / (f64)y_true->len;
}
