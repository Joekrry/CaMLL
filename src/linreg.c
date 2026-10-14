#include "linreg.h"

#include <assert.h>
#include <math.h>
#include <string.h>

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

/* Prepend a column of ones for the bias term. */
static Mat augment_ones(Arena *a, const Mat *X) {
    Mat Xb = mat_alloc(a, X->rows, X->cols + 1);
    for (usize i = 0; i < X->rows; i++) {
        mat_set(&Xb, i, 0, 1.0);
        for (usize j = 0; j < X->cols; j++) {
            mat_set(&Xb, i, j + 1, mat_get(X, i, j));
        }
    }
    return Xb;
}

/* Solve A w = b for w via Gaussian elimination with partial pivoting.
   A is n x n row-major, both A and b are overwritten; the solution ends up
   in b. */
static void gauss_solve(f64 *A, f64 *b, usize n) {
    for (usize col = 0; col < n; col++) {
        usize piv = col;
        f64 best = fabs(A[col * n + col]);
        for (usize r = col + 1; r < n; r++) {
            f64 v = fabs(A[r * n + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (usize c = 0; c < n; c++) {
                f64 t = A[col * n + c];
                A[col * n + c] = A[piv * n + c];
                A[piv * n + c] = t;
            }
            f64 t = b[col];
            b[col] = b[piv];
            b[piv] = t;
        }
        f64 pv = A[col * n + col];
        assert(pv != 0.0);
        for (usize r = col + 1; r < n; r++) {
            f64 factor = A[r * n + col] / pv;
            for (usize c = col; c < n; c++) {
                A[r * n + c] -= factor * A[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }
    for (isize row = (isize)n - 1; row >= 0; row--) {
        f64 s = b[row];
        for (usize c = (usize)row + 1; c < n; c++) {
            s -= A[(usize)row * n + c] * b[c];
        }
        b[row] = s / A[(usize)row * n + (usize)row];
    }
}

LinReg linreg_fit_normal(Arena *a, const Mat *X, const Vec *y) {
    assert(X->rows == y->len);
    Mat Xb = augment_ones(a, X);
    Mat Xt = mat_transpose(a, &Xb);
    Mat XtX = mat_matmul(a, &Xt, &Xb);
    Vec Xty = mat_vec_mul(a, &Xt, y);

    usize n = XtX.rows;
    f64 *A = (f64 *)arena_alloc(a, n * n * sizeof(f64));
    f64 *b = (f64 *)arena_alloc(a, n * sizeof(f64));
    memcpy(A, XtX.data, n * n * sizeof(f64));
    memcpy(b, Xty.data, n * sizeof(f64));

    gauss_solve(A, b, n);

    LinReg model;
    model.bias = b[0];
    model.weights = vec_alloc(a, n - 1);
    for (usize i = 1; i < n; i++) {
        model.weights.data[i - 1] = b[i];
    }
    return model;
}

LinReg linreg_fit_gd(Arena *a, const Mat *X, const Vec *y, f64 lr, usize epochs, f64 l2) {
    assert(X->rows == y->len);
    usize n = X->rows;
    usize d = X->cols;

    LinReg model;
    model.weights = vec_zeros(a, d);
    model.bias = 0.0;

    for (usize epoch = 0; epoch < epochs; epoch++) {
        Vec pred = linreg_predict(a, &model, X);
        Vec err = vec_alloc(a, n);
        for (usize i = 0; i < n; i++) {
            err.data[i] = pred.data[i] - y->data[i];
        }

        Vec grad_w = vec_mat_mul(a, &err, X);
        f64 grad_b = 0.0;
        for (usize i = 0; i < n; i++) {
            grad_b += err.data[i];
        }
        grad_b /= (f64)n;

        for (usize j = 0; j < d; j++) {
            model.weights.data[j] -= lr * (grad_w.data[j] / (f64)n + l2 * model.weights.data[j]);
        }
        model.bias -= lr * grad_b;
    }
    return model;
}

Vec linreg_predict(Arena *a, const LinReg *model, const Mat *X) {
    assert(X->cols == model->weights.len);
    Vec pred = mat_vec_mul(a, X, &model->weights);
    for (usize i = 0; i < pred.len; i++) {
        pred.data[i] += model->bias;
    }
    return pred;
}
