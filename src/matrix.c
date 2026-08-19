#include "matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CAMML_TWO_PI 6.283185307179586476925286766559

Vec vec_alloc(Arena *a, usize len) {
    Vec v;
    v.len = len;
    v.data = (f64 *)arena_alloc(a, len * sizeof(f64));
    return v;
}

Vec vec_zeros(Arena *a, usize len) {
    Vec v = vec_alloc(a, len);
    memset(v.data, 0, len * sizeof(f64));
    return v;
}

Mat mat_alloc(Arena *a, usize rows, usize cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (f64 *)arena_alloc(a, rows * cols * sizeof(f64));
    return m;
}

Mat mat_zeros(Arena *a, usize rows, usize cols) {
    Mat m = mat_alloc(a, rows, cols);
    memset(m.data, 0, rows * cols * sizeof(f64));
    return m;
}

Vec vec_ones(Arena *a, usize len) {
    Vec v = vec_alloc(a, len);
    for (usize i = 0; i < len; i++) {
        v.data[i] = 1.0;
    }
    return v;
}

Vec vec_copy(Arena *a, const Vec *x) {
    Vec v = vec_alloc(a, x->len);
    memcpy(v.data, x->data, x->len * sizeof(f64));
    return v;
}

Mat mat_ones(Arena *a, usize rows, usize cols) {
    Mat m = mat_alloc(a, rows, cols);
    usize n = rows * cols;
    for (usize i = 0; i < n; i++) {
        m.data[i] = 1.0;
    }
    return m;
}

Mat mat_eye(Arena *a, usize n) {
    Mat m = mat_zeros(a, n, n);
    for (usize i = 0; i < n; i++) {
        mat_set(&m, i, i, 1.0);
    }
    return m;
}

Mat mat_copy(Arena *a, const Mat *x) {
    Mat m = mat_alloc(a, x->rows, x->cols);
    memcpy(m.data, x->data, x->rows * x->cols * sizeof(f64));
    return m;
}

Mat mat_add(Arena *a, const Mat *x, const Mat *y) {
    assert(x->rows == y->rows && x->cols == y->cols);
    Mat r = mat_alloc(a, x->rows, x->cols);
    usize n = x->rows * x->cols;
    for (usize i = 0; i < n; i++) {
        r.data[i] = x->data[i] + y->data[i];
    }
    return r;
}

Mat mat_sub(Arena *a, const Mat *x, const Mat *y) {
    assert(x->rows == y->rows && x->cols == y->cols);
    Mat r = mat_alloc(a, x->rows, x->cols);
    usize n = x->rows * x->cols;
    for (usize i = 0; i < n; i++) {
        r.data[i] = x->data[i] - y->data[i];
    }
    return r;
}

Mat mat_scale(Arena *a, const Mat *x, f64 s) {
    Mat r = mat_alloc(a, x->rows, x->cols);
    usize n = x->rows * x->cols;
    for (usize i = 0; i < n; i++) {
        r.data[i] = x->data[i] * s;
    }
    return r;
}

Mat mat_matmul(Arena *a, const Mat *x, const Mat *y) {
    assert(x->cols == y->rows);
    Mat r = mat_zeros(a, x->rows, y->cols);
    /* ikj order keeps the inner loop sequential over both y and r rows */
    for (usize i = 0; i < x->rows; i++) {
        for (usize k = 0; k < x->cols; k++) {
            f64 xik = x->data[i * x->cols + k];
            const f64 *yrow = y->data + k * y->cols;
            f64 *rrow = r.data + i * r.cols;
            for (usize j = 0; j < y->cols; j++) {
                rrow[j] += xik * yrow[j];
            }
        }
    }
    return r;
}

Mat mat_transpose(Arena *a, const Mat *x) {
    Mat r = mat_alloc(a, x->cols, x->rows);
    for (usize i = 0; i < x->rows; i++) {
        for (usize j = 0; j < x->cols; j++) {
            r.data[j * r.cols + i] = x->data[i * x->cols + j];
        }
    }
    return r;
}

f64 vec_dot(const Vec *x, const Vec *y) {
    assert(x->len == y->len);
    f64 s = 0.0;
    for (usize i = 0; i < x->len; i++) {
        s += x->data[i] * y->data[i];
    }
    return s;
}

f64 vec_norm1(const Vec *x) {
    f64 s = 0.0;
    for (usize i = 0; i < x->len; i++) {
        s += fabs(x->data[i]);
    }
    return s;
}

f64 vec_norm2(const Vec *x) {
    return sqrt(vec_dot(x, x));
}

Vec vec_mul(Arena *a, const Vec *x, const Vec *y) {
    assert(x->len == y->len);
    Vec r = vec_alloc(a, x->len);
    for (usize i = 0; i < x->len; i++) {
        r.data[i] = x->data[i] * y->data[i];
    }
    return r;
}

Vec vec_div(Arena *a, const Vec *x, const Vec *y) {
    assert(x->len == y->len);
    Vec r = vec_alloc(a, x->len);
    for (usize i = 0; i < x->len; i++) {
        r.data[i] = x->data[i] / y->data[i];
    }
    return r;
}

Mat mat_outer(Arena *a, const Vec *x, const Vec *y) {
    Mat r = mat_alloc(a, x->len, y->len);
    for (usize i = 0; i < x->len; i++) {
        f64 xi = x->data[i];
        f64 *rrow = r.data + i * r.cols;
        for (usize j = 0; j < y->len; j++) {
            rrow[j] = xi * y->data[j];
        }
    }
    return r;
}

Rng rng_seed(u64 seed) {
    Rng r;
    /* xorshift64star is undefined at state 0 */
    r.state = seed != 0 ? seed : 0x9E3779B97F4A7C15ULL;
    return r;
}

static u64 rng_next(Rng *r) {
    u64 x = r->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

f64 rng_uniform(Rng *r, f64 lo, f64 hi) {
    /* 53 significant bits, matching f64 mantissa precision */
    f64 u = (f64)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
    return lo + u * (hi - lo);
}

f64 rng_normal(Rng *r, f64 mean, f64 stddev) {
    f64 u1 = rng_uniform(r, 0.0, 1.0);
    f64 u2 = rng_uniform(r, 0.0, 1.0);
    /* avoid log(0) */
    if (u1 < 1e-300) u1 = 1e-300;
    f64 z = sqrt(-2.0 * log(u1)) * cos(CAMML_TWO_PI * u2);
    return mean + z * stddev;
}

Vec vec_rand_uniform(Arena *a, usize len, Rng *r, f64 lo, f64 hi) {
    Vec v = vec_alloc(a, len);
    for (usize i = 0; i < len; i++) {
        v.data[i] = rng_uniform(r, lo, hi);
    }
    return v;
}

Vec vec_rand_normal(Arena *a, usize len, Rng *r, f64 mean, f64 stddev) {
    Vec v = vec_alloc(a, len);
    for (usize i = 0; i < len; i++) {
        v.data[i] = rng_normal(r, mean, stddev);
    }
    return v;
}

Mat mat_rand_uniform(Arena *a, usize rows, usize cols, Rng *r, f64 lo, f64 hi) {
    Mat m = mat_alloc(a, rows, cols);
    usize n = rows * cols;
    for (usize i = 0; i < n; i++) {
        m.data[i] = rng_uniform(r, lo, hi);
    }
    return m;
}

Mat mat_rand_normal(Arena *a, usize rows, usize cols, Rng *r, f64 mean, f64 stddev) {
    Mat m = mat_alloc(a, rows, cols);
    usize n = rows * cols;
    for (usize i = 0; i < n; i++) {
        m.data[i] = rng_normal(r, mean, stddev);
    }
    return m;
}

void vec_print(const Vec *v) {
    printf("[");
    for (usize i = 0; i < v->len; i++) {
        printf(i == 0 ? "%g" : ", %g", v->data[i]);
    }
    printf("]\n");
}

void mat_print(const Mat *m) {
    for (usize i = 0; i < m->rows; i++) {
        printf("[");
        for (usize j = 0; j < m->cols; j++) {
            printf(j == 0 ? "%g" : ", %g", mat_get(m, i, j));
        }
        printf("]\n");
    }
}
