#ifndef CAMLL_SRC_MATRIX_H
#define CAMLL_SRC_MATRIX_H

#include "types.h"
#include "alloc/arena.h"

/* Dense f64 vector. */
typedef struct {
    usize len;
    f64 *data;
} Vec;

/* Dense f64 matrix, row-major: element (r, c) is data[r * cols + c]. */
typedef struct {
    usize rows;
    usize cols;
    f64 *data;
} Mat;

/* Storage is arena-backed, so there is no per-object free. Reclaim in bulk
   with arena_reset or arena_reset_to on the owning arena. */
Vec vec_alloc(Arena *a, usize len);
Vec vec_zeros(Arena *a, usize len);
Vec vec_ones(Arena *a, usize len);
Vec vec_copy(Arena *a, const Vec *x);

Mat mat_alloc(Arena *a, usize rows, usize cols);
Mat mat_zeros(Arena *a, usize rows, usize cols);
Mat mat_ones(Arena *a, usize rows, usize cols);
/* n x n identity matrix. */
Mat mat_eye(Arena *a, usize n);
Mat mat_copy(Arena *a, const Mat *x);

static inline f64 mat_get(const Mat *m, usize r, usize c) {
    return m->data[r * m->cols + c];
}

static inline void mat_set(Mat *m, usize r, usize c, f64 v) {
    m->data[r * m->cols + c] = v;
}

static inline f64 *mat_row(const Mat *m, usize r) {
    return m->data + r * m->cols;
}

/* Core linear algebra. Each op writes into a freshly allocated result from
   the given arena; the operands are left untouched. Shape preconditions
   (matching dims for add/sub, x->cols == y->rows for matmul) are checked
   with assert. */
Mat mat_add(Arena *a, const Mat *x, const Mat *y);
Mat mat_sub(Arena *a, const Mat *x, const Mat *y);
Mat mat_scale(Arena *a, const Mat *x, f64 s);
Mat mat_matmul(Arena *a, const Mat *x, const Mat *y);
Mat mat_transpose(Arena *a, const Mat *x);

/* Vector ops. Lengths must match on x and y; checked with assert. */
f64 vec_dot(const Vec *x, const Vec *y);
f64 vec_norm1(const Vec *x);
f64 vec_norm2(const Vec *x);
Vec vec_mul(Arena *a, const Vec *x, const Vec *y);
Vec vec_div(Arena *a, const Vec *x, const Vec *y);

/* Outer product: x->len rows by y->len cols, result[i][j] = x[i] * y[j]. */
Mat mat_outer(Arena *a, const Vec *x, const Vec *y);

/* xorshift64star PRNG. Deterministic given a seed, no external entropy
   source, so results are reproducible across runs. */
typedef struct {
    u64 state;
} Rng;

Rng rng_seed(u64 seed);
/* Uniform in [lo, hi). */
f64 rng_uniform(Rng *r, f64 lo, f64 hi);
/* Normal via Box-Muller transform. */
f64 rng_normal(Rng *r, f64 mean, f64 stddev);

Vec vec_rand_uniform(Arena *a, usize len, Rng *r, f64 lo, f64 hi);
Vec vec_rand_normal(Arena *a, usize len, Rng *r, f64 mean, f64 stddev);
Mat mat_rand_uniform(Arena *a, usize rows, usize cols, Rng *r, f64 lo, f64 hi);
Mat mat_rand_normal(Arena *a, usize rows, usize cols, Rng *r, f64 mean, f64 stddev);

/* Debug printing, one line per row for matrices. */
void vec_print(const Vec *v);
void mat_print(const Mat *m);

#endif /* CAMLL_SRC_MATRIX_H */
