#include "kmeans.h"

#include <assert.h>
#include <string.h>

/* Index of the centroid closest to row (squared Euclidean), writing its
   squared distance to *out_dist if non-NULL. */
static usize kmeans_nearest(const KMeans *model, const f64 *row, usize d, f64 *out_dist) {
    usize best = 0;
    f64 best_dist = 0.0;
    for (usize c = 0; c < model->k; c++) {
        const f64 *cen = mat_row(&model->centroids, c);
        f64 dist = 0.0;
        for (usize j = 0; j < d; j++) {
            f64 diff = row[j] - cen[j];
            dist += diff * diff;
        }
        if (c == 0 || dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    if (out_dist != NULL) *out_dist = best_dist;
    return best;
}

KMeans kmeans_fit(Arena *a, const Mat *X, usize k, usize max_iters, Rng *rng) {
    assert(k > 0 && k <= X->rows);
    usize n = X->rows;
    usize d = X->cols;

    usize *idx = (usize *)arena_alloc(a, n * sizeof(usize));
    for (usize i = 0; i < n; i++) idx[i] = i;
    for (usize i = n; i > 1; i--) {
        usize j = (usize)rng_uniform(rng, 0.0, (f64)i);
        if (j >= i) j = i - 1;      /* guard the half-open upper bound */
        usize t = idx[i - 1];
        idx[i - 1] = idx[j];
        idx[j] = t;
    }

    KMeans model;
    model.k = k;
    model.centroids = mat_alloc(a, k, d);
    for (usize c = 0; c < k; c++) {
        memcpy(mat_row(&model.centroids, c), mat_row(X, idx[c]), d * sizeof(f64));
    }

    usize *assign = (usize *)arena_alloc(a, n * sizeof(usize));
    for (usize i = 0; i < n; i++) assign[i] = k; /* sentinel: no cluster yet */

    f64 *sums = (f64 *)arena_alloc(a, k * d * sizeof(f64));
    usize *counts = (usize *)arena_alloc(a, k * sizeof(usize));

    for (usize iter = 0; iter < max_iters; iter++) {
        int changed = 0;
        for (usize i = 0; i < n; i++) {
            usize best = kmeans_nearest(&model, mat_row(X, i), d, NULL);
            if (assign[i] != best) {
                assign[i] = best;
                changed = 1;
            }
        }
        if (!changed) break;

        memset(sums, 0, k * d * sizeof(f64));
        memset(counts, 0, k * sizeof(usize));
        for (usize i = 0; i < n; i++) {
            usize c = assign[i];
            counts[c]++;
            const f64 *row = mat_row(X, i);
            f64 *sum = sums + c * d;
            for (usize j = 0; j < d; j++) sum[j] += row[j];
        }
        for (usize c = 0; c < k; c++) {
            if (counts[c] == 0) continue; /* empty cluster: keep previous centroid */
            f64 *cen = mat_row(&model.centroids, c);
            f64 *sum = sums + c * d;
            for (usize j = 0; j < d; j++) cen[j] = sum[j] / (f64)counts[c];
        }
    }

    return model;
}

Vec kmeans_predict(Arena *a, const KMeans *model, const Mat *X) {
    assert(X->cols == model->centroids.cols);
    Vec pred = vec_alloc(a, X->rows);
    for (usize i = 0; i < X->rows; i++) {
        pred.data[i] = (f64)kmeans_nearest(model, mat_row(X, i), X->cols, NULL);
    }
    return pred;
}

f64 kmeans_inertia(const KMeans *model, const Mat *X) {
    assert(X->cols == model->centroids.cols);
    f64 total = 0.0;
    for (usize i = 0; i < X->rows; i++) {
        f64 dist;
        kmeans_nearest(model, mat_row(X, i), X->cols, &dist);
        total += dist;
    }
    return total;
}
