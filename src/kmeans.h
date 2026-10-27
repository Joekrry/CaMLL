#ifndef CAMLL_SRC_KMEANS_H
#define CAMLL_SRC_KMEANS_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Lloyd's algorithm: iterate nearest-centroid assignment and centroid
   averaging until assignments settle or max_iters hits. */
typedef struct {
    Mat centroids; /* k x cols */
    usize k;
} KMeans;

/* Initial centroids are k distinct rows of X, picked via a seeded
   Fisher-Yates shuffle. k must be in [1, X->rows]. An empty cluster keeps
   its previous centroid instead of going to NaN. */
KMeans kmeans_fit(Arena *a, const Mat *X, usize k, usize max_iters, Rng *rng);

/* Nearest centroid index (0..k-1) for each row, as f64 labels. */
Vec kmeans_predict(Arena *a, const KMeans *model, const Mat *X);

/* Sum of squared distances to each row's nearest centroid (inertia). */
f64 kmeans_inertia(const KMeans *model, const Mat *X);

#endif /* CAMLL_SRC_KMEANS_H */
