#ifndef CAMLL_SRC_KMEANS_H
#define CAMLL_SRC_KMEANS_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Unsupervised: Lloyd's algorithm. Assigns each row to its nearest centroid,
   then recomputes each centroid as the mean of its assigned rows, until
   assignments stop changing or max_iters is reached. */
typedef struct {
    Mat centroids; /* k x cols */
    usize k;
} KMeans;

/* Initial centroids are k distinct rows of X, chosen by a Fisher-Yates
   shuffle of the row indices seeded by rng. k must be in [1, X->rows]. A
   cluster left with no assigned rows keeps its previous centroid rather than
   going to NaN. */
KMeans kmeans_fit(Arena *a, const Mat *X, usize k, usize max_iters, Rng *rng);

/* Nearest centroid index (0..k-1) for each row, as f64 labels. */
Vec kmeans_predict(Arena *a, const KMeans *model, const Mat *X);

/* Sum of squared distances from each row to its nearest centroid
   (within-cluster sum of squares), a.k.a. inertia. */
f64 kmeans_inertia(const KMeans *model, const Mat *X);

#endif /* CAMLL_SRC_KMEANS_H */
