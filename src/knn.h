#ifndef CAMLL_SRC_KNN_H
#define CAMLL_SRC_KNN_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Lazy learner: predicts from the k rows closest by squared Euclidean distance. */
typedef struct {
    Mat X;
    Vec y;
    usize k;
} KNN;

/* Copies X and y. k must be in [1, X->rows]. */
KNN knn_fit(Arena *a, const Mat *X, const Vec *y, usize k);

Vec knn_predict_regress(Arena *a, const KNN *model, const Mat *X);

/* Ties go to the smallest label value. */
Vec knn_predict_classify(Arena *a, const KNN *model, const Mat *X);

#endif /* CAMLL_SRC_KNN_H */
