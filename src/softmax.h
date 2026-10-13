#ifndef CAMLL_SRC_SOFTMAX_H
#define CAMLL_SRC_SOFTMAX_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Softmax regression (multinomial logistic regression) */
typedef struct {
    Mat weights;
    Vec bias;
} Softmax;

/* Batch gradient descent fit on multinomial cross-entropy loss: */
Softmax softmax_fit_gd(Arena *a, const Mat *X, const Vec *y, usize num_classes, f64 lr, usize epochs);

/* Per-class predicted probabilities */
Mat softmax_predict_proba(Arena *a, const Softmax *model, const Mat *X);

/* Predicted class index */
Vec softmax_predict(Arena *a, const Softmax *model, const Mat *X);

#endif /* CAMLL_SRC_SOFTMAX_H */
