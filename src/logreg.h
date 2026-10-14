#ifndef CAMLL_SRC_LOGREG_H
#define CAMLL_SRC_LOGREG_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Logistic regression: p ~= sigmoid(X w + bias), one weight per column of X. */
typedef struct {
    Vec weights;
    f64 bias;
} LogReg;

/* Batch gradient descent fit on binary cross-entropy loss: epochs full
   passes over the data at the given learning rate, starting from zero
   weights and bias. y entries must be 0.0 or 1.0. l2 is the ridge penalty
   strength applied to the weights only (not the bias); 0.0 disables
   regularization. */
LogReg logreg_fit_gd(Arena *a, const Mat *X, const Vec *y, f64 lr, usize epochs, f64 l2);

/* Predicted probability of class 1. */
Vec logreg_predict_proba(Arena *a, const LogReg *model, const Mat *X);

/* Predicted class, 0.0 or 1.0, by thresholding the probability at 0.5. */
Vec logreg_predict(Arena *a, const LogReg *model, const Mat *X);

#endif /* CAMLL_SRC_LOGREG_H */
