#ifndef CAMLL_SRC_LINREG_H
#define CAMLL_SRC_LINREG_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Linear regression: y ~= X w + bias, one weight per column of X. */
typedef struct {
    Vec weights;
    f64 bias;
} LinReg;

/* Closed-form fit via the normal equations: w = (X^T X)^-1 X^T y, with an
   implicit bias column. Exact for well-conditioned, full-rank X. */
LinReg linreg_fit_normal(Arena *a, const Mat *X, const Vec *y);

/* Batch gradient descent fit: epochs full passes over the data at the
   given learning rate, starting from zero weights and bias. */
LinReg linreg_fit_gd(Arena *a, const Mat *X, const Vec *y, f64 lr, usize epochs);

Vec linreg_predict(Arena *a, const LinReg *model, const Mat *X);

/* Mean squared error between predictions and targets. */
f64 linreg_mse(const Vec *y_true, const Vec *y_pred);

#endif /* CAMLL_SRC_LINREG_H */
