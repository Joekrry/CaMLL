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

/* Closed-form fit via the normal equations w = (X^T X)^-1 X^T y, with an
   implicit bias column; exact only for well-conditioned, full-rank X. */
LinReg linreg_fit_normal(Arena *a, const Mat *X, const Vec *y);

/* Batch GD: epochs full passes at the given learning rate, from zero
   weights/bias. l2 is the ridge penalty on weights only (bias stays
   unregularized); 0.0 disables it. */
LinReg linreg_fit_gd(Arena *a, const Mat *X, const Vec *y, f64 lr, usize epochs, f64 l2);

Vec linreg_predict(Arena *a, const LinReg *model, const Mat *X);

#endif /* CAMLL_SRC_LINREG_H */
