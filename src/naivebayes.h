#ifndef CAMLL_SRC_NAIVEBAYES_H
#define CAMLL_SRC_NAIVEBAYES_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Each feature modeled as an independent per-class Gaussian. */
typedef struct {
    usize num_classes;
    Mat mean;      /* num_classes x cols */
    Mat var;       /* num_classes x cols */
    Vec log_prior; /* num_classes */
} NaiveBayes;

/* Classes run 0..num_classes-1. var_smoothing guards against a zero-variance feature. */
NaiveBayes naivebayes_fit(Arena *a, const Mat *X, const Vec *y, usize num_classes, f64 var_smoothing);

/* Log-likelihood per class: log prior plus summed log Gaussian density. */
Mat naivebayes_predict_log_proba(Arena *a, const NaiveBayes *model, const Mat *X);

/* Predicted class: argmax of the log-likelihood. */
Vec naivebayes_predict(Arena *a, const NaiveBayes *model, const Mat *X);

#endif /* CAMLL_SRC_NAIVEBAYES_H */
