#ifndef CAMLL_SRC_METRICS_H
#define CAMLL_SRC_METRICS_H

#include "types.h"
#include "matrix.h"

/* Mean squared error between predictions and targets. */
f64 metrics_mse(const Vec *y_true, const Vec *y_pred);

/* Binary cross-entropy: true labels (0.0/1.0) vs. predicted probability of
   class 1. Probabilities are clamped away from 0/1 to keep the log finite. */
f64 metrics_bce(const Vec *y_true, const Vec *y_proba);

/* Fraction of predicted labels matching the true labels. */
f64 metrics_accuracy(const Vec *y_true, const Vec *y_pred);

/* Binary confusion matrix counts; positive class is 1.0. */
typedef struct {
    usize tp;
    usize fp;
    usize tn;
    usize fn;
} ConfusionMatrix;

ConfusionMatrix metrics_confusion(const Vec *y_true, const Vec *y_pred);

/* tp / (tp + fp). 0.0 when there are no predicted positives. */
f64 metrics_precision(const ConfusionMatrix *cm);

/* tp / (tp + fn). 0.0 when there are no true positives. */
f64 metrics_recall(const ConfusionMatrix *cm);

/* Harmonic mean of precision and recall. 0.0 when both are 0. */
f64 metrics_f1(const ConfusionMatrix *cm);

#endif /* CAMLL_SRC_METRICS_H */
