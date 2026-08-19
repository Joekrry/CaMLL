#ifndef CAMLL_SRC_DATASET_H
#define CAMLL_SRC_DATASET_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

/* Feature matrix (one sample per row) paired with its target vector. */
typedef struct {
    Mat X;
    Vec y;
} Dataset;

/* Parse comma-separated f64 rows from an in-memory buffer. label_col picks
   the target column; the remaining columns become X. Blank lines are skipped
   and every populated row must share the field count of the first. */
Dataset dataset_from_csv(Arena *a, const char *text, usize label_col);
/* Same, reading the whole file at path first. */
Dataset dataset_load_csv(Arena *a, const char *path, usize label_col);

typedef struct {
    Dataset train;
    Dataset test;
} Split;

/* Row-wise partition after a Fisher-Yates shuffle; train_frac in [0, 1]. */
Split dataset_split(Arena *a, const Dataset *d, f64 train_frac, Rng *rng);

/* Per-column mean and standard deviation. */
typedef struct {
    Vec mean;
    Vec std;
} Standardizer;

Standardizer standardizer_fit(Arena *a, const Mat *X);
Mat standardizer_apply(Arena *a, const Standardizer *s, const Mat *X);

#endif /* CAMLL_SRC_DATASET_H */
