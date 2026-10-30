#ifndef CAMLL_SRC_FOREST_H
#define CAMLL_SRC_FOREST_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"
#include "tree.h"

typedef struct {
    Tree *trees;
    usize num_trees;
    usize num_classes;
    int is_classification;
} Forest;

Forest forest_fit_classify(Arena *a, const Mat *X, const Vec *y, usize num_classes,
                           TreeCriterion criterion, usize num_trees, usize max_depth,
                           usize min_samples_split, usize max_features, Rng *rng);

Forest forest_fit_regress(Arena *a, const Mat *X, const Vec *y, usize num_trees, usize max_depth,
                          usize min_samples_split, usize max_features, Rng *rng);

Vec forest_predict(Arena *a, const Forest *model, const Mat *X);

#endif /* CAMLL_SRC_FOREST_H */
