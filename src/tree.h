#ifndef CAMLL_SRC_TREE_H
#define CAMLL_SRC_TREE_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

typedef enum {
    TREE_GINI,
    TREE_ENTROPY,
    TREE_VARIANCE
} TreeCriterion;

typedef struct TreeNode TreeNode;
struct TreeNode {
    int is_leaf;
    f64 value;
    usize feature;
    f64 threshold;
    TreeNode *left;
    TreeNode *right;
};

typedef struct {
    TreeNode *root;
    TreeCriterion criterion;
} Tree;

Tree tree_fit_classify(Arena *a, const Mat *X, const Vec *y, usize num_classes,
                       TreeCriterion criterion, usize max_depth, usize min_samples_split,
                       usize max_features, Rng *rng);

Tree tree_fit_regress(Arena *a, const Mat *X, const Vec *y, usize max_depth, usize min_samples_split,
                      usize max_features, Rng *rng);

Vec tree_predict(Arena *a, const Tree *model, const Mat *X);

#endif /* CAMLL_SRC_TREE_H */
