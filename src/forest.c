#include "forest.h"

#include <assert.h>
#include <string.h>

static void forest_bootstrap(Mat *Xb, Vec *yb, const Mat *X, const Vec *y, Rng *rng) {
    usize n = X->rows;
    for (usize i = 0; i < n; i++) {
        usize src = (usize)rng_uniform(rng, 0.0, (f64)n);
        if (src >= n) src = n - 1;
        memcpy(mat_row(Xb, i), mat_row(X, src), X->cols * sizeof(f64));
        yb->data[i] = y->data[src];
    }
}

Forest forest_fit_classify(Arena *a, const Mat *X, const Vec *y, usize num_classes,
                           TreeCriterion criterion, usize num_trees, usize max_depth,
                           usize min_samples_split, usize max_features, Rng *rng) {
    assert(X->rows == y->len);
    assert(X->rows > 0);
    assert(num_trees > 0);

    Forest model;
    model.is_classification = 1;
    model.num_classes = num_classes;
    model.num_trees = num_trees;
    model.trees = (Tree *)arena_alloc(a, num_trees * sizeof(Tree));

    Mat Xb = mat_alloc(a, X->rows, X->cols);
    Vec yb = vec_alloc(a, X->rows);
    for (usize t = 0; t < num_trees; t++) {
        forest_bootstrap(&Xb, &yb, X, y, rng);
        model.trees[t] = tree_fit_classify(a, &Xb, &yb, num_classes, criterion, max_depth,
                                           min_samples_split, max_features, rng);
    }
    return model;
}

Forest forest_fit_regress(Arena *a, const Mat *X, const Vec *y, usize num_trees, usize max_depth,
                          usize min_samples_split, usize max_features, Rng *rng) {
    assert(X->rows == y->len);
    assert(X->rows > 0);
    assert(num_trees > 0);

    Forest model;
    model.is_classification = 0;
    model.num_classes = 0;
    model.num_trees = num_trees;
    model.trees = (Tree *)arena_alloc(a, num_trees * sizeof(Tree));

    Mat Xb = mat_alloc(a, X->rows, X->cols);
    Vec yb = vec_alloc(a, X->rows);
    for (usize t = 0; t < num_trees; t++) {
        forest_bootstrap(&Xb, &yb, X, y, rng);
        model.trees[t] = tree_fit_regress(a, &Xb, &yb, max_depth, min_samples_split, max_features, rng);
    }
    return model;
}

Vec forest_predict(Arena *a, const Forest *model, const Mat *X) {
    usize n = X->rows;
    Vec pred = vec_alloc(a, n);

    if (model->is_classification) {
        Mat votes = mat_zeros(a, n, model->num_classes);
        for (usize t = 0; t < model->num_trees; t++) {
            Vec p = tree_predict(a, &model->trees[t], X);
            for (usize i = 0; i < n; i++) {
                usize c = (usize)p.data[i];
                mat_set(&votes, i, c, mat_get(&votes, i, c) + 1.0);
            }
        }
        for (usize i = 0; i < n; i++) {
            usize best = 0;
            f64 best_count = mat_get(&votes, i, 0);
            for (usize c = 1; c < model->num_classes; c++) {
                f64 v = mat_get(&votes, i, c);
                if (v > best_count) {
                    best_count = v;
                    best = c;
                }
            }
            pred.data[i] = (f64)best;
        }
    } else {
        for (usize i = 0; i < n; i++) pred.data[i] = 0.0;
        for (usize t = 0; t < model->num_trees; t++) {
            Vec p = tree_predict(a, &model->trees[t], X);
            for (usize i = 0; i < n; i++) pred.data[i] += p.data[i];
        }
        for (usize i = 0; i < n; i++) pred.data[i] /= (f64)model->num_trees;
    }

    return pred;
}
