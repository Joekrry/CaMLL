#include "tree.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
    f64 value;
    usize idx;
} TreeSplitPoint;

static int tree_splitpoint_cmp(const void *pa, const void *pb) {
    const TreeSplitPoint *a = (const TreeSplitPoint *)pa;
    const TreeSplitPoint *b = (const TreeSplitPoint *)pb;
    if (a->value < b->value) return -1;
    if (a->value > b->value) return 1;
    return 0;
}

static TreeNode *tree_node_alloc(Arena *a) {
    return (TreeNode *)arena_alloc(a, sizeof(TreeNode));
}

static void tree_partition(Arena *a, const Mat *X, const usize *idx, usize n, usize feature, f64 threshold,
                            usize **out_left, usize *out_nl, usize **out_right, usize *out_nr) {
    usize *left = (usize *)arena_alloc(a, n * sizeof(usize));
    usize *right = (usize *)arena_alloc(a, n * sizeof(usize));
    usize nl = 0, nr = 0;
    for (usize i = 0; i < n; i++) {
        usize row = idx[i];
        if (mat_get(X, row, feature) <= threshold) {
            left[nl++] = row;
        } else {
            right[nr++] = row;
        }
    }
    *out_left = left;
    *out_nl = nl;
    *out_right = right;
    *out_nr = nr;
}

typedef struct {
    const Mat *X;
    const Vec *y;
    usize num_classes;
    TreeCriterion criterion;
    usize max_depth;
    usize min_samples_split;
    usize max_features;
    Rng *rng;
    Arena *a;
    usize *count_total;
    usize *count_left;
    usize *count_right;
    usize *cols;
} TreeClassifyCtx;

static usize tree_select_features(usize *cols, usize d, usize max_features, Rng *rng) {
    if (max_features == 0 || max_features >= d) return d;
    for (usize i = 0; i < max_features; i++) {
        usize j = i + (usize)rng_uniform(rng, 0.0, (f64)(d - i));
        if (j >= d) j = d - 1;
        usize t = cols[i];
        cols[i] = cols[j];
        cols[j] = t;
    }
    return max_features;
}

static f64 tree_impurity_counts(const usize *count, usize num_classes, usize n, TreeCriterion criterion) {
    if (n == 0) return 0.0;
    if (criterion == TREE_GINI) {
        f64 sum_sq = 0.0;
        for (usize c = 0; c < num_classes; c++) {
            f64 p = (f64)count[c] / (f64)n;
            sum_sq += p * p;
        }
        return 1.0 - sum_sq;
    }
    f64 h = 0.0;
    for (usize c = 0; c < num_classes; c++) {
        if (count[c] == 0) continue;
        f64 p = (f64)count[c] / (f64)n;
        h -= p * log(p);
    }
    return h;
}

static f64 tree_majority(const usize *count, usize num_classes) {
    usize best = 0;
    usize best_count = count[0];
    for (usize c = 1; c < num_classes; c++) {
        if (count[c] > best_count) {
            best_count = count[c];
            best = c;
        }
    }
    return (f64)best;
}

static TreeNode *tree_build_classify(TreeClassifyCtx *ctx, usize *idx, usize n, usize depth) {
    usize k = ctx->num_classes;
    for (usize c = 0; c < k; c++) ctx->count_total[c] = 0;
    for (usize i = 0; i < n; i++) ctx->count_total[(usize)ctx->y->data[idx[i]]]++;

    f64 parent_impurity = tree_impurity_counts(ctx->count_total, k, n, ctx->criterion);

    TreeNode *node = tree_node_alloc(ctx->a);
    node->left = NULL;
    node->right = NULL;

    if (depth >= ctx->max_depth || n < ctx->min_samples_split || parent_impurity <= 0.0) {
        node->is_leaf = 1;
        node->value = tree_majority(ctx->count_total, k);
        return node;
    }

    usize best_feature = 0;
    f64 best_threshold = 0.0;
    f64 best_impurity = parent_impurity;
    int found = 0;

    usize d = ctx->X->cols;
    usize num_feat = tree_select_features(ctx->cols, d, ctx->max_features, ctx->rng);
    TreeSplitPoint *points = (TreeSplitPoint *)arena_alloc(ctx->a, n * sizeof(TreeSplitPoint));

    for (usize fi = 0; fi < num_feat; fi++) {
        usize j = ctx->cols[fi];
        for (usize i = 0; i < n; i++) {
            points[i].value = mat_get(ctx->X, idx[i], j);
            points[i].idx = idx[i];
        }
        qsort(points, n, sizeof(TreeSplitPoint), tree_splitpoint_cmp);

        for (usize c = 0; c < k; c++) {
            ctx->count_left[c] = 0;
            ctx->count_right[c] = ctx->count_total[c];
        }

        for (usize i = 0; i + 1 < n; i++) {
            usize c = (usize)ctx->y->data[points[i].idx];
            ctx->count_left[c]++;
            ctx->count_right[c]--;
            if (points[i].value == points[i + 1].value) continue;

            usize nl = i + 1;
            usize nr = n - nl;
            f64 imp_l = tree_impurity_counts(ctx->count_left, k, nl, ctx->criterion);
            f64 imp_r = tree_impurity_counts(ctx->count_right, k, nr, ctx->criterion);
            f64 weighted = ((f64)nl / (f64)n) * imp_l + ((f64)nr / (f64)n) * imp_r;

            if (weighted < best_impurity - 1e-12) {
                best_impurity = weighted;
                best_feature = j;
                best_threshold = (points[i].value + points[i + 1].value) / 2.0;
                found = 1;
            }
        }
    }

    if (!found) {
        node->is_leaf = 1;
        node->value = tree_majority(ctx->count_total, k);
        return node;
    }

    usize *left_idx, *right_idx, nl, nr;
    tree_partition(ctx->a, ctx->X, idx, n, best_feature, best_threshold, &left_idx, &nl, &right_idx, &nr);

    node->is_leaf = 0;
    node->feature = best_feature;
    node->threshold = best_threshold;
    node->left = tree_build_classify(ctx, left_idx, nl, depth + 1);
    node->right = tree_build_classify(ctx, right_idx, nr, depth + 1);
    return node;
}

Tree tree_fit_classify(Arena *a, const Mat *X, const Vec *y, usize num_classes,
                       TreeCriterion criterion, usize max_depth, usize min_samples_split,
                       usize max_features, Rng *rng) {
    assert(X->rows == y->len);
    assert(X->rows > 0);
    assert(criterion == TREE_GINI || criterion == TREE_ENTROPY);

    usize *idx = (usize *)arena_alloc(a, X->rows * sizeof(usize));
    for (usize i = 0; i < X->rows; i++) idx[i] = i;

    TreeClassifyCtx ctx;
    ctx.X = X;
    ctx.y = y;
    ctx.num_classes = num_classes;
    ctx.criterion = criterion;
    ctx.max_depth = max_depth;
    ctx.min_samples_split = min_samples_split;
    ctx.max_features = max_features;
    ctx.rng = rng;
    ctx.a = a;
    ctx.count_total = (usize *)arena_alloc(a, num_classes * sizeof(usize));
    ctx.count_left = (usize *)arena_alloc(a, num_classes * sizeof(usize));
    ctx.count_right = (usize *)arena_alloc(a, num_classes * sizeof(usize));
    ctx.cols = (usize *)arena_alloc(a, X->cols * sizeof(usize));
    for (usize j = 0; j < X->cols; j++) ctx.cols[j] = j;

    Tree model;
    model.criterion = criterion;
    model.root = tree_build_classify(&ctx, idx, X->rows, 0);
    return model;
}

typedef struct {
    const Mat *X;
    const Vec *y;
    usize max_depth;
    usize min_samples_split;
    usize max_features;
    Rng *rng;
    Arena *a;
    usize *cols;
} TreeRegressCtx;

static f64 tree_variance(f64 sum, f64 sumsq, usize n) {
    if (n == 0) return 0.0;
    f64 mean = sum / (f64)n;
    f64 var = sumsq / (f64)n - mean * mean;
    return var < 0.0 ? 0.0 : var;
}

static TreeNode *tree_build_regress(TreeRegressCtx *ctx, usize *idx, usize n, usize depth) {
    f64 sum = 0.0, sumsq = 0.0;
    for (usize i = 0; i < n; i++) {
        f64 v = ctx->y->data[idx[i]];
        sum += v;
        sumsq += v * v;
    }
    f64 mean = sum / (f64)n;
    f64 parent_impurity = tree_variance(sum, sumsq, n);

    TreeNode *node = tree_node_alloc(ctx->a);
    node->left = NULL;
    node->right = NULL;

    if (depth >= ctx->max_depth || n < ctx->min_samples_split || parent_impurity <= 0.0) {
        node->is_leaf = 1;
        node->value = mean;
        return node;
    }

    usize best_feature = 0;
    f64 best_threshold = 0.0;
    f64 best_impurity = parent_impurity;
    int found = 0;

    usize d = ctx->X->cols;
    usize num_feat = tree_select_features(ctx->cols, d, ctx->max_features, ctx->rng);
    TreeSplitPoint *points = (TreeSplitPoint *)arena_alloc(ctx->a, n * sizeof(TreeSplitPoint));

    for (usize fi = 0; fi < num_feat; fi++) {
        usize j = ctx->cols[fi];
        for (usize i = 0; i < n; i++) {
            points[i].value = mat_get(ctx->X, idx[i], j);
            points[i].idx = idx[i];
        }
        qsort(points, n, sizeof(TreeSplitPoint), tree_splitpoint_cmp);

        f64 sum_left = 0.0, sumsq_left = 0.0;
        for (usize i = 0; i + 1 < n; i++) {
            f64 v = ctx->y->data[points[i].idx];
            sum_left += v;
            sumsq_left += v * v;
            if (points[i].value == points[i + 1].value) continue;

            usize nl = i + 1;
            usize nr = n - nl;
            f64 sum_right = sum - sum_left;
            f64 sumsq_right = sumsq - sumsq_left;
            f64 imp_l = tree_variance(sum_left, sumsq_left, nl);
            f64 imp_r = tree_variance(sum_right, sumsq_right, nr);
            f64 weighted = ((f64)nl / (f64)n) * imp_l + ((f64)nr / (f64)n) * imp_r;

            if (weighted < best_impurity - 1e-12) {
                best_impurity = weighted;
                best_feature = j;
                best_threshold = (points[i].value + points[i + 1].value) / 2.0;
                found = 1;
            }
        }
    }

    if (!found) {
        node->is_leaf = 1;
        node->value = mean;
        return node;
    }

    usize *left_idx, *right_idx, nl, nr;
    tree_partition(ctx->a, ctx->X, idx, n, best_feature, best_threshold, &left_idx, &nl, &right_idx, &nr);

    node->is_leaf = 0;
    node->feature = best_feature;
    node->threshold = best_threshold;
    node->left = tree_build_regress(ctx, left_idx, nl, depth + 1);
    node->right = tree_build_regress(ctx, right_idx, nr, depth + 1);
    return node;
}

Tree tree_fit_regress(Arena *a, const Mat *X, const Vec *y, usize max_depth, usize min_samples_split,
                      usize max_features, Rng *rng) {
    assert(X->rows == y->len);
    assert(X->rows > 0);

    usize *idx = (usize *)arena_alloc(a, X->rows * sizeof(usize));
    for (usize i = 0; i < X->rows; i++) idx[i] = i;

    TreeRegressCtx ctx;
    ctx.X = X;
    ctx.y = y;
    ctx.max_depth = max_depth;
    ctx.min_samples_split = min_samples_split;
    ctx.max_features = max_features;
    ctx.rng = rng;
    ctx.a = a;
    ctx.cols = (usize *)arena_alloc(a, X->cols * sizeof(usize));
    for (usize j = 0; j < X->cols; j++) ctx.cols[j] = j;

    Tree model;
    model.criterion = TREE_VARIANCE;
    model.root = tree_build_regress(&ctx, idx, X->rows, 0);
    return model;
}

static f64 tree_predict_row(const TreeNode *node, const f64 *row) {
    while (!node->is_leaf) {
        node = row[node->feature] <= node->threshold ? node->left : node->right;
    }
    return node->value;
}

Vec tree_predict(Arena *a, const Tree *model, const Mat *X) {
    Vec pred = vec_alloc(a, X->rows);
    for (usize i = 0; i < X->rows; i++) {
        pred.data[i] = tree_predict_row(model->root, mat_row(X, i));
    }
    return pred;
}
