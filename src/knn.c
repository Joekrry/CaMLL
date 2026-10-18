#include "knn.h"

#include <assert.h>

KNN knn_fit(Arena *a, const Mat *X, const Vec *y, usize k) {
    assert(X->rows == y->len);
    assert(k > 0 && k <= X->rows);
    KNN model;
    model.X = mat_copy(a, X);
    model.y = vec_copy(a, y);
    model.k = k;
    return model;
}

/* idx/dist hold the k smallest distances found so far, sorted ascending. */
static void knn_neighbors(const KNN *model, const f64 *query, usize *idx, f64 *dist) {
    usize n = model->X.rows;
    usize d = model->X.cols;
    usize k = model->k;

    for (usize s = 0; s < k; s++) {
        dist[s] = -1.0;
    }

    for (usize i = 0; i < n; i++) {
        const f64 *row = mat_row(&model->X, i);
        f64 di = 0.0;
        for (usize j = 0; j < d; j++) {
            f64 diff = row[j] - query[j];
            di += diff * diff;
        }

        usize pos = k;
        for (usize s = 0; s < k; s++) {
            if (dist[s] < 0.0 || di < dist[s]) {
                pos = s;
                break;
            }
        }
        if (pos == k) {
            continue;
        }
        for (usize s = k - 1; s > pos; s--) {
            dist[s] = dist[s - 1];
            idx[s] = idx[s - 1];
        }
        dist[pos] = di;
        idx[pos] = i;
    }
}

Vec knn_predict_regress(Arena *a, const KNN *model, const Mat *X) {
    assert(X->cols == model->X.cols);
    usize k = model->k;
    usize *idx = (usize *)arena_alloc(a, k * sizeof(usize));
    f64 *dist = (f64 *)arena_alloc(a, k * sizeof(f64));

    Vec pred = vec_alloc(a, X->rows);
    for (usize i = 0; i < X->rows; i++) {
        knn_neighbors(model, mat_row(X, i), idx, dist);
        f64 sum = 0.0;
        for (usize s = 0; s < k; s++) {
            sum += model->y.data[idx[s]];
        }
        pred.data[i] = sum / (f64)k;
    }
    return pred;
}

Vec knn_predict_classify(Arena *a, const KNN *model, const Mat *X) {
    assert(X->cols == model->X.cols);
    usize k = model->k;
    usize *idx = (usize *)arena_alloc(a, k * sizeof(usize));
    f64 *dist = (f64 *)arena_alloc(a, k * sizeof(f64));

    Vec pred = vec_alloc(a, X->rows);
    for (usize i = 0; i < X->rows; i++) {
        knn_neighbors(model, mat_row(X, i), idx, dist);

        f64 best_label = 0.0;
        usize best_count = 0;
        for (usize s = 0; s < k; s++) {
            f64 label = model->y.data[idx[s]];
            usize count = 0;
            for (usize t = 0; t < k; t++) {
                if (model->y.data[idx[t]] == label) {
                    count++;
                }
            }
            if (count > best_count || (count == best_count && label < best_label)) {
                best_count = count;
                best_label = label;
            }
        }
        pred.data[i] = best_label;
    }
    return pred;
}
