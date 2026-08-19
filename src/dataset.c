#include "dataset.h"

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int line_is_blank(const char *b, const char *e) {
    for (const char *q = b; q < e; q++) {
        if (!isspace((unsigned char)*q)) return 0;
    }
    return 1;
}

static usize count_fields(const char *b, const char *e) {
    usize fields = 1;
    for (const char *q = b; q < e; q++) {
        if (*q == ',') fields++;
    }
    return fields;
}

Dataset dataset_from_csv(Arena *a, const char *text, usize label_col) {
    usize rows = 0, cols = 0;
    for (const char *p = text; *p;) {
        const char *nl = strchr(p, '\n');
        const char *end = nl ? nl : p + strlen(p);
        if (!line_is_blank(p, end)) {
            if (rows == 0) cols = count_fields(p, end);
            rows++;
        }
        if (!nl) break;
        p = nl + 1;
    }
    assert(cols >= 2);          /* at least one feature plus the label */
    assert(label_col < cols);

    Dataset d;
    d.X = mat_alloc(a, rows, cols - 1);
    d.y = vec_alloc(a, rows);

    usize r = 0;
    for (const char *p = text; *p && r < rows;) {
        const char *nl = strchr(p, '\n');
        const char *end = nl ? nl : p + strlen(p);
        if (!line_is_blank(p, end)) {
            const char *cur = p;
            usize out = 0;
            for (usize c = 0; c < cols; c++) {
                char *tok_end;
                f64 v = strtod(cur, &tok_end);
                assert(tok_end != cur);     /* a number was actually read */
                if (c == label_col) d.y.data[r] = v;
                else mat_set(&d.X, r, out++, v);
                cur = tok_end;
                while (cur < end && *cur != ',') cur++;
                if (cur < end) cur++;
            }
            r++;
        }
        if (!nl) break;
        p = nl + 1;
    }
    return d;
}

Dataset dataset_load_csv(Arena *a, const char *path, usize label_col) {
    FILE *f = fopen(path, "rb");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    assert(size >= 0);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)arena_alloc(a, (usize)size + 1);
    usize n = fread(buf, 1, (usize)size, f);
    buf[n] = '\0';
    fclose(f);
    return dataset_from_csv(a, buf, label_col);
}

Split dataset_split(Arena *a, const Dataset *d, f64 train_frac, Rng *rng) {
    assert(train_frac >= 0.0 && train_frac <= 1.0);
    usize n = d->X.rows;
    usize cols = d->X.cols;

    usize *idx = (usize *)arena_alloc(a, n * sizeof(usize));
    for (usize i = 0; i < n; i++) idx[i] = i;
    for (usize i = n; i > 1; i--) {
        usize j = (usize)rng_uniform(rng, 0.0, (f64)i);
        if (j >= i) j = i - 1;      /* guard the half-open upper bound */
        usize t = idx[i - 1];
        idx[i - 1] = idx[j];
        idx[j] = t;
    }

    usize n_train = (usize)((f64)n * train_frac);
    Split s;
    s.train.X = mat_alloc(a, n_train, cols);
    s.train.y = vec_alloc(a, n_train);
    s.test.X = mat_alloc(a, n - n_train, cols);
    s.test.y = vec_alloc(a, n - n_train);

    for (usize i = 0; i < n; i++) {
        usize src = idx[i];
        Dataset *dst = (i < n_train) ? &s.train : &s.test;
        usize row = (i < n_train) ? i : i - n_train;
        memcpy(mat_row(&dst->X, row), mat_row(&d->X, src), cols * sizeof(f64));
        dst->y.data[row] = d->y.data[src];
    }
    return s;
}

Standardizer standardizer_fit(Arena *a, const Mat *X) {
    usize rows = X->rows, cols = X->cols;
    Standardizer s;
    s.mean = vec_zeros(a, cols);
    s.std = vec_zeros(a, cols);
    for (usize j = 0; j < cols; j++) {
        f64 sum = 0.0;
        for (usize i = 0; i < rows; i++) sum += mat_get(X, i, j);
        f64 mean = sum / (f64)rows;
        f64 var = 0.0;
        for (usize i = 0; i < rows; i++) {
            f64 diff = mat_get(X, i, j) - mean;
            var += diff * diff;
        }
        var /= (f64)rows;                       /* population variance */
        s.mean.data[j] = mean;
        s.std.data[j] = (var > 0.0) ? sqrt(var) : 1.0;  /* keep constant cols finite */
    }
    return s;
}

Mat standardizer_apply(Arena *a, const Standardizer *s, const Mat *X) {
    assert(X->cols == s->mean.len);
    Mat out = mat_alloc(a, X->rows, X->cols);
    for (usize i = 0; i < X->rows; i++) {
        for (usize j = 0; j < X->cols; j++) {
            f64 v = (mat_get(X, i, j) - s->mean.data[j]) / s->std.data[j];
            mat_set(&out, i, j, v);
        }
    }
    return out;
}
