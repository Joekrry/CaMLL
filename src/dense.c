#include "dense.h"
#include "activation.h"

#include <assert.h>
#include <math.h>

Dense dense_create(Arena *a, usize in_dim, usize out_dim, DenseActivation activation, Rng *r) {
    Dense layer;
    f64 scale = 1.0 / sqrt((f64)in_dim);
    layer.weights = mat_rand_normal(a, out_dim, in_dim, r, 0.0, scale);
    layer.bias = vec_zeros(a, out_dim);
    layer.activation = activation;
    return layer;
}

Mat dense_forward(Arena *a, const Dense *layer, const Mat *X) {
    assert(X->cols == layer->weights.cols);
    usize n = X->rows;
    usize out_dim = layer->weights.rows;

    Mat wt = mat_transpose(a, &layer->weights);
    Mat z = mat_matmul(a, X, &wt);
    for (usize i = 0; i < n; i++) {
        f64 *zrow = mat_row(&z, i);
        for (usize c = 0; c < out_dim; c++) {
            zrow[c] += layer->bias.data[c];
        }
    }

    if (layer->activation == DENSE_LINEAR) {
        return z;
    }

    Mat out = mat_alloc(a, n, out_dim);
    if (layer->activation == DENSE_SIGMOID) {
        for (usize i = 0; i < n * out_dim; i++) {
            out.data[i] = sigmoid(z.data[i]);
        }
        return out;
    }

    assert(layer->activation == DENSE_SOFTMAX);
    for (usize i = 0; i < n; i++) {
        Vec row = { .len = out_dim, .data = mat_row(&z, i) };
        Vec sm = vec_softmax(a, &row);
        f64 *orow = mat_row(&out, i);
        for (usize c = 0; c < out_dim; c++) {
            orow[c] = sm.data[c];
        }
    }
    return out;
}

Mat dense_backward(Arena *a, const Dense *layer, const Mat *X, const Mat *out, const Mat *dout, Mat *grad_w, Vec *grad_b) {
    usize n = X->rows;
    usize in_dim = X->cols;
    usize out_dim = layer->weights.rows;
    assert(in_dim == layer->weights.cols);
    assert(out->rows == n && out->cols == out_dim);
    assert(dout->rows == n && dout->cols == out_dim);

    Mat dz = mat_alloc(a, n, out_dim);
    if (layer->activation == DENSE_LINEAR) {
        for (usize i = 0; i < n * out_dim; i++) {
            dz.data[i] = dout->data[i];
        }
    } else if (layer->activation == DENSE_SIGMOID) {
        for (usize i = 0; i < n * out_dim; i++) {
            dz.data[i] = dout->data[i] * sigmoid_deriv(out->data[i]);
        }
    } else {
        assert(layer->activation == DENSE_SOFTMAX);
        for (usize i = 0; i < n; i++) {
            const f64 *orow = mat_row(out, i);
            const f64 *dorow = mat_row(dout, i);
            f64 *dzrow = mat_row(&dz, i);
            f64 dot = 0.0;
            for (usize c = 0; c < out_dim; c++) {
                dot += dorow[c] * orow[c];
            }
            for (usize c = 0; c < out_dim; c++) {
                dzrow[c] = orow[c] * (dorow[c] - dot);
            }
        }
    }

    *grad_w = mat_zeros(a, out_dim, in_dim);
    *grad_b = vec_zeros(a, out_dim);
    for (usize i = 0; i < n; i++) {
        const f64 *xrow = mat_row(X, i);
        const f64 *dzrow = mat_row(&dz, i);
        for (usize c = 0; c < out_dim; c++) {
            grad_b->data[c] += dzrow[c];
            f64 *gwrow = mat_row(grad_w, c);
            for (usize j = 0; j < in_dim; j++) {
                gwrow[j] += dzrow[c] * xrow[j];
            }
        }
    }

    return mat_matmul(a, &dz, &layer->weights);
}
