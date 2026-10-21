#include "mlp.h"

#include <assert.h>

MLP mlp_create(Arena *a, const usize *layer_dims, usize num_layers, const DenseActivation *activations, Rng *r) {
    MLP net;
    net.num_layers = num_layers;
    net.layers = (Dense *)arena_alloc(a, num_layers * sizeof(Dense));
    for (usize i = 0; i < num_layers; i++) {
        net.layers[i] = dense_create(a, layer_dims[i], layer_dims[i + 1], activations[i], r);
    }
    return net;
}

Mat mlp_forward(Arena *a, const MLP *net, const Mat *X) {
    Mat out = *X;
    for (usize i = 0; i < net->num_layers; i++) {
        out = dense_forward(a, &net->layers[i], &out);
    }
    return out;
}

void mlp_fit_gd(Arena *a, MLP *net, const Mat *X, const Vec *y, f64 lr, usize epochs, f64 l2) {
    assert(X->rows == y->len);
    usize n = X->rows;
    usize L = net->num_layers;
    Dense *out_layer = &net->layers[L - 1];
    usize out_dim = out_layer->weights.rows;

    Mat target = mat_zeros(a, n, out_dim);
    if (out_layer->activation == DENSE_SOFTMAX) {
        for (usize i = 0; i < n; i++) {
            usize yi = (usize)y->data[i];
            mat_set(&target, i, yi, 1.0);
        }
    } else {
        assert(out_dim == 1);
        for (usize i = 0; i < n; i++) {
            mat_set(&target, i, 0, y->data[i]);
        }
    }

    Mat *inputs = (Mat *)arena_alloc(a, (L + 1) * sizeof(Mat));
    Mat *outputs = (Mat *)arena_alloc(a, L * sizeof(Mat));
    Mat *grad_w = (Mat *)arena_alloc(a, L * sizeof(Mat));
    Vec *grad_b = (Vec *)arena_alloc(a, L * sizeof(Vec));

    for (usize epoch = 0; epoch < epochs; epoch++) {
        inputs[0] = *X;
        for (usize i = 0; i < L; i++) {
            outputs[i] = dense_forward(a, &net->layers[i], &inputs[i]);
            inputs[i + 1] = outputs[i];
        }

        /* Combined activation+loss gradient is output - target for all three
           pairings (MSE, binary CE, multinomial CE). Pass as DENSE_LINEAR to
           skip dense_backward's activation derivative. */
        Mat dz = mat_sub(a, &outputs[L - 1], &target);
        Dense out_identity = *out_layer;
        out_identity.activation = DENSE_LINEAR;
        Mat dX = dense_backward(a, &out_identity, &inputs[L - 1], &outputs[L - 1], &dz, &grad_w[L - 1], &grad_b[L - 1]);

        for (isize li = (isize)L - 2; li >= 0; li--) {
            dX = dense_backward(a, &net->layers[li], &inputs[li], &outputs[li], &dX, &grad_w[li], &grad_b[li]);
        }

        for (usize li = 0; li < L; li++) {
            Dense *layer = &net->layers[li];
            usize od = layer->weights.rows, id = layer->weights.cols;
            for (usize c = 0; c < od; c++) {
                f64 *wrow = mat_row(&layer->weights, c);
                const f64 *gwrow = mat_row(&grad_w[li], c);
                for (usize j = 0; j < id; j++) {
                    wrow[j] -= lr * (gwrow[j] / (f64)n + l2 * wrow[j]);
                }
                layer->bias.data[c] -= lr * (grad_b[li].data[c] / (f64)n);
            }
        }
    }
}

Vec mlp_predict(Arena *a, const MLP *net, const Mat *X) {
    Mat out = mlp_forward(a, net, X);
    const Dense *last = &net->layers[net->num_layers - 1];
    assert(last->activation != DENSE_LINEAR);

    Vec pred = vec_alloc(a, out.rows);
    if (last->activation == DENSE_SOFTMAX) {
        for (usize i = 0; i < out.rows; i++) {
            const f64 *row = mat_row(&out, i);
            usize best = 0;
            f64 best_v = row[0];
            for (usize c = 1; c < out.cols; c++) {
                if (row[c] > best_v) {
                    best_v = row[c];
                    best = c;
                }
            }
            pred.data[i] = (f64)best;
        }
        return pred;
    }

    assert(out.cols == 1);
    for (usize i = 0; i < out.rows; i++) {
        pred.data[i] = out.data[i] >= 0.5 ? 1.0 : 0.0;
    }
    return pred;
}
