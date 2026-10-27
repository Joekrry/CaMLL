#include "test.h"
#include "src/dense.h"
#include "src/matrix.h"
#include "alloc/arena.h"

#include <math.h>

/* L = sum(dout .* dense_forward(layer, X)), with dout = dL/dout. Finite
   differences on L check dense_backward's gradients. */
static f64 forward_loss(Arena *a, const Dense *layer, const Mat *X, const Mat *dout) {
    Mat out = dense_forward(a, layer, X);
    f64 l = 0.0;
    for (usize i = 0; i < out.rows * out.cols; i++) {
        l += dout->data[i] * out.data[i];
    }
    return l;
}

static void check_backward(Arena *a, Dense *layer, Mat *X, const Mat *dout, f64 eps, f64 tol) {
    Mat out = dense_forward(a, layer, X);
    Mat grad_w;
    Vec grad_b;
    Mat dX = dense_backward(a, layer, X, &out, dout, &grad_w, &grad_b);

    for (usize i = 0; i < layer->weights.rows * layer->weights.cols; i++) {
        f64 orig = layer->weights.data[i];
        layer->weights.data[i] = orig + eps;
        f64 lp = forward_loss(a, layer, X, dout);
        layer->weights.data[i] = orig - eps;
        f64 lm = forward_loss(a, layer, X, dout);
        layer->weights.data[i] = orig;
        CHECK_NEAR((lp - lm) / (2.0 * eps), grad_w.data[i], tol);
    }

    for (usize i = 0; i < layer->bias.len; i++) {
        f64 orig = layer->bias.data[i];
        layer->bias.data[i] = orig + eps;
        f64 lp = forward_loss(a, layer, X, dout);
        layer->bias.data[i] = orig - eps;
        f64 lm = forward_loss(a, layer, X, dout);
        layer->bias.data[i] = orig;
        CHECK_NEAR((lp - lm) / (2.0 * eps), grad_b.data[i], tol);
    }

    for (usize i = 0; i < X->rows * X->cols; i++) {
        f64 orig = X->data[i];
        X->data[i] = orig + eps;
        f64 lp = forward_loss(a, layer, X, dout);
        X->data[i] = orig - eps;
        f64 lm = forward_loss(a, layer, X, dout);
        X->data[i] = orig;
        CHECK_NEAR((lp - lm) / (2.0 * eps), dX.data[i], tol);
    }
}

int main(void) {
    Arena a = arena_create(0);
    Rng r = rng_seed(42);

    usize n = 3, in_dim = 2;
    Mat X = mat_alloc(&a, n, in_dim);
    f64 xs[6] = {0.5, -0.3, 1.0, 0.2, -0.7, 0.8};
    for (usize i = 0; i < n * in_dim; i++) X.data[i] = xs[i];

    f64 eps = 1e-5, tol = 1e-4;

    {
        usize out_dim = 2;
        Dense layer = dense_create(&a, in_dim, out_dim, DENSE_LINEAR, &r);
        Mat out = dense_forward(&a, &layer, &X);
        CHECK(out.rows == n && out.cols == out_dim);

        Mat dout = mat_alloc(&a, n, out_dim);
        f64 douts[6] = {1.0, -1.0, 0.5, 2.0, -0.5, 1.5};
        for (usize i = 0; i < n * out_dim; i++) dout.data[i] = douts[i];
        check_backward(&a, &layer, &X, &dout, eps, tol);
    }

    {
        usize out_dim = 1;
        Dense layer = dense_create(&a, in_dim, out_dim, DENSE_SIGMOID, &r);
        Mat out = dense_forward(&a, &layer, &X);
        CHECK(out.rows == n && out.cols == out_dim);
        for (usize i = 0; i < n; i++) {
            CHECK(out.data[i] > 0.0 && out.data[i] < 1.0);
        }

        Mat dout = mat_alloc(&a, n, out_dim);
        f64 douts[3] = {1.0, -0.5, 0.3};
        for (usize i = 0; i < n; i++) dout.data[i] = douts[i];
        check_backward(&a, &layer, &X, &dout, eps, tol);
    }

    {
        usize out_dim = 3;
        Dense layer = dense_create(&a, in_dim, out_dim, DENSE_SOFTMAX, &r);
        Mat out = dense_forward(&a, &layer, &X);
        CHECK(out.rows == n && out.cols == out_dim);
        for (usize i = 0; i < n; i++) {
            f64 sum = 0.0;
            for (usize c = 0; c < out_dim; c++) {
                f64 p = mat_get(&out, i, c);
                CHECK(p >= 0.0 && p <= 1.0);
                sum += p;
            }
            CHECK_NEAR(sum, 1.0, 1e-9);
        }

        Mat dout = mat_alloc(&a, n, out_dim);
        f64 douts[9] = {1.0, -1.0, 0.5, 0.2, -0.3, 0.7, -0.6, 1.2, -0.4};
        for (usize i = 0; i < n * out_dim; i++) dout.data[i] = douts[i];
        check_backward(&a, &layer, &X, &dout, eps, tol);
    }

    arena_destroy(&a);
    return TEST_SUMMARY("dense");
}
