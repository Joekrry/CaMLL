#include "test.h"
#include "src/mlp.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);
    Rng r = rng_seed(7);

    /* XOR is not linearly separable; a passing result exercises hidden-layer backprop. */
    usize n = 4, in_dim = 2;
    Mat X = mat_alloc(&a, n, in_dim);
    Vec y = vec_alloc(&a, n);
    f64 xs[8] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    f64 ys[4] = {0.0, 1.0, 1.0, 0.0};
    for (usize i = 0; i < n * in_dim; i++) X.data[i] = xs[i];
    for (usize i = 0; i < n; i++) y.data[i] = ys[i];

    usize dims[3] = {in_dim, 4, 1};
    DenseActivation acts[2] = {DENSE_SIGMOID, DENSE_SIGMOID};
    MLP net = mlp_create(&a, dims, 2, acts, &r);
    CHECK(net.num_layers == 2);

    mlp_fit_gd(&a, &net, &X, &y, 0.5, 20000, 0.0);

    Mat proba = mlp_forward(&a, &net, &X);
    CHECK(proba.rows == n && proba.cols == 1);
    for (usize i = 0; i < n; i++) {
        CHECK(proba.data[i] >= 0.0 && proba.data[i] <= 1.0);
    }

    Vec pred = mlp_predict(&a, &net, &X);
    CHECK(pred.len == n);
    CHECK_NEAR(metrics_accuracy(&y, &pred), 1.0, 1e-12);

    /* strong ridge penalty shrinks every layer's weights toward zero */
    Rng r2 = rng_seed(7);
    MLP net_l2 = mlp_create(&a, dims, 2, acts, &r2);
    mlp_fit_gd(&a, &net_l2, &X, &y, 0.5, 20000, 1.0);

    f64 norm = 0.0, norm_l2 = 0.0;
    for (usize li = 0; li < net.num_layers; li++) {
        const Mat *w = &net.layers[li].weights;
        const Mat *w_l2 = &net_l2.layers[li].weights;
        for (usize i = 0; i < w->rows * w->cols; i++) {
            norm += w->data[i] * w->data[i];
            norm_l2 += w_l2->data[i] * w_l2->data[i];
        }
    }
    CHECK(norm_l2 < norm);

    /* three-class softmax, same thresholds as test_softmax.c, through a hidden layer */
    usize nk = 9, k = 3;
    Mat Xk = mat_alloc(&a, nk, 1);
    Vec yk = vec_alloc(&a, nk);
    f64 kxs[9] = {-3.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0};
    for (usize i = 0; i < nk; i++) {
        mat_set(&Xk, i, 0, kxs[i]);
        f64 x = kxs[i];
        yk.data[i] = x < -1.0 ? 0.0 : (x > 1.0 ? 2.0 : 1.0);
    }

    usize kdims[3] = {1, 4, k};
    DenseActivation kacts[2] = {DENSE_SIGMOID, DENSE_SOFTMAX};
    Rng rk = rng_seed(3);
    MLP netk = mlp_create(&a, kdims, 2, kacts, &rk);
    mlp_fit_gd(&a, &netk, &Xk, &yk, 0.5, 5000, 0.0);

    Mat kproba = mlp_forward(&a, &netk, &Xk);
    CHECK(kproba.rows == nk && kproba.cols == k);
    for (usize i = 0; i < nk; i++) {
        f64 sum = 0.0;
        for (usize c = 0; c < k; c++) {
            f64 p = mat_get(&kproba, i, c);
            CHECK(p >= 0.0 && p <= 1.0);
            sum += p;
        }
        CHECK_NEAR(sum, 1.0, 1e-9);
    }

    Vec kpred = mlp_predict(&a, &netk, &Xk);
    CHECK_NEAR(metrics_accuracy(&yk, &kpred), 1.0, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("mlp");
}
