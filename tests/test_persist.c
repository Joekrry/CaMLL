#include "test.h"
#include "src/persist.h"
#include "src/linreg.h"
#include "src/logreg.h"
#include "src/softmax.h"
#include "src/mlp.h"
#include "src/matrix.h"
#include "src/metrics.h"
#include "alloc/arena.h"

#include <stdio.h>

int main(void) {
    Arena a = arena_create(0);

    /* linreg: closed-form fit on exactly-linear data, round-tripped weights
       and predictions must match the original exactly */
    usize n = 6;
    Mat X = mat_alloc(&a, n, 1);
    Vec y = vec_alloc(&a, n);
    for (usize i = 0; i < n; i++) {
        X.data[i] = (f64)i;
        y.data[i] = 2.0 * (f64)i + 1.0;
    }
    LinReg lin = linreg_fit_normal(&a, &X, &y);
    const char *lin_path = "test_persist_linreg.bin";
    persist_linreg_save(&lin, lin_path);
    LinReg lin2 = persist_linreg_load(&a, lin_path);
    remove(lin_path);

    CHECK_NEAR(lin2.bias, lin.bias, 1e-12);
    CHECK(lin2.weights.len == lin.weights.len);
    Vec lin_pred = linreg_predict(&a, &lin, &X);
    Vec lin_pred2 = linreg_predict(&a, &lin2, &X);
    for (usize i = 0; i < n; i++) CHECK_NEAR(lin_pred2.data[i], lin_pred.data[i], 1e-12);

    /* logreg: single-feature separable fit from test_logreg.c */
    Mat Xb = mat_alloc(&a, n, 1);
    Vec yb = vec_alloc(&a, n);
    f64 xs[6] = {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0};
    for (usize i = 0; i < n; i++) {
        mat_set(&Xb, i, 0, xs[i]);
        yb.data[i] = xs[i] > 0.0 ? 1.0 : 0.0;
    }
    LogReg log_model = logreg_fit_gd(&a, &Xb, &yb, 0.5, 3000, 0.0);
    const char *log_path = "test_persist_logreg.bin";
    persist_logreg_save(&log_model, log_path);
    LogReg log_model2 = persist_logreg_load(&a, log_path);
    remove(log_path);

    CHECK_NEAR(log_model2.bias, log_model.bias, 1e-12);
    Vec log_proba = logreg_predict_proba(&a, &log_model, &Xb);
    Vec log_proba2 = logreg_predict_proba(&a, &log_model2, &Xb);
    for (usize i = 0; i < n; i++) CHECK_NEAR(log_proba2.data[i], log_proba.data[i], 1e-12);

    /* softmax: three-class fit from test_softmax.c */
    usize nk = 9, k = 3;
    Mat Xk = mat_alloc(&a, nk, 1);
    Vec yk = vec_alloc(&a, nk);
    f64 kxs[9] = {-3.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0};
    for (usize i = 0; i < nk; i++) {
        mat_set(&Xk, i, 0, kxs[i]);
        f64 x = kxs[i];
        yk.data[i] = x < -1.0 ? 0.0 : (x > 1.0 ? 2.0 : 1.0);
    }
    Softmax sm = softmax_fit_gd(&a, &Xk, &yk, k, 0.5, 3000, 0.0);
    const char *sm_path = "test_persist_softmax.bin";
    persist_softmax_save(&sm, sm_path);
    Softmax sm2 = persist_softmax_load(&a, sm_path);
    remove(sm_path);

    CHECK(sm2.weights.rows == sm.weights.rows && sm2.weights.cols == sm.weights.cols);
    Mat sm_proba = softmax_predict_proba(&a, &sm, &Xk);
    Mat sm_proba2 = softmax_predict_proba(&a, &sm2, &Xk);
    for (usize i = 0; i < nk * k; i++) CHECK_NEAR(sm_proba2.data[i], sm_proba.data[i], 1e-12);

    /* mlp: hidden-sigmoid, output-softmax net from test_mlp.c, round-tripping
       every layer's activation, weights and bias */
    usize dims[3] = {1, 4, k};
    DenseActivation acts[2] = {DENSE_SIGMOID, DENSE_SOFTMAX};
    Rng r = rng_seed(3);
    MLP net = mlp_create(&a, dims, 2, acts, &r);
    mlp_fit_gd(&a, &net, &Xk, &yk, 0.5, 5000, 0.0);
    const char *mlp_path = "test_persist_mlp.bin";
    persist_mlp_save(&net, mlp_path);
    MLP net2 = persist_mlp_load(&a, mlp_path);
    remove(mlp_path);

    CHECK(net2.num_layers == net.num_layers);
    for (usize li = 0; li < net.num_layers; li++) {
        CHECK(net2.layers[li].activation == net.layers[li].activation);
        const Mat *w = &net.layers[li].weights, *w2 = &net2.layers[li].weights;
        CHECK(w2->rows == w->rows && w2->cols == w->cols);
        for (usize i = 0; i < w->rows * w->cols; i++) CHECK_NEAR(w2->data[i], w->data[i], 1e-12);
        const Vec *b = &net.layers[li].bias, *b2 = &net2.layers[li].bias;
        CHECK(b2->len == b->len);
        for (usize i = 0; i < b->len; i++) CHECK_NEAR(b2->data[i], b->data[i], 1e-12);
    }
    Vec mlp_pred = mlp_predict(&a, &net, &Xk);
    Vec mlp_pred2 = mlp_predict(&a, &net2, &Xk);
    CHECK_NEAR(metrics_accuracy(&yk, &mlp_pred2), metrics_accuracy(&yk, &mlp_pred), 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("persist");
}
