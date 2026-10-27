#ifndef CAMLL_SRC_MLP_H
#define CAMLL_SRC_MLP_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"
#include "dense.h"

typedef struct {
    Dense *layers;
    usize num_layers;
} MLP;

/* layer_dims has num_layers + 1 entries (input, each hidden, output dim). */
MLP mlp_create(Arena *a, const usize *layer_dims, usize num_layers, const DenseActivation *activations, Rng *r);

Mat mlp_forward(Arena *a, const MLP *net, const Mat *X);

/* Last layer's activation picks the loss: DENSE_LINEAR -> MSE, DENSE_SIGMOID
   -> binary cross-entropy (y in {0, 1}), DENSE_SOFTMAX -> multinomial
   cross-entropy (y is a class index). l2 regularizes weights only. */
void mlp_fit_gd(Arena *a, MLP *net, const Mat *X, const Vec *y, f64 lr, usize epochs, f64 l2);

/* argmax for DENSE_SOFTMAX, 0.5 threshold for DENSE_SIGMOID. Invalid on a
   DENSE_LINEAR output; use mlp_forward instead. */
Vec mlp_predict(Arena *a, const MLP *net, const Mat *X);

#endif /* CAMLL_SRC_MLP_H */
