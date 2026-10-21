#ifndef CAMLL_SRC_MLP_H
#define CAMLL_SRC_MLP_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"
#include "dense.h"

/* A stack of dense.h layers, each layer's output feeding the next layer's
   input. */
typedef struct {
    Dense *layers;
    usize num_layers;
} MLP;

/* layer_dims has num_layers + 1 entries: input dim, each hidden dim, output
   dim. activations has num_layers entries, one per layer. */
MLP mlp_create(Arena *a, const usize *layer_dims, usize num_layers, const DenseActivation *activations, Rng *r);

/* Runs every layer in sequence, returning the last layer's output. */
Mat mlp_forward(Arena *a, const MLP *net, const Mat *X);

/* Batch gradient descent, backpropagating through the whole stack. The last
   layer's activation decides the loss paired with it via the standard
   combined activation+loss gradient (output - target): DENSE_LINEAR -> MSE
   (out_dim 1), DENSE_SIGMOID -> binary cross-entropy (out_dim 1, y in
   {0.0, 1.0}), DENSE_SOFTMAX -> multinomial cross-entropy (out_dim = number
   of classes, y holds class indices). l2 regularizes every layer's weights;
   biases are left unregularized. */
void mlp_fit_gd(Arena *a, MLP *net, const Mat *X, const Vec *y, f64 lr, usize epochs, f64 l2);

/* Predicted class: argmax over the last layer's output for DENSE_SOFTMAX,
   or a 0.0/1.0 threshold at 0.5 for DENSE_SIGMOID. Not for a DENSE_LINEAR
   (regression) output - call mlp_forward directly instead. */
Vec mlp_predict(Arena *a, const MLP *net, const Mat *X);

#endif /* CAMLL_SRC_MLP_H */
