#ifndef CAMLL_SRC_DENSE_H
#define CAMLL_SRC_DENSE_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

typedef enum {
    DENSE_LINEAR,
    DENSE_SIGMOID,
    DENSE_SOFTMAX,
} DenseActivation;

/* One feed-forward layer: z = X * weights^T + bias, then the activation.
   weights is out_dim rows x in_dim cols; bias has length out_dim. */
typedef struct {
    Mat weights;
    Vec bias;
    DenseActivation activation;
} Dense;

Dense dense_create(Arena *a, usize in_dim, usize out_dim, DenseActivation activation, Rng *r);

Mat dense_forward(Arena *a, const Dense *layer, const Mat *X);

/* X and out are the matching forward call's input/output; dout is dL/dout.
   Writes unaveraged weight/bias gradients into grad_w/grad_b, returns dL/dX. */
Mat dense_backward(Arena *a, const Dense *layer, const Mat *X, const Mat *out, const Mat *dout, Mat *grad_w, Vec *grad_b);

#endif /* CAMLL_SRC_DENSE_H */
