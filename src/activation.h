#ifndef CAMLL_SRC_ACTIVATION_H
#define CAMLL_SRC_ACTIVATION_H

#include "types.h"
#include "alloc/arena.h"
#include "matrix.h"

f64 sigmoid(f64 x);

Vec vec_sigmoid(Arena *a, const Vec *x);

f64 sigmoid_deriv(f64 s);

Vec vec_softmax(Arena *a, const Vec *x);

#endif
