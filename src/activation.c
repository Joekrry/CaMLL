#include "activation.h"
#include <math.h>

f64 sigmoid(f64 x) {
    if (x >=0.0){
        f64 z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    f64 z = exp(x);
    return z / (1.0 + z);
}

Vec vec_sigmoid(Arena *a, const Vec *x) {
    Vec r = vec_alloc(a, x ->len);
    for(usize i = 0; i < x->len; i++) {
        r.data[i] = sigmoid(x->data[i]);
    }
    return r;
}

f64 sigmoid_deriv(f64 s) {
    return s * (1.0 - s);
}

Vec vec_softmax(Arena *a, const Vec *x) {
    Vec r = vec_alloc(a, x->len);

    f64 max = x->data[0];
    for (usize i = 1; i < x->len; i++) {
        if (x->data[i] > max) max = x->data[i];
    }

    f64 sum = 0.0;
    for (usize i = 0; i < x->len; i++) {
        f64 e = exp(x->data[i] - max);
        r.data[i] = e;
        sum += e;
    }

    for (usize i = 0; i < x->len; i++) {
        r.data[i] /= sum;
    }

    return r;
}
