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
