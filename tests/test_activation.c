#include "test.h"
#include "src/activation.h"
#include "alloc/arena.h"

int main(void) {
    Arena a = arena_create(0);

    CHECK_NEAR(sigmoid(0.0), 0.5, 1e-12);
    CHECK(sigmoid(50.0) > 0.999);
    CHECK(sigmoid(-50.0) < 0.001);

    CHECK_NEAR(sigmoid_deriv(0.5), 0.25, 1e-12);
    CHECK_NEAR(sigmoid_deriv(sigmoid(0.0)), 0.25, 1e-12);

    Vec x = vec_alloc(&a, 3);
    x.data[0] = -1.0; x.data[1] = 0.0; x.data[2] = 1.0;

    Vec s = vec_sigmoid(&a, &x);
    CHECK(s.len == 3);
    CHECK_NEAR(s.data[1], 0.5, 1e-12);
    CHECK_NEAR(s.data[0], sigmoid(-1.0), 1e-12);
    CHECK_NEAR(s.data[2], sigmoid(1.0), 1e-12);

    /* softmax sums to 1 and preserves relative ordering */
    Vec sm = vec_softmax(&a, &x);
    f64 sum = sm.data[0] + sm.data[1] + sm.data[2];
    CHECK_NEAR(sum, 1.0, 1e-12);
    CHECK(sm.data[0] < sm.data[1]);
    CHECK(sm.data[1] < sm.data[2]);

    /* shift invariance: softmax(x + c) == softmax(x) */
    Vec xs = vec_alloc(&a, 3);
    xs.data[0] = x.data[0] + 1000.0;
    xs.data[1] = x.data[1] + 1000.0;
    xs.data[2] = x.data[2] + 1000.0;
    Vec sms = vec_softmax(&a, &xs);
    CHECK_NEAR(sms.data[0], sm.data[0], 1e-9);
    CHECK_NEAR(sms.data[1], sm.data[1], 1e-9);
    CHECK_NEAR(sms.data[2], sm.data[2], 1e-9);

    /* uniform input gives uniform distribution */
    Vec u = vec_zeros(&a, 4);
    Vec su = vec_softmax(&a, &u);
    CHECK_NEAR(su.data[0], 0.25, 1e-12);
    CHECK_NEAR(su.data[3], 0.25, 1e-12);

    arena_destroy(&a);
    return TEST_SUMMARY("activation");
}
