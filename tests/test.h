#ifndef CAMML_TESTS_TEST_H
#define CAMML_TESTS_TEST_H

#include <stdio.h>
#include <math.h>
#include "types.h"

/* Each test file is its own executable: it defines main(), runs CHECK
   macros, and returns TEST_SUMMARY() as its exit status. */

static i32 g_checks = 0;
static i32 g_failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        g_checks++;                                                     \
        if (!(cond)) {                                                  \
            g_failures++;                                               \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
        }                                                               \
    } while (0)

/* absolute-tolerance compare for floating point values */
#define CHECK_NEAR(a, b, eps)                                           \
    do {                                                                \
        g_checks++;                                                     \
        if (fabs((f64)(a) - (f64)(b)) > (eps)) {                        \
            g_failures++;                                               \
            printf("  FAIL %s:%d  |%s - %s| > %s  (%g vs %g)\n",        \
                   __FILE__, __LINE__, #a, #b, #eps,                    \
                   (f64)(a), (f64)(b));                                 \
        }                                                               \
    } while (0)

static inline int test_summary(const char *name) {
    if (g_failures == 0) {
        printf("PASS %s (%d checks)\n", name, g_checks);
        return 0;
    }
    printf("FAIL %s (%d/%d checks failed)\n", name, g_failures, g_checks);
    return 1;
}

#define TEST_SUMMARY(name) test_summary(name)

#endif /* CAMML_TESTS_TEST_H */
