#include "test.h"
#include "src/kmeans.h"
#include "src/matrix.h"
#include "alloc/arena.h"

static void set_row(Mat *m, usize r, f64 x, f64 y) {
    mat_set(m, r, 0, x);
    mat_set(m, r, 1, y);
}

int main(void) {
    Arena a = arena_create(0);

    /* Three well-separated 2x2 "plus" clusters, centered at (0,0), (10,0)
       and (0,10); each point is exactly 1 unit from its center. Expected
       inertia: 3 * 4 * 1 = 12. */
    usize n = 12;
    Mat X = mat_alloc(&a, n, 2);
    set_row(&X, 0, -1.0, 0.0);
    set_row(&X, 1, 1.0, 0.0);
    set_row(&X, 2, 0.0, -1.0);
    set_row(&X, 3, 0.0, 1.0);
    set_row(&X, 4, 9.0, 0.0);
    set_row(&X, 5, 11.0, 0.0);
    set_row(&X, 6, 10.0, -1.0);
    set_row(&X, 7, 10.0, 1.0);
    set_row(&X, 8, -1.0, 10.0);
    set_row(&X, 9, 1.0, 10.0);
    set_row(&X, 10, 0.0, 9.0);
    set_row(&X, 11, 0.0, 11.0);

    Rng rng = rng_seed(42);
    KMeans model = kmeans_fit(&a, &X, 3, 50, &rng);
    CHECK(model.k == 3);
    CHECK(model.centroids.rows == 3);

    Vec pred = kmeans_predict(&a, &model, &X);
    CHECK(pred.len == n);

    /* Rows in the same true cluster land on the same label; different
       clusters land on different labels. Cluster membership is rows 0-3,
       4-7, 8-11. */
    CHECK(pred.data[0] == pred.data[1]);
    CHECK(pred.data[0] == pred.data[2]);
    CHECK(pred.data[0] == pred.data[3]);
    CHECK(pred.data[4] == pred.data[5]);
    CHECK(pred.data[4] == pred.data[6]);
    CHECK(pred.data[4] == pred.data[7]);
    CHECK(pred.data[8] == pred.data[9]);
    CHECK(pred.data[8] == pred.data[10]);
    CHECK(pred.data[8] == pred.data[11]);
    CHECK(pred.data[0] != pred.data[4]);
    CHECK(pred.data[0] != pred.data[8]);
    CHECK(pred.data[4] != pred.data[8]);

    /* Each converged centroid sits exactly on one true cluster center. */
    f64 *c0 = mat_row(&model.centroids, (usize)pred.data[0]);
    f64 *c4 = mat_row(&model.centroids, (usize)pred.data[4]);
    f64 *c8 = mat_row(&model.centroids, (usize)pred.data[8]);
    CHECK_NEAR(c0[0], 0.0, 1e-9);
    CHECK_NEAR(c0[1], 0.0, 1e-9);
    CHECK_NEAR(c4[0], 10.0, 1e-9);
    CHECK_NEAR(c4[1], 0.0, 1e-9);
    CHECK_NEAR(c8[0], 0.0, 1e-9);
    CHECK_NEAR(c8[1], 10.0, 1e-9);

    CHECK_NEAR(kmeans_inertia(&model, &X), 12.0, 1e-9);

    /* k=1 collapses to the mean of all rows. */
    Mat Xsq = mat_alloc(&a, 4, 2);
    set_row(&Xsq, 0, 0.0, 0.0);
    set_row(&Xsq, 1, 2.0, 0.0);
    set_row(&Xsq, 2, 0.0, 2.0);
    set_row(&Xsq, 3, 2.0, 2.0);
    Rng rng1 = rng_seed(7);
    KMeans model1 = kmeans_fit(&a, &Xsq, 1, 50, &rng1);
    CHECK_NEAR(mat_row(&model1.centroids, 0)[0], 1.0, 1e-9);
    CHECK_NEAR(mat_row(&model1.centroids, 0)[1], 1.0, 1e-9);
    CHECK_NEAR(kmeans_inertia(&model1, &Xsq), 8.0, 1e-9);

    /* k == n: every point becomes its own centroid, zero inertia. */
    Rng rng2 = rng_seed(99);
    KMeans model2 = kmeans_fit(&a, &Xsq, 4, 50, &rng2);
    CHECK_NEAR(kmeans_inertia(&model2, &Xsq), 0.0, 1e-9);

    arena_destroy(&a);
    return TEST_SUMMARY("kmeans");
}
