#include "persist.h"

#include <assert.h>
#include <stdio.h>

typedef enum {
    PERSIST_LINREG = 1,
    PERSIST_LOGREG = 2,
    PERSIST_SOFTMAX = 3,
    PERSIST_MLP = 4,
} PersistTag;

static void write_u32(FILE *f, u32 v) { assert(fwrite(&v, sizeof(v), 1, f) == 1); }
static void write_usize(FILE *f, usize v) { assert(fwrite(&v, sizeof(v), 1, f) == 1); }
static void write_f64(FILE *f, f64 v) { assert(fwrite(&v, sizeof(v), 1, f) == 1); }

static void write_vec(FILE *f, const Vec *v) {
    write_usize(f, v->len);
    assert(fwrite(v->data, sizeof(f64), v->len, f) == v->len);
}

static void write_mat(FILE *f, const Mat *m) {
    write_usize(f, m->rows);
    write_usize(f, m->cols);
    usize n = m->rows * m->cols;
    assert(fwrite(m->data, sizeof(f64), n, f) == n);
}

static u32 read_u32(FILE *f) {
    u32 v;
    assert(fread(&v, sizeof(v), 1, f) == 1);
    return v;
}

static usize read_usize(FILE *f) {
    usize v;
    assert(fread(&v, sizeof(v), 1, f) == 1);
    return v;
}

static f64 read_f64(FILE *f) {
    f64 v;
    assert(fread(&v, sizeof(v), 1, f) == 1);
    return v;
}

static Vec read_vec(Arena *a, FILE *f) {
    usize len = read_usize(f);
    Vec v = vec_alloc(a, len);
    assert(fread(v.data, sizeof(f64), len, f) == len);
    return v;
}

static Mat read_mat(Arena *a, FILE *f) {
    usize rows = read_usize(f);
    usize cols = read_usize(f);
    Mat m = mat_alloc(a, rows, cols);
    usize n = rows * cols;
    assert(fread(m.data, sizeof(f64), n, f) == n);
    return m;
}

static FILE *open_write(const char *path, PersistTag tag) {
    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    write_u32(f, (u32)tag);
    return f;
}

static FILE *open_read(const char *path, PersistTag tag) {
    FILE *f = fopen(path, "rb");
    assert(f != NULL);
    assert(read_u32(f) == (u32)tag);
    return f;
}

void persist_linreg_save(const LinReg *model, const char *path) {
    FILE *f = open_write(path, PERSIST_LINREG);
    write_vec(f, &model->weights);
    write_f64(f, model->bias);
    fclose(f);
}

LinReg persist_linreg_load(Arena *a, const char *path) {
    FILE *f = open_read(path, PERSIST_LINREG);
    LinReg model;
    model.weights = read_vec(a, f);
    model.bias = read_f64(f);
    fclose(f);
    return model;
}

void persist_logreg_save(const LogReg *model, const char *path) {
    FILE *f = open_write(path, PERSIST_LOGREG);
    write_vec(f, &model->weights);
    write_f64(f, model->bias);
    fclose(f);
}

LogReg persist_logreg_load(Arena *a, const char *path) {
    FILE *f = open_read(path, PERSIST_LOGREG);
    LogReg model;
    model.weights = read_vec(a, f);
    model.bias = read_f64(f);
    fclose(f);
    return model;
}

void persist_softmax_save(const Softmax *model, const char *path) {
    FILE *f = open_write(path, PERSIST_SOFTMAX);
    write_mat(f, &model->weights);
    write_vec(f, &model->bias);
    fclose(f);
}

Softmax persist_softmax_load(Arena *a, const char *path) {
    FILE *f = open_read(path, PERSIST_SOFTMAX);
    Softmax model;
    model.weights = read_mat(a, f);
    model.bias = read_vec(a, f);
    fclose(f);
    return model;
}

void persist_mlp_save(const MLP *net, const char *path) {
    FILE *f = open_write(path, PERSIST_MLP);
    write_usize(f, net->num_layers);
    for (usize i = 0; i < net->num_layers; i++) {
        const Dense *layer = &net->layers[i];
        write_u32(f, (u32)layer->activation);
        write_mat(f, &layer->weights);
        write_vec(f, &layer->bias);
    }
    fclose(f);
}

MLP persist_mlp_load(Arena *a, const char *path) {
    FILE *f = open_read(path, PERSIST_MLP);
    MLP net;
    net.num_layers = read_usize(f);
    net.layers = (Dense *)arena_alloc(a, net.num_layers * sizeof(Dense));
    for (usize i = 0; i < net.num_layers; i++) {
        Dense *layer = &net.layers[i];
        layer->activation = (DenseActivation)read_u32(f);
        layer->weights = read_mat(a, f);
        layer->bias = read_vec(a, f);
    }
    fclose(f);
    return net;
}
