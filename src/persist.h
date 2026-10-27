#ifndef CAMLL_SRC_PERSIST_H
#define CAMLL_SRC_PERSIST_H

#include "types.h"
#include "alloc/arena.h"
#include "linreg.h"
#include "logreg.h"
#include "softmax.h"
#include "mlp.h"

/* Binary save/load for fitted weights. Each file starts with a tag for its
   model type — loading the wrong file asserts instead of silently
   misreading floats. Not portable across endianness or f64/usize width. */

void persist_linreg_save(const LinReg *model, const char *path);
LinReg persist_linreg_load(Arena *a, const char *path);

void persist_logreg_save(const LogReg *model, const char *path);
LogReg persist_logreg_load(Arena *a, const char *path);

void persist_softmax_save(const Softmax *model, const char *path);
Softmax persist_softmax_load(Arena *a, const char *path);

void persist_mlp_save(const MLP *net, const char *path);
MLP persist_mlp_load(Arena *a, const char *path);

#endif /* CAMLL_SRC_PERSIST_H */
