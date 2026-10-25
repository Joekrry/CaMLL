# CaMLL

**Project**: `CaMLL`, a machine learning library written in C (Raw C11).

**About**: `CaMLL` provides the classical and deep learning building blocks for supervised learning — regression and classification — on dense numeric data. It is written from scratch in C11 with no external dependencies: the numeric type aliases, an arena allocator, dense vector and matrix types, dataset loading and the models are all in-repository.

---

## Table of Contents
1. General Description
2. Software Testing

---

## General Description

### Overview

`CaMLL` is organised as a static C library with an accompanying test suite. The build is driven by `Makefile`, which archives the sources under `src/` and `alloc/` into `build/libcamll.a` and compiles each test under `tests/` into its own executable.

Layered architecture (current):
```
types.h  --->  alloc/  --->  src/  --->  tests/
```

| Layer | Folder | Responsibility |
|-------|--------|----------------|
| Numeric types | `types.h` | Fixed-width integer and floating point aliases used throughout the library. |
| Allocation | `alloc/` | Arena (bump) allocator that backs all library data. |
| Library | `src/` | Machine learning implementation: dense linear algebra, dataset handling and models. |
| Tests | `tests/` | Unit tests and the `test.h` assertion harness. |

**Main components**
- **`types.h`** - defines the numeric aliases (`i8` to `i64`, `u8` to `u64`, `f32`, `f64`, `usize`, `isize`). Wraps `<stdint.h>` and `<stddef.h>` so the rest of the library never spells out a raw C integer type.
- **`alloc/arena.h`** - bump allocator that hands out memory by advancing an offset into `mmap`ed blocks. There is no per-object free; memory is reclaimed in bulk with `arena_reset` or rolled back to an `arena_mark`.
- **`src/matrix.h`** - the `Vec` and `Mat` types, both dense `f64` with `Mat` stored row-major. Allocation is arena-backed, so the types carry no ownership and are freed with the arena rather than individually. Provides the core matrix operations (add, sub, scale, matmul, transpose), vector operations (dot, norms, elementwise mul/div, outer product), a seeded xorshift PRNG and debug printing.
- **`src/dataset.h`** - the `Dataset` type (feature matrix plus target vector), CSV parsing, shuffled train/test splitting and per-column standardization.
- **`src/metrics.h`** - shared evaluation metrics: mean squared error, binary cross-entropy, accuracy, and precision/recall/F1 via a binary confusion matrix.
- **`src/linreg.h`** - linear regression, fit either in closed form via the normal equations or by batch gradient descent with an optional L2 (ridge) penalty on the weights.
- **`src/activation.h`** - sigmoid (with derivative) and softmax, both scalar and vectorized.
- **`src/logreg.h`** - logistic regression, fit by batch gradient descent on binary cross-entropy loss with an optional L2 penalty, with probability and class prediction.
- **`src/softmax.h`** - softmax regression (multinomial logistic regression), fit by batch gradient descent on cross-entropy loss with an optional L2 penalty, with per-class probability and argmax class prediction.
- **`src/knn.h`** - k-nearest neighbors, a lazy learner that stores the training set and predicts from the k closest rows by squared Euclidean distance, with both classification (majority vote) and regression (mean) prediction.
- **`src/dense.h`** - a single feed-forward layer (`z = X * weightsᵗ + bias`) with a linear, sigmoid or softmax activation, exposing forward and backward passes so layers can be chained into a network.
- **`src/mlp.h`** - a multi-layer perceptron: a stack of `dense.h` layers, fit by batch gradient descent with backpropagation through the whole stack and an optional L2 penalty, supporting regression (linear output), binary classification (sigmoid output) and multiclass classification (softmax output).
- **`src/persist.h`** - binary save/load of fitted weights for `linreg`, `logreg`, `softmax` and `mlp`, each file tagged with its model type so loading the wrong file asserts instead of silently misreading floats.
- **`tests/test.h`** - single-header assertion harness. Each test file is an independent executable that reports a pass or fail summary through `test_summary()`.

### Installation Instructions

**Prerequisites**

| Requirement | Notes |
|-------------|-------|
| C11 compiler | `cc`, `gcc` or `clang`. |
| `make` | Drives the build. |
| POSIX host | The arena allocator uses `mmap` via `<sys/mman.h>`. |

**Step-by-step setup**
```bash
git clone https://github.com/Joekrry/CaMLL
cd CaMLL
make
```
`make` archives the static library into `build/`. The `build/` directory is created on demand and is not tracked in version control.

### Usage Instructions

Build the library and run the full test suite:
```bash
make test
```

| Path | Contents |
|------|----------|
| `build/libcamll.a` | Compiled static library. |
| `build/test_*` | Compiled test executables. |

---

## Software Testing

### Testing Framework

**Framework Used**: a small in-repository harness, `tests/test.h` (no external test framework).

**Testing Approach**: each test source compiles to a standalone executable that exercises one area of the library and returns a non-zero exit code on failure. `make test` builds and runs every `tests/test_*.c`, so a failing check breaks the build. The tests are hermetic and take no external input.

### Test Cases

| Fixture | Covers |
|---------|--------|
| `tests/test_types.c` | Width and signedness of every alias in `types.h`. |
| `tests/test_matrix.c` | Arena-backed allocation, add, sub, scale, matmul, transpose, dot, norms, elementwise ops, outer product and the seeded PRNG. |
| `tests/test_dataset.c` | CSV parsing with the label in varying columns, train/test split row pairing, and standardization including the zero-variance guard. |
| `tests/test_metrics.c` | Mean squared error, binary cross-entropy on confident correct/wrong predictions, accuracy, confusion matrix counts, and precision/recall/F1 including the no-positives degenerate case. |
| `tests/test_linreg.c` | Closed-form and gradient descent fits on exactly-linear data, prediction, mean squared error, and weight shrinkage under an L2 penalty. |
| `tests/test_activation.c` | Sigmoid and its derivative, and softmax (sums to 1, shift invariance, uniform input). |
| `tests/test_logreg.c` | Gradient descent fits on linearly separable data, probability/class prediction, accuracy, binary cross-entropy, and weight shrinkage under an L2 penalty. |
| `tests/test_softmax.c` | Gradient descent fit on a three-class, linearly separable problem; per-class probabilities summing to 1, argmax class prediction, accuracy, and weight shrinkage under an L2 penalty. |
| `tests/test_knn.c` | Majority-vote classification on separated clusters (including an exact-match query with k=1), and mean regression on a linear grid (exact match at k=1, averaged neighbors at k=2). |
| `tests/test_dense.c` | Forward-pass shapes and activation ranges for linear, sigmoid and softmax layers, and a finite-difference gradient check of the weight, bias and input gradients returned by the backward pass. |
| `tests/test_mlp.c` | A hidden-sigmoid, output-sigmoid network solving XOR (not linearly separable, so only passes if backprop reaches the hidden layer), weight shrinkage across every layer under an L2 penalty, and a hidden-sigmoid, output-softmax network on the three-class problem from `test_softmax.c`. |
| `tests/test_persist.c` | Binary save/load round-trips for `linreg`, `logreg`, `softmax` and `mlp`, checking that weights, biases and (for `mlp`) per-layer activations are recovered exactly and predictions on the loaded model match the original. |

### Test Results
```bash
make test
```
Each fixture prints its own pass line, for example `PASS dataset (25 checks)`, and the run exits zero when all pass. To run a single fixture, build and invoke it directly, for example `make build/test_matrix && ./build/test_matrix`.
