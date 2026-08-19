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
- **`src/linreg.h`** - linear regression, fit either in closed form via the normal equations or by batch gradient descent.
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
| `tests/test_linreg.c` | Closed-form and gradient descent fits on exactly-linear data, prediction and mean squared error. |

### Test Results
```bash
make test
```
Each fixture prints its own pass line, for example `PASS dataset (25 checks)`, and the run exits zero when all pass. To run a single fixture, build and invoke it directly, for example `make build/test_matrix && ./build/test_matrix`.
