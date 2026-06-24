#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tensor tensor_alloc(int rows, int cols) {
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    size_t n = (size_t)rows * cols;
    /* A zero-size tensor is legal (e.g. a rank that owns no head columns).
     * malloc(0) may return NULL, which is not an error here. */
    t.data = n ? (float *)malloc(n * sizeof(float)) : NULL;
    if (n && !t.data) {
        fprintf(stderr, "tensor_alloc: out of memory (%d x %d)\n", rows, cols);
        exit(1);
    }
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->rows = t->cols = 0;
}

void tensor_zero(Tensor *t) {
    memset(t->data, 0, (size_t)t->rows * t->cols * sizeof(float));
}

Tensor transpose(const Tensor *A) {
    Tensor t = tensor_alloc(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            AT(t, j, i) = AT(*A, i, j);
    return t;
}
