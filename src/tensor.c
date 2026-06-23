#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

void tensor_copy(Tensor *dst, const Tensor *src) {
    memcpy(dst->data, src->data, (size_t)src->rows * src->cols * sizeof(float));
}

/* dst[i][j] = sum_k A[i][k] * B[k][j] */
void matmul(Tensor *dst, const Tensor *A, const Tensor *B) {
    int M = A->rows, K = A->cols, N = B->cols;
    tensor_zero(dst);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float aik = AT(*A, i, k);
            for (int j = 0; j < N; j++)
                AT(*dst, i, j) += aik * AT(*B, k, j);
        }
}

Tensor transpose(const Tensor *A) {
    Tensor t = tensor_alloc(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            AT(t, j, i) = AT(*A, i, j);
    return t;
}

/* numerically stable row-wise softmax */
void softmax_rows(Tensor *t) {
    for (int i = 0; i < t->rows; i++) {
        float max_val = AT(*t, i, 0);
        for (int j = 1; j < t->cols; j++)
            if (AT(*t, i, j) > max_val) max_val = AT(*t, i, j);

        float sum = 0.0f;
        for (int j = 0; j < t->cols; j++) {
            AT(*t, i, j) = expf(AT(*t, i, j) - max_val);
            sum += AT(*t, i, j);
        }
        for (int j = 0; j < t->cols; j++)
            AT(*t, i, j) /= sum;
    }
}

void tensor_scale(Tensor *t, float s) {
    size_t n = (size_t)t->rows * t->cols;
    for (size_t i = 0; i < n; i++) t->data[i] *= s;
}

void tensor_add_inplace(Tensor *dst, const Tensor *src) {
    size_t n = (size_t)dst->rows * dst->cols;
    for (size_t i = 0; i < n; i++) dst->data[i] += src->data[i];
}
