#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    float  *data;
    int     rows;
    int     cols;
} Tensor;

Tensor  tensor_alloc(int rows, int cols);
void    tensor_free(Tensor *t);
void    tensor_zero(Tensor *t);
void    tensor_copy(Tensor *dst, const Tensor *src);

/* dst = A * B  (rows_A x cols_A) * (rows_B x cols_B) -> rows_A x cols_B */
void    matmul(Tensor *dst, const Tensor *A, const Tensor *B);

/* dst = A^T */
Tensor  transpose(const Tensor *A);

/* in-place row-wise softmax with numerical stability */
void    softmax_rows(Tensor *t);

/* element-wise operations */
void    tensor_scale(Tensor *t, float s);
void    tensor_add_inplace(Tensor *dst, const Tensor *src);

/* at(t, r, c) — inline access macro */
#define AT(t, r, c)  ((t).data[(r) * (t).cols + (c)])

#endif
