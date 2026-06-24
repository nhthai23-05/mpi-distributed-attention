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

/* dst = A^T  (used by the Cannon path to form K^T) */
Tensor  transpose(const Tensor *A);

/* at(t, r, c) — inline access macro */
#define AT(t, r, c)  ((t).data[(r) * (t).cols + (c)])

#endif
