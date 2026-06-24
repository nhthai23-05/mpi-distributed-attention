#include "data_gen.h"
#include <stdlib.h>

void data_gen_fill(Tensor *t, unsigned int seed) {
    srand(seed);
    size_t n = (size_t)t->rows * t->cols;   /* size_t: avoid int overflow at large N */
    for (size_t i = 0; i < n; i++)
        t->data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}
