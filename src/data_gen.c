#include "data_gen.h"
#include <stdlib.h>

void data_gen_fill(Tensor *t, unsigned int seed) {
    srand(seed);
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++)
        t->data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}
