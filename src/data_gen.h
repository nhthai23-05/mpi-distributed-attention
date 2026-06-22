#ifndef DATA_GEN_H
#define DATA_GEN_H

#include "tensor.h"

/*
 * Fill a tensor with deterministic pseudo-random values in (-1, 1).
 * Same seed -> same data on every rank, so all nodes start identical.
 */
void data_gen_fill(Tensor *t, unsigned int seed);

#endif
