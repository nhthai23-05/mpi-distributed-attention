#include "attention.h"
#include "progress.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void attention_row(const float *q, const float *K, const float *V,
                   int n, int d_k, float *o, float *s) {
    float scale = 1.0f / sqrtf((float)d_k);

    /* scores s[j] = (q . K[j]) * scale, tracking the row max for stability */
    float mx = -INFINITY;
    for (int j = 0; j < n; j++) {
        const float *krow = &K[(size_t)j * d_k];
        float dot = 0.0f;
        for (int k = 0; k < d_k; k++) dot += q[k] * krow[k];
        s[j] = dot * scale;
        if (s[j] > mx) mx = s[j];
    }

    /* softmax over the row (this whole row is local to one caller) */
    float sum = 0.0f;
    for (int j = 0; j < n; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
    for (int j = 0; j < n; j++) s[j] /= sum;

    /* o = s . V — accumulate row-by-row so V is read sequentially (row-major)
     * and o[k] sums j in 0..n-1 order, matching the old matmul exactly. */
    for (int k = 0; k < d_k; k++) o[k] = 0.0f;
    for (int j = 0; j < n; j++) {
        float w = s[j];
        const float *vrow = &V[(size_t)j * d_k];
        for (int k = 0; k < d_k; k++) o[k] += w * vrow[k];
    }
}

void attention_seq(const Tensor *Q, const Tensor *K, const Tensor *V, Tensor *out) {
    int seq_len = Q->rows;
    int d_k     = Q->cols;

    /* Streaming + thread-parallel over query rows. Each query row is an
     * independent computation, so we split them across OpenMP threads; every
     * thread keeps its own O(seq_len) score scratch (no shared state, and still
     * no seq_len x seq_len matrix). Without -fopenmp this runs as a plain serial
     * loop with a single scratch buffer. */
    #pragma omp parallel
    {
        float *s = (float *)malloc((size_t)seq_len * sizeof(float));
        #pragma omp for schedule(static)
        for (int i = 0; i < seq_len; i++)
            attention_row(&AT(*Q, i, 0), K->data, V->data, seq_len, d_k,
                          &AT(*out, i, 0), s);
        free(s);
    }
}

void mha_seq(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
             Tensor *out_full, int num_heads) {
    int seq_len = Q_full->rows;
    int d_model = Q_full->cols;
    int d_k     = d_model / num_heads;

    tensor_zero(out_full);

    progress_begin("seq baseline heads", num_heads);
    for (int h = 0; h < num_heads; h++) {
        /* extract head slice: columns [h*d_k, (h+1)*d_k) */
        Tensor Qh = tensor_alloc(seq_len, d_k);
        Tensor Kh = tensor_alloc(seq_len, d_k);
        Tensor Vh = tensor_alloc(seq_len, d_k);
        Tensor Oh = tensor_alloc(seq_len, d_k);

        for (int i = 0; i < seq_len; i++) {
            memcpy(&AT(Qh, i, 0), &AT(*Q_full, i, h * d_k), d_k * sizeof(float));
            memcpy(&AT(Kh, i, 0), &AT(*K_full, i, h * d_k), d_k * sizeof(float));
            memcpy(&AT(Vh, i, 0), &AT(*V_full, i, h * d_k), d_k * sizeof(float));
        }

        attention_seq(&Qh, &Kh, &Vh, &Oh);

        /* write head result back into out_full */
        for (int i = 0; i < seq_len; i++)
            memcpy(&AT(*out_full, i, h * d_k), &AT(Oh, i, 0), d_k * sizeof(float));

        tensor_free(&Qh); tensor_free(&Kh);
        tensor_free(&Vh); tensor_free(&Oh);
        progress_update(h + 1);
    }
    progress_end();
}
