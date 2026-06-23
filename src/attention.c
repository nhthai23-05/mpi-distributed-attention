#include "attention.h"
#include "progress.h"
#include <math.h>
#include <string.h>

void attention_seq(const Tensor *Q, const Tensor *K, const Tensor *V, Tensor *out) {
    int seq_len = Q->rows;
    int d_k     = Q->cols;
    int d_v     = V->cols;

    /* S = Q * K^T   [seq_len x seq_len] */
    Tensor Kt = transpose(K);
    Tensor S  = tensor_alloc(seq_len, seq_len);
    matmul(&S, Q, &Kt);
    tensor_free(&Kt);

    /* scale by 1/sqrt(d_k) */
    tensor_scale(&S, 1.0f / sqrtf((float)d_k));

    /* softmax row-wise */
    softmax_rows(&S);

    /* out = S * V   [seq_len x d_v] */
    (void)d_v;
    matmul(out, &S, V);

    tensor_free(&S);
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
