#ifndef BLOOMEVAL_H
#define BLOOMEVAL_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "bloommodel.h"
// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
int bloomEvaluate( bloomModel *m,int n_threads,
        const int n_past,
        int n,
        int32_t *embd_inp,
        float *embd_w,
        size_t *mem_per_token);

#endif