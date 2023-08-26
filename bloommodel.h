#ifndef BLOOMMODEL_H
#define BLOOMMODEL_H

#include <stdint.h>
#include "ggml.h"

typedef struct {
    int32_t n_vocab;// = 32000;
    int32_t n_ctx;//   = 512;   // this is provided as user input?  TODO REMOVE AND READ WHOLE THING IN ONE READ!
    int32_t n_embd;//  = 4096;
    int32_t n_mult;//  = 256;
    int32_t n_head;//  = 32;
    int32_t n_layer;// = 32;
    int32_t f16;//     = 1;
} bloom_hparams;

void debugprintHparams(bloom_hparams *pars);

typedef struct {
    // normalization
    struct ggml_tensor *attention_norm;
    struct ggml_tensor *attention_norm_b;

    // attention
    struct ggml_tensor *query_key_value;
    struct ggml_tensor *query_key_value_b;
    struct ggml_tensor *wo;
    struct ggml_tensor *wo_b;

    // normalization
    struct ggml_tensor *ffn_norm;
    struct ggml_tensor *ffn_norm_b;

    // ff
    struct ggml_tensor *w1;
    struct ggml_tensor *w1_b;
    struct ggml_tensor *w2;
    struct ggml_tensor *w2_b;
}bloomLayer;

typedef struct {
    bloom_hparams hparams;

    struct ggml_tensor *tok_embeddings;
    struct ggml_tensor *norm;
    struct ggml_tensor *norm_b;

    struct ggml_tensor *output_norm;
    struct ggml_tensor *output_norm_b;
    struct ggml_tensor *output;
    

    //std::vector<bloom_layer> layers;
    bloomLayer *layers; //  hparams.nLayer

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;  //Ok, pointer to place where things happen
    char **tokenmap;//This is really n_vocab length of char strings... seriously? using map on this??? really?
}bloomModel;

int loadBloomModel(const char *filename,bloomModel *m,int n_ctx);

#endif