#include "bloomeval.h"

int bloomEvaluate( bloomModel *m,
        int n_threads, //TODO MODELLIN SISÄLTÄ!
        const int n_past, //tää on se paikkakoodauksen indeksi
        int n, //length of embed_inp  C-versioon lisätty
        int32_t *embd_inp, //INPUT GOES HERE  tämä on se "embd" vektori johon aina pushataan embd.push_back(id); Normaalisti 1 promptilla muuten tokenit
        float *embd_w, //output length of hparams.n_vocab
        size_t *mem_per_token){
    
    size_t buf_size = 512u*1024*1024;
    void *buf = (void *)malloc(buf_size);

    float eps=1e-5f; //TODO WHY in hparam?

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    //gf.n_threads = n_threads;

    struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,n);
    memcpy(embd->data, embd_inp, n*ggml_element_size(embd)); //TODO kopioidaa ID? emvd tyyppi?

    struct ggml_tensor *inpL = ggml_get_rows(ctx0, m->tok_embeddings, embd);
    // word embeddings norm
    inpL = ggml_norm(ctx0, inpL,eps);
    inpL = ggml_mul(ctx0, ggml_repeat(ctx0, m->norm, inpL), inpL);
    inpL = ggml_add(ctx0, ggml_repeat(ctx0, m->norm_b, inpL), inpL);

    for (int il = 0; il < m->hparams.n_layer; ++il) {
        struct ggml_tensor *inpSA = inpL; //TODO: copy?
        struct ggml_tensor *cur;
        // norm
        {
            cur = ggml_norm(ctx0, inpL,eps);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, m->layers[il].attention_norm, cur),
                        cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, m->layers[il].attention_norm_b, cur), cur);
        }

        // attn
        {
            cur = ggml_mul_mat(ctx0,m->layers[il].query_key_value, cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, m->layers[il].query_key_value_b, cur),
                    cur);
        }

        // cur = ggml_debug(ctx0, cur);

        // self-attention
        {
            struct ggml_tensor *Qcur = ggml_view_2d(ctx0, cur, m->hparams.n_embd, n, cur->nb[1], 0*sizeof(float)*m->hparams.n_embd);
            struct ggml_tensor *Kcur = ggml_view_2d(ctx0, cur, m->hparams.n_embd, n, cur->nb[1], 1*sizeof(float)*m->hparams.n_embd); //TODO: float or fp16?
            struct ggml_tensor *Vcur = ggml_view_2d(ctx0, cur, m->hparams.n_embd, n, cur->nb[1], 2*sizeof(float)*m->hparams.n_embd);

            // store key and value to memory
            if (n >= 1) {
                struct ggml_tensor *k = ggml_view_1d(ctx0, m->memory_k, n*m->hparams.n_embd, (ggml_element_size(m->memory_k)*m->hparams.n_embd)*(il*m->hparams.n_ctx + n_past));
                struct ggml_tensor *v = ggml_view_1d(ctx0, m->memory_v, n*m->hparams.n_embd, (ggml_element_size(m->memory_v)*m->hparams.n_embd)*(il*m->hparams.n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            //TODO dkey
            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor *Q =
                ggml_permute(ctx0,
                            ggml_cpy(ctx0, Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, m->hparams.n_embd/m->hparams.n_head, m->hparams.n_head, n)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor *K =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, m->memory_k, (n_past + n)*m->hparams.n_embd, il*m->hparams.n_ctx*ggml_element_size(m->memory_k)*m->hparams.n_embd),
                                m->hparams.n_embd/m->hparams.n_head, m->hparams.n_head, n_past + n),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor *KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1/sqrt(((float)m->hparams.n_embd)/m->hparams.n_head))
                        );

            // Alibi
            // KQ_scaled_alibi = KQ_scaled + alibi_bias //TODO: optimize
            float TODOwhatIsMaxBias=999; //where load this? from original bloom checkpoint?
            struct ggml_tensor *KQ_scaled_alibi = ggml_alibi(ctx0, KQ_scaled, n_past, m->hparams.n_head,TODOwhatIsMaxBias);

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor *KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled_alibi, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor *V_trans =
                    ggml_cpy(ctx0,
                             ggml_permute(ctx0,
                                          ggml_reshape_3d(ctx0,
                                                          ggml_view_1d(ctx0, m->memory_v, (n_past + n) * m->hparams.n_embd,
                                                                       il * m->hparams.n_ctx * ggml_element_size(m->memory_v) *
                                                                       m->hparams.n_embd),
                                                          m->hparams.n_embd / m->hparams.n_head, m->hparams.n_head, n_past + n),
                                          1, 2, 0, 3),
                             ggml_new_tensor_3d(ctx0, m->memory_v->type, n_past + n, m->hparams.n_embd / m->hparams.n_head, m->hparams.n_head));
            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, m->hparams.n_embd, n));

            // projection
            cur = ggml_mul_mat(ctx0,
                    m->layers[il].wo,
                    cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, m->layers[il].wo_b, cur), cur);
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF,eps);

                // cur = ffn_norm*cur + ffn_norm_b
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, m->layers[il].ffn_norm, cur),
                        cur);
                cur = ggml_add(ctx0, ggml_repeat(ctx0, m->layers[il].ffn_norm_b, cur), cur);
            }

            cur = ggml_mul_mat(ctx0,
                    m->layers[il].w1,
                    cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, m->layers[il].w1_b, cur), cur);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_mul_mat(ctx0,
                    m->layers[il].w2,
                    cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, m->layers[il].w2_b, cur), cur);
        }

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }
    // norm
    {
        inpL = ggml_norm(ctx0, inpL,eps);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, m->output_norm, inpL),
                    inpL);
        
        inpL = ggml_add(ctx0, ggml_repeat(ctx0, m->output_norm_b, inpL), inpL);
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, m->output, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL); //for some reason this was commented out

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute_with_ctx(ctx0, &gf,n_threads);
    memcpy(embd_w, (float *) ggml_get_data(inpL) + (m->hparams.n_vocab*(n-1)), sizeof(float)*m->hparams.n_vocab);
    if (mem_per_token[0] == 0) {
        mem_per_token[0] = ggml_used_mem(ctx0)/n;
    }
    ggml_free(ctx0);
    return 0;
}