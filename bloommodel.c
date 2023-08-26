#include "bloommodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include <assert.h>

#define MAGICFILENUMBER 0x67676d6c

int loadHyperparams(FILE *file, bloom_hparams *p){
    //thanks to n_ctx... not loading with single read :(  TODO CHANGE!!
    if(fread(&p->n_vocab, sizeof(uint32_t), 1, file) != 1) return 1;
    if(fread(&p->n_embd, sizeof(uint32_t), 1, file) != 1) return 1;
    if(fread(&p->n_mult, sizeof(uint32_t), 1, file) != 1) return 1;
    if(fread(&p->n_head, sizeof(uint32_t), 1, file) != 1) return 1;
    if(fread(&p->n_layer, sizeof(uint32_t), 1, file) != 1) return 1;
    if(fread(&p->f16, sizeof(uint32_t), 1, file) != 1) return 1;
    return 0;
}

int loadVocabulary(FILE *file,bloomModel *model){
    model->tokenmap=(char **)calloc(model->hparams.n_vocab, sizeof(char *));
    uint32_t length;
    for (int i = 0; i < model->hparams.n_vocab; i++) {
        if(fread(&length, sizeof(uint32_t), 1, file) != 1){
            return 1;
        }
        model->tokenmap[i]=(char*)calloc(length+1,1);
        if(fread(model->tokenmap[i],1,length,file)!=length){
            return 2;
        }
    }
    return 0;
}

size_t countContextSize(bloom_hparams *h){
    enum ggml_type wtype = GGML_TYPE_COUNT;
    switch (h->f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
            fprintf(stderr, "%s: bad f16 value %d\n", __func__,h->f16);
            return 1;
    }

    size_t ctx_size = 0;
    int n_ff = ((4*h->n_embd + h->n_mult - 1)/h->n_mult)*h->n_mult;

    ctx_size += h->n_embd*h->n_vocab   *ggml_type_sizef(wtype); // tok_embeddings

    ctx_size += h->n_embd           *ggml_type_sizef(GGML_TYPE_F32); // norm
    ctx_size += h->n_embd           *ggml_type_sizef(GGML_TYPE_F32); // norm_b
    ctx_size += h->n_embd           *ggml_type_sizef(GGML_TYPE_F32); // output_norm
    ctx_size += h->n_embd           *ggml_type_sizef(GGML_TYPE_F32); // output_norm_b

    ctx_size += h->n_embd*h->n_vocab*ggml_type_sizef(wtype); // output
    
    ctx_size += h->n_layer*(
        h->n_embd               *ggml_type_sizef(GGML_TYPE_F32)+ // attention_norm
        h->n_embd               *ggml_type_sizef(GGML_TYPE_F32)+ // attention_norm_b
        3*h->n_embd*h->n_embd   *ggml_type_sizef(wtype)+ // query_key_value
        3*h->n_embd             *ggml_type_sizef(GGML_TYPE_F32)+ // query_key_value_b
        h->n_embd*h->n_embd     *ggml_type_sizef(wtype)+ // wo
        h->n_embd               *ggml_type_sizef(GGML_TYPE_F32)+ // wo_b

        h->n_embd               *ggml_type_sizef(GGML_TYPE_F32)+ // ffn_norm
        h->n_embd               *ggml_type_sizef(GGML_TYPE_F32)+ // ffn_norm_b

        n_ff*h->n_embd          *ggml_type_sizef(wtype)+ // w1
        n_ff                    *ggml_type_sizef(GGML_TYPE_F32)+ // w1_b
        n_ff*h->n_embd          *ggml_type_sizef(wtype)+ // w2
        n_ff                    *ggml_type_sizef(GGML_TYPE_F32)+ // w2_b

        h->n_ctx*h->n_embd         *ggml_type_sizef(GGML_TYPE_F32)+ // memory_k
        h->n_ctx*h->n_embd         *ggml_type_sizef(GGML_TYPE_F32) // memory_v            
    );
    ctx_size += (5 + 10*h->n_layer)*256; // object overhead TODO:
    return ctx_size;
}

int prepareMemForModel(bloomModel *m){ //TODO m on mukava muuttujanimi
    //TODO FIKSATUT TYYPIT TÄÄLLÄ.  Voisiko alustaa samalla kertaa kun ladataan data? Yhdessä paikkaa
    //eka vaan kääntyväksi!

    enum ggml_type wtype = GGML_TYPE_COUNT;
    switch (m->hparams.f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
            fprintf(stderr, "%s (bad f16 value %d)\n",__func__, m->hparams.f16);
            return 1;
    }
    //printf("preparing for model m->tok_embeddings   %d x %d = %d\n",m->hparams.n_embd, m->hparams.n_vocab,m->hparams.n_embd*m->hparams.n_vocab);
    m->tok_embeddings = ggml_new_tensor_2d(m->ctx, wtype,         m->hparams.n_embd, m->hparams.n_vocab);
    m->norm   =         ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
    m->norm_b =         ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);

    m->output_norm =    ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
    m->output_norm_b =  ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
    m->output =         ggml_new_tensor_2d(m->ctx, wtype,         m->hparams.n_embd, m->hparams.n_vocab);
    

    const int n_elements      = m->hparams.n_layer * m->hparams.n_ctx*m->hparams.n_embd;
    m->memory_k = ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, n_elements);
    m->memory_v = ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, n_elements);


    m->layers=calloc(m->hparams.n_layer,sizeof(bloomLayer));

    int n_ff = ((4*m->hparams.n_embd + m->hparams.n_mult - 1)/m->hparams.n_mult)*m->hparams.n_mult;

    for (int i = 0; i < m->hparams.n_layer; ++i) {
         bloomLayer *layer=&m->layers[i];
         
         layer->attention_norm =     ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
         layer->attention_norm_b =   ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
         
         layer->query_key_value =    ggml_new_tensor_2d(m->ctx, wtype,         m->hparams.n_embd,       3*m->hparams.n_embd);
         layer->query_key_value_b =  ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, 3*m->hparams.n_embd);
         layer->wo =                 ggml_new_tensor_2d(m->ctx, wtype,         m->hparams.n_embd,       m->hparams.n_embd);
         layer->wo_b =               ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
         
         layer->ffn_norm =           ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
         layer->ffn_norm_b =         ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
         
         layer->w1 =                 ggml_new_tensor_2d(m->ctx, wtype,         m->hparams.n_embd,       n_ff);
         layer->w1_b =               ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, n_ff);
         layer->w2 =                 ggml_new_tensor_2d(m->ctx, wtype,         n_ff,         m->hparams.n_embd);
         layer->w2_b =               ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, m->hparams.n_embd);
    }
    return 0;
}

//pointer to tensor and what kind of it is and what dimensions are expected
int getPointerToTensor(bloomModel *m,char *varname,struct ggml_tensor **out,int *splitByRows, int *dim0Expect,int *dim1Expect){
    // split_type = 0: split by columns
    // split_type = 1: split by rows

    //TODO why different?
    // split_type = 0:
    //   - tok_embeddings.*
    //   - layers.*.attention.wo.weight
    //   - layers.*.feed_forward.w2.weight

    // split_type = 1:
    //   - output.*
    //   - layers.*.attention.wq.weight
    //   - layers.*.attention.wk.weight
    //   - layers.*.attention.wv.weight
    //   - layers.*.feed_forward.w1.weight
    //   - layers.*.feed_forward.w3.weight
    splitByRows[0]=1;
    out[0]=NULL;
    dim0Expect[0]=0;
    dim1Expect[0]=0;

    if (strcmp(varname,"tok_embeddings.weight")==0){
        splitByRows[0]=0;
        out[0]=m->tok_embeddings;
        dim0Expect[0]=m->hparams.n_embd;
        dim1Expect[0]=m->hparams.n_vocab;
        return 0;
    }
    if (strcmp(varname,"norm.weight")==0){
        out[0]=m->norm;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strcmp(varname,"norm.bias")==0){
        out[0]=m->norm_b;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strcmp(varname,"output_norm.weight")==0){
        out[0]=m->output_norm;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strcmp(varname,"output_norm.bias")==0){
        out[0]=m->output_norm_b;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strcmp(varname,"output.weight")==0){
        out[0]=m->output;
        dim0Expect[0]=m->hparams.n_embd;
        dim1Expect[0]=m->hparams.n_vocab;
        return 0;
    }

    int n_ff = ((4*m->hparams.n_embd + m->hparams.n_mult - 1)/m->hparams.n_mult)*m->hparams.n_mult;

    int32_t layerNumber=0;
    if (sscanf(varname,"layers.%d",&layerNumber)==0) {
        fprintf(stderr,"%s: no match for %s\n",__func__,varname);
        return -1;
    }
    if ((layerNumber<0)||(m->hparams.n_layer<=layerNumber)){
        fprintf(stderr,"%s: max layers %d have variable %s\n",__func__,layerNumber,varname); 
        return -1;
    }
    if (strstr(varname,".attention_norm.weight")!=NULL){
        out[0]=m->layers[layerNumber].attention_norm;
        dim0Expect[0]=m->hparams.n_embd;
        dim1Expect[0]=0;
        return 0;
    }
    if (strstr(varname,".attention_norm.bias")!=NULL){
        out[0]=m->layers[layerNumber].attention_norm_b;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strstr(varname,".attention.query_key_value.weight")!=NULL){
        out[0]=m->layers[layerNumber].query_key_value;
        dim0Expect[0]=m->hparams.n_embd;
        dim1Expect[0]=m->hparams.n_embd*3;
        return 0;
    }
    if (strstr(varname,".attention.query_key_value.bias")!=NULL){
        out[0]=m->layers[layerNumber].query_key_value_b;
        dim0Expect[0]=3*m->hparams.n_embd;
        return 0;
    }
    if (strstr(varname,".attention.wo.weight")!=NULL){ // split_type = 0
        out[0]=m->layers[layerNumber].wo;
        splitByRows[0]=0;
        dim0Expect[0]=m->hparams.n_embd;
        dim1Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strstr(varname,".attention.wo.bias")!=NULL){
        out[0]=m->layers[layerNumber].wo_b;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strstr(varname,".ffn_norm.weight")!=NULL){
        out[0]=m->layers[layerNumber].ffn_norm;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strstr(varname,".ffn_norm.bias")!=NULL){
        out[0]=m->layers[layerNumber].ffn_norm_b;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    
    if (strstr(varname,".feed_forward.w1.weight")!=NULL){
        out[0]=m->layers[layerNumber].w1;
        dim0Expect[0]=m->hparams.n_embd;
        dim1Expect[0]=n_ff;
        return 0;
    }
    
    if (strstr(varname,".feed_forward.w1.bias")!=NULL){
        out[0]=m->layers[layerNumber].w1_b;
        dim0Expect[0]=n_ff;
        return 0;
    }
    if (strstr(varname,".feed_forward.w2.weight")!=NULL){ // split_type = 0
        out[0]=m->layers[layerNumber].w2;
        splitByRows[0]=0;
        dim0Expect[0]=n_ff,         
        dim1Expect[0]=m->hparams.n_embd;
        return 0;
    }
    if (strstr(varname,".feed_forward.w2.bias")!=NULL){
        out[0]=m->layers[layerNumber].w2_b;
        dim0Expect[0]=m->hparams.n_embd;
        return 0;
    }
    fprintf(stderr,"no matching layer for %s\n",varname);

    return -1;
}

int loadBloomModel(const char *filename,bloomModel *m,int n_ctx){
    FILE *file = fopen(filename, "rb");
    uint32_t magic;
    if(fread(&magic, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr,"%s: error reading magic",__func__);
        return 1;
    }
    if (magic!=MAGICFILENUMBER){
        fprintf(stderr,"%s: invalid magic number",__func__);
        return 1;
    }
    if (loadHyperparams(file, &m->hparams)){
        fprintf(stderr,"%s: error loading hyperparams",__func__);
        return 1;
    }
    m->hparams.n_ctx=n_ctx; //yes, why this in this way?

    //debugprintHparams(&m->hparams);  TODO print on golang not inside library

    if (loadVocabulary(file,m)){
        fprintf(stderr,"%s: error loading vocubalory",__func__);
        return 1;
    }

    size_t ctx_size=countContextSize(&m->hparams);
    // create the ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
    };

    m->ctx = ggml_init(params);
    if (!m->ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }
    // prepare memory for the weights  (prepare... malloc and set ggml oper)
    if (prepareMemForModel(m)){
        fprintf(stderr,"%s: preparing memory for model failed\n",__func__);
        return 1;
    }
    size_t total_size = 0;

    int32_t n_dims;
    int32_t namelength;
    int32_t ftype;
    int32_t dimensions[2];
    char *varname=NULL;
    while(1){
        // dim, length, type

        if(fread(&n_dims, sizeof(uint32_t), 1, file) != 1){
            if( feof(file) ) {//good spot to break 
                break ;
            }
            fprintf(stderr,"%s: error reading n_dims\n",__func__);
            return 1;
        }
        if(fread(&namelength, sizeof(uint32_t), 1, file) != 1){
            fprintf(stderr,"%s: error reading namelength\n",__func__);
            return 1;
        }
        if(fread(&ftype, sizeof(uint32_t), 1, file) != 1){
            fprintf(stderr,"%s: error reading ftype\n",__func__);
            return 1;
        }

        if ((n_dims!=1)&&(n_dims!=2)){ //This model have only 1d and 2d
            fprintf(stderr,"%s: not supported %dD matrixies\n",__func__,n_dims);
            return 1;
        }

        dimensions[0]=0;
        dimensions[1]=0;
        if(fread(dimensions, sizeof(uint32_t), n_dims, file) != n_dims) return 1;

        if ((2<n_dims)||(n_dims==0)){
            fprintf(stderr,"%s: internal error only 1d and 2d are supported not %dd\n",__func__,n_dims);
            return -1;
        }

        varname=calloc(namelength+1,1);
        if(varname==NULL){
            fprintf(stderr,"%s: error allocating varname\n",__func__);
            return -1;
        }

        if(fread(varname, sizeof(char), namelength, file) != namelength){
            free(varname);
            fprintf(stderr,"%s: error reading name",__func__);
            return 1;
        }
        struct ggml_tensor *targetTensor;
        int splitByRows;
        int dim0Expect;
        int dim1Expect;

        getPointerToTensor(m,varname, &targetTensor,&splitByRows, &dim0Expect,&dim1Expect);
        
        if (targetTensor==NULL){
            fprintf(stderr,"%s: unknown variable %s. TODO JUST SKIP?\n",__func__,varname);
            return -1;
        }
        int nelements=dimensions[0];
        if (n_dims==2){
            nelements*=dimensions[1];
            if ((dim0Expect!=dimensions[0])||(dim1Expect!=dimensions[1])) {
                fprintf(stderr,"%s: on %s expected length %d x %d got now %d x %d\n",__func__,varname,dim0Expect,dim1Expect,dimensions[0],dimensions[1]);
                return -1;
            }

        }else{
            if (dim0Expect!=dimensions[0]){
                fprintf(stderr,"%s: on %s expected length %d got now %d\n",__func__,varname,dim0Expect,dimensions[0]);
                return -1;
            }
        }
        if (ggml_nelements(targetTensor) != nelements) { //TODO BETTER WAY TO CHECK
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, varname);
            return false;
        }
        //Reading actual data

        if (n_dims == 1){ //1d is easy
            if(fread(targetTensor->data, ggml_nbytes(targetTensor), 1, file) != 1){
                fprintf(stderr,"%s: error reading 1d %s\n", __func__, varname);
                return 1;
            }
        }else{
            //2d two options
            if(splitByRows){ // "1" in original
                const size_t row_size = (targetTensor->ne[0]/ggml_blck_size(targetTensor->type))*ggml_type_size(targetTensor->type);
                for (int i1 = 0; i1 < dimensions[1]; ++i1) {
                    if(fread(targetTensor->data+ i1*row_size, row_size, 1, file) != 1){
                        fprintf(stderr,"%s: error reading 2d %s\n", __func__, varname);
                        return 1;
                    }
                }
            }else{
                const size_t row_size = (targetTensor->ne[0]/ggml_blck_size(targetTensor->type))*ggml_type_size(targetTensor->type);
                assert(row_size == targetTensor->nb[1]); //Way to simplify?  use tensor->nb[1] ?
                for (int i1 = 0; i1 < dimensions[1]; ++i1) {
                    if(fread(targetTensor->data+ i1*row_size, row_size, 1, file) != 1){
                        fprintf(stderr,"%s: error reading 2d %s\n", __func__, varname);
                        return 1;
                    }
                }
            }
        }
        total_size += ggml_nbytes(targetTensor);
    }
    if (fclose(file)){
        fprintf(stderr,"%s: error closing file %s\n",__func__,filename);
        return -1;
    }
    return 0;
}

void debugprintHparams(bloom_hparams *pars){
    printf("n_vocab=%d\n n_ctx=%d\n n_embd=%d\n n_mult=%d\n n_head=%d\n n_layer=%d\n f16;=%d\n",
    pars->n_vocab,
    pars->n_ctx,
    pars->n_embd,
    pars->n_mult,
    pars->n_head,
    pars->n_layer,
    pars->f16);
}