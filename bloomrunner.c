#include <stdio.h>
#include <time.h>
#include "bloomrunner.h"
#include "ggml.h"

#include "bloommodel.h"
#include "tokenize.h"
#include "bloomeval.h"
#include "sampling.h"

typedef struct { //FOR ORGANIZING
    int32_t seed;//      = -1; // RNG seed
    int32_t n_threads;// = 4; //std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict;// = 128; // new tokens to predict
    int32_t repeat_last_n;// = 64;  // last n tokens to penalize

    // sampling parameters
    int32_t top_k;// = 40; // unused
    float   top_p;// = 0.95f;
    float   temp;//  = 0.80f;
    float   repeat_penalty;//  = 1.30f;
} gpt_params;

void getDefaultParams(gpt_params *p){
    p->seed = -1; // RNG seed
    p->n_threads = 4; //std::min(4, (int32_t) std::thread::hardware_concurrency());
    p->n_predict = 128; // new tokens to predict
    p->repeat_last_n = 64;  // last n tokens to penalize

    // sampling parameters
    p->top_k = 40; // unused
    p->top_p = 0.95f;
    p->temp  = 0.80f;
    p->repeat_penalty  = 1.30f;
}

//Helper function. TODO separate module? Or make bloomrunner as helper function module
void writeRollingIntBuf(int *buf,int cap,int newToken,int *counter){
    for(int i=1;i<cap;i++){
        buf[i-1]=buf[i];
    }
    buf[cap-1]=newToken;
    counter[0]++; //Only place where is incremented
}

void getLatestFromRollingIntBuf(int *buf,int cap,int counter,int **arr,int *arrLen){
    if (counter==0){
        return;
    }
    if (cap<=counter){ //abstract off by one errors away
        arrLen[0]=cap;
        arr[0]=buf;
        return;
    }
    arrLen[0]=counter;
    arr[0]=&buf[cap-counter];
}

int loadModel(const char *modelfilename,bloomModel *model){
    const int n_ctx = 512; //TODO VOIKO MUUTTAA!
    if(loadBloomModel(modelfilename,model,n_ctx)){
        fprintf(stderr,"%s: error loading model",__func__);
        return 1;
    }
    return 0;
}


//gives initial prompt
int startPrediction(bloomModel *model,const char *prompt,int repeat_last_n,int *npast,int *lastTokenBuffer){
    int bos=1; //TODO CHECK MEANING
    #define MAXNUMBEROFTOKENS 1024
    int tokenized[MAXNUMBEROFTOKENS]; //TODO MAXLIMIT.. 
    int numberOfPromptTokens;
    if (bloomTokenize(model->tokenmap, model->hparams.n_vocab, prompt,bos, tokenized, &numberOfPromptTokens,MAXNUMBEROFTOKENS)){
        fprintf(stderr,"%s: bloom tokenizer fail\n",__func__);
        return -1;
    }
    npast[0]=0; //Lets go to start
    for(int i=0;i<numberOfPromptTokens;i++) //lets run prompt tokenized to this and fill up lastTokenBuffer for prediction
        writeRollingIntBuf(lastTokenBuffer,repeat_last_n,tokenized[i],npast);

    return 0;
}

int runPredict(bloomModel *model,float *embed_w,int repeat_last_n, 
    int *npast,int *lastTokenBuffer,
    int n_threads){

    int32_t *latestTokensPointer=NULL;
    int numberOfLatestTokens=0;
    getLatestFromRollingIntBuf(
        lastTokenBuffer,
        repeat_last_n,
        npast[0],&latestTokensPointer,&numberOfLatestTokens);

    size_t mem_per_token=0;

    return bloomEvaluate(model,
        n_threads,
        npast[0], //this is position coding index
        numberOfLatestTokens,latestTokensPointer,
        embed_w, //output length of hparams.n_vocab
        &mem_per_token);
}

int sample(bloomModel *model,char **result,int repeat_last_n, 
    float *embed_w,
    int *npast,int *lastTokenBuffer,
    float repeat_penalty,float top_p,float temp){

    int newid=bloomSampleTopP(model->hparams.n_vocab,
    embed_w,
    repeat_last_n, lastTokenBuffer,  //TODO maksimikoko ja sit siitä ei-nollat
    repeat_penalty,top_p,temp);

    //free(embed_w);

    if (model->hparams.n_vocab<=newid){
        fprintf(stderr,"%s: internal error id is too high id=%d n_vocab=%d\n",__func__,newid,model->hparams.n_vocab);
        return -1;
    }

    //Not cat just point to next token  strcat(result,g_model.tokenmap[newid]); //TODO get that BEFORE? like when insert promt. Now lagging behind
    result[0]=model->tokenmap[newid]; //Is this better behaviour?
    if (newid == 2) { //last element
        return 0;
    }
    writeRollingIntBuf(lastTokenBuffer,repeat_last_n,newid,npast);
    return 1;
}

int freeModel(bloomModel *model){ //TODO free model
    ggml_free(model->ctx);
    return 0;
}
