#include "bloomrunner.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
Just minimal example how to use library from C
Primary use is on golang interface. This C implementation is used for detecting memory leaks and debugging

This program takes only one parameter. filename of model file

This can be compiled with just single command
gcc -std=c11 -pthread -I../ -O3 -Wall -lm example.c ../*.c

*/

int main(int argc, char ** argv) {
    printf("-----bloomrunner----\n");
    #define REPEAT_LAST_N 64
    #define NTHREADS 4
    float   top_p = 0.95;
    float   temp= 0.80;
    float   repeat_penalty  = 1.30;
    bloomModel llm;
    srand(time(NULL));

    if (argc<2){
        printf("Please give model filename as command line argument\n");
        return 0;
    }

    //if (loadModel("../aimallit/bloomz.cpp/models/ggml-model-gpt3-finnish-xl-f16.bin",&llm)){
    //if (loadModel("../aimallit/bloomz.cpp/models/ggml-model-gpt3-finnish-large-f16.bin",&llm)){
    //if (loadModel("../aimallit/bloomz.cpp/models/ggml-model-gpt3-finnish-small-f16.bin",&llm)){
    //if (loadModel("/home/henri/aimallit/bloomz.cpp/models/ggml-model-gpt3-finnish-small-f32.bin",&llm)){
    //if (loadModel("../aimallit/bloomz.cpp/models/ggml-model-gpt3-finnish-small-f16_q4_0.bin",&llm)){
    if (loadModel(argv[1],&llm)){
        printf("ERR: failed loading model\n");
        return -1;
    }
    float *embed_w=calloc(llm.hparams.n_vocab,sizeof(float));
    int npast=0; //Position counter
    int lastTokenBuffer[1024];

    //static const char prompt[]="levanteri tiedusteli pieksujen hintaa";    
    static const char prompt[]="Suomen kieltä käsittelevien tekoälymenetelmien kehitykselle on keskeisen tärkeää, että";
    //static const char prompt[]="Onko seuraava lause positiivinen vai negatiivinen?\nLause: Täällä on erittäin hauskaa!\nVastaus: positiivinen\nOnko seuraava lause positiivinen vai negatiivinen?\nLause: elokuva oli surkea!\nVastaus:";

    printf("%s",prompt);
    fflush(stdout);
    if (startPrediction(&llm,prompt,REPEAT_LAST_N,&npast,lastTokenBuffer)){
        printf("ERROR on prediction\n");
        return -1;
    }

    char *newTokenText;
    for(int stepNumber=0;stepNumber<100;stepNumber++){
        runPredict(&llm,embed_w,REPEAT_LAST_N, 
            &npast,lastTokenBuffer,
        NTHREADS);

        int gotNew = sample(&llm,&newTokenText,REPEAT_LAST_N,
        embed_w,
        &npast,lastTokenBuffer,
        repeat_penalty,top_p,temp);


        if (gotNew==0){
            break;
        }
        printf("%s",newTokenText);
        fflush(stdout);
    }

    if (freeModel(&llm)){
        printf("ERROR free memory\n");
        return -1;
    }
    free(embed_w);
    printf("\n\nDONE\n");
    return 0;
}