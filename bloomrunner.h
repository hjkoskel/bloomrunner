#ifndef BLOOMRUNNER_H
#define BLOOMRUNNER_H

#include <stdint.h>
#include "bloommodel.h"

int loadModel(const char *modelfilename,bloomModel *model);
int freeModel(bloomModel *model);

int startPrediction(bloomModel *model,const char *prompt,int repeat_last_n,int *npast,int *lastTokenBuffer);

int runPredict(bloomModel *model,float *embed_w,int repeat_last_n, 
    int *npast,int *lastTokenBuffer,
    int n_threads);

int sample(bloomModel *model,char **result,int repeat_last_n, 
    float *embed_w,
    int *npast,int *lastTokenBuffer,
    float repeat_penalty,float top_p,float temp);


#endif