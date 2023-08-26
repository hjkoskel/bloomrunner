#ifndef SAMPLING_H
#define SAMPLING_H

int bloomSampleTopP(unsigned int n_logits,float *logits,
    int lastTokensN,int *lastTokens,
    float repeat_penalty,
    float top_p,
    float temperature);

#endif