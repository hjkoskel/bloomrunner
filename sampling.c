#include "sampling.h"
#include <stdlib.h>
#include <math.h>

typedef struct {
    float prob;
    int id;
} ProbEntry; // struct used when sorting probabilities during top-p sampling

int compareProbEntry(const void* a, const void* b) {
    ProbEntry* a_ = (ProbEntry*) a;
    ProbEntry* b_ = (ProbEntry*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

void penalizeRepetition(int n_logits,float *logits,
    int lastTokensN,int *lastTokens,
    float repeat_penalty){

    for(int id=0;id<n_logits;id++){ //index is token id number
        //is there already candidate
        for(int j=0;j<lastTokensN;j++){
            if(lastTokens[j]==id){ //yes, there is make it worse
                if(logits[id]<0){
                    logits[id]*=repeat_penalty;
                }else{
                    logits[id]/=repeat_penalty;
                }
                break;
            }
        }
    }

}

void weightTemperature(int n_logits,float *logits,float temperature){
    if(temperature==0)
        return;
    for(int id=0;id<n_logits;id++) //index is token id number
        logits[id]= (float)((double)(logits[id])/temperature); //TODO float64 vs float32  TEST!
}

//logits tell the propability. tokens tell wich logits
int bloomSampleTopP(unsigned int n_logits,float *logits,
    int lastTokensN,int *lastTokens,
    float repeat_penalty,
    float top_p,float temperature){
    
    weightTemperature(n_logits,logits,temperature);
    penalizeRepetition(n_logits,logits,  lastTokensN,lastTokens,repeat_penalty);

    //TODO Float vs double in calc?
    float maximumvalue=logits[0]; //TODO just pick last or first?
    for(int id=0;id<n_logits;id++){
        if (maximumvalue<logits[id]) maximumvalue=logits[id];
    }

    double sum=0;
    for(int id=0;id<n_logits;id++){
        logits[id]=exp(logits[id]-maximumvalue);
        sum+=logits[id];
    }
    //normalize
    for(int id=0;id<n_logits;id++){
        logits[id]=logits[id]/sum;
    }

    ProbEntry *arr=calloc(n_logits,sizeof(ProbEntry));
    for(int id=0;id<n_logits;id++){
        arr[id].id=id;
        arr[id].prob=logits[id];
    }
    qsort(arr, n_logits, sizeof(ProbEntry), compareProbEntry);

    //Why not simplify this like this?
    float treshold = ((float)rand()/(float)(RAND_MAX)) * top_p;

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for(int i=0;i<n_logits;i++){
            cumsum+=arr[i].prob;
            if(treshold<=cumsum){
                int response=arr[i].id;
                free(arr);
                return response;
            }
        }
    }

    int response=arr[0].id;
    free(arr);
    return response; //the highest propability is picked up
}