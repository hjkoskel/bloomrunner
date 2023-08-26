#include "tokenize.h"
#include <stdio.h>

int bloomTokenize(char **tokenmap,int n_vocab,const char *text,int bos, int *tokens, int *n_tokens,int maxnumberof){
    n_tokens[0]=0;
    if (bos){
        tokens[0]=1;
        n_tokens[0]++;
    }
    int position = 0;
    while(1){
        int tokentextlen=0;
        int candidate=0;
        for(int id=0;id<n_vocab;id++){
            int a=strlen(tokenmap[id]);
            if (a<tokentextlen) continue; //got already longer
            if (strlen(text)+position<a) continue;//goes over limit
            if (strncmp(&text[position],tokenmap[id],a)==0){//Matches
                tokentextlen=a;
                candidate=id;
            }
        }
        if (tokentextlen==0) 
            return 0;
        position+=tokentextlen;
        tokens[n_tokens[0]]=candidate;
        n_tokens[0]++;
        if (maxnumberof<=n_tokens[0]) return 1; //reaches limit... not good but not crash
    }
}