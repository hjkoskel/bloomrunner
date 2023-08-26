#ifndef TOKENIZE_H
#define TOKENIZE_H

#include <string.h>
int bloomTokenize(char **tokenmap,int n_vocab,const char *text,int bos, int *tokens, int *n_tokens,int maxnumberof);

#endif