# Bloomrunner

Bloomrunner runs bloom language models. Bloom models are not the best language models (especially when doing dialog) but some language researches have been training bloom models on some rare and esoteric language like finnish. 

Bloomrunner is based on https://github.com/NouamaneTazi/bloomz.cpp but written in C. This also uses great tensor calculation library made by Georgi Gerganov https://github.com/ggerganov/ggml  (ggml.c and ggml.h files)

Code written in C provides more low level, hands on experience for development.
Also wrapping code into golang or some other language library comes more straightforward.

This code is still in early stages of development and it is developed during personal LLM experiece

# Usage

## Getting and converting models

There are some finnish language models made at University of Turku
https://turkunlp.org/gpt3-finnish

models are hosted on huggingface. For this example here
https://huggingface.co/TurkuNLP/gpt3-finnish-small


Then clone original bloomz.cpp library  (separate conversion/quantization utility is developed later)
```bash
cd bloomz.cppdir
# install required libraries
python3 -m pip install torch numpy transformers accelerate
# download and convert the 7B1 model to ggml FP16 format
python3 convert-hf-to-ggml.py TurkuNLP/gpt3-finnish-small ./models
# Note: you can add --use-f32 to convert to FP32 instead of FP16
```

Optionally, follow bloomz.cpp instructions if quantization is required

```bash
./quantize ./models/ggml-model-gpt3-finnish-small-f16.bin ./models/ggml-model-gpt3-finnish-small-f16_q4_0.bin 2
```

Finally you have some models on *./models* directory like

* ggml-model-gpt3-finnish-small-f16.bin
* ggml-model-gpt3-finnish-small-f16_q4_0.bin
* ggml-model-gpt3-finnish-small-f32.bin

## Using in C
folder example have example file

On example directory, the simple way to build is
```bash
gcc -std=c11 -pthread -I../ -I../ggml/include/ggml/ -O3 -lm example.c ../*.c
```

## Using in golang

import "github.com/hjkoskel/bloomrunner"

*bloomrunner.LoadModel(* loads *BloomModel*. The BloomModel holds pointer to loaded model.
Loaded BloomModel is stateless and C-code part does not have global variables.

Method 
```go
func (p *BloomModel) StartPredicting(prompt string, repeatLast int32) (BloomPrediction, error) {
```
Creates BloomModelPrediction, think that as "session". StartPredicting initializes BloomPrediction with default settings values.

If needed it is possible to adjust Settings on that prediction.
```go
type BloomSampleSettings struct {
	RepeatPenalty float32
	Top_p         float32
	Temp          float32
}
```
Actual prediction happens when method
```go
func (p *BloomModel) Predict(prediction *BloomPrediction) (string, error) {
```
is called continously. If end token is reached, function returns empty string.
Because there are no other state than what is stored on *BloomPrediction*
It is possible to start predicting tokens on multiple promps on same bloom model (if they take turns). This would be handy on situations where one server serves same model on multiple clients


### example program in go
Check example **./cmd/verysimple/verysimple.go**

it builds as normal golang program with 
```bash
go mod tidy
go build
```

It have some basic functionalies for testing bloom language model
```
Usage of ./verysimple:
  -f string
        prompt filename. If left empty uses default prompt string
  -m string
        model file in ggml binary format
  -max int
        how many tokens are run max. If negative then no limit except end token (default 10)
  -rep float
        repeat penalty factor on sampling (default 1.3)
  -temp float
        temperature for sampling (default 0.8)
  -topp float
        top p sampling treshold (default 0.95)
```

Program have default prompt 
```
Suomen kieltä käsittelevien tekoälymenetelmien kehitykselle on keskeisen tärkeää, että
```

