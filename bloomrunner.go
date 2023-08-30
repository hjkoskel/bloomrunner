package bloomrunner

// #include <stdlib.h>
// #include <time.h>
// #include <bloomrunner.h>
//void helperintmalloc(int n,int **mem){
//	mem[0]=malloc(n);
//}
//float *helperEmbedwAlloc(bloomModel *m){
//	return calloc(m->hparams.n_vocab,sizeof(float));
//}
//void helperFreeFloatArr(float *arr){
//	free(arr); //golang type cast helper
//	arr=NULL;
//}
//void helperFreeIntArr(int *arr){
//	free(arr);
//	arr=NULL;
//}
//void randomizesrand(int a){
//	if (a==0) return; //NOP
//	if (0<a){
//		srand(a);
//		return;
//	}
//	srand(time(NULL));
//}
// #cgo LDFLAGS: -L. -L${SRCDIR}/src -lm
// #cgo CPPFLAGS: -I. -I./ggml/include/ggml -pthread -O3
import "C"
import (
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

/*
note to self...  ggml.c on root dir is symlink. need that so c sources are found in build
TODO check how git handles symlink

*/

// Bloom model holds pointer to loaded bloom model
type BloomModel struct {
	Filename   string
	model      C.bloomModel
	MaxThreads int
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

// Negative seed is by time, zero "do not seed" and positive numbers are "set seed"
func LoadModel(filename string, seed int32) (BloomModel, error) {
	if !fileExists(filename) { //The most trivial check.. does file exists
		return BloomModel{}, fmt.Errorf("the model file %s does not exists\n", filename)
	}

	result := BloomModel{
		Filename:   filename,
		MaxThreads: runtime.NumCPU()} //TODO number of cores vs number of virtual cores?

	ret := C.loadModel(C.CString(result.Filename), &result.model)
	if ret == 0 {
		C.randomizesrand(C.int(seed)) //TODO custom made pseudorandom gen instead of srandom?  And per prediction have own seed and status
		return result, nil
	}
	return result, fmt.Errorf("load model %v failed", filename)
}

func (p *BloomModel) Free() {
	C.freeModel(&p.model)
}

type BloomSampleSettings struct {
	RepeatPenalty float32
	Top_p         float32
	Temp          float32
}

func getDefaultBloomSampleSettings() BloomSampleSettings {
	return BloomSampleSettings{RepeatPenalty: 1.3, Top_p: 0.95, Temp: 0.8}
}

// BloomPrediction allows to keep prediction status inside struct and execute multiple predictions.. by taking turns etc...
type BloomPrediction struct {
	npast           C.int
	embed_w         *C.float
	lastTokenBuffer *C.int //Extract to slice if needed at future, else just pass
	repeatLastN     int32

	//Settings, possible to change if needed. Else going
	Settings BloomSampleSettings
	Threads  int //on creation, max number of threads set. can be changed later if decided

	//accesible even after free
	Prompt    string
	Predicted string
}

// StartPredicting feeds prompt and starts prediction. For each new token Predict method had to be called
func (p *BloomModel) StartPredicting(prompt string, repeatLast int32) (BloomPrediction, error) {
	result := BloomPrediction{Prompt: prompt,
		npast:       C.int(0),
		embed_w:     C.helperEmbedwAlloc(&p.model),
		repeatLastN: repeatLast, //TODO solve or understand this! repeatLast vs max context size issue!!!
		Threads:     p.MaxThreads,
		Settings:    getDefaultBloomSampleSettings(),
	}

	C.helperintmalloc(1024*4, &result.lastTokenBuffer) //HACK!

	pro := C.CString(prompt)
	ret := C.startPrediction(&p.model, pro, C.int(result.repeatLastN), &result.npast, result.lastTokenBuffer)
	C.free(unsafe.Pointer(pro))

	if ret != 0 {
		return result, fmt.Errorf("start prediction failed %#v\n", ret)
	}

	return result, nil
}

// Predict predicts next token string .  Empty return string means no new stuff
func (p *BloomModel) Predict(prediction *BloomPrediction) (string, error) {

	C.runPredict(&p.model, prediction.embed_w, C.int(prediction.repeatLastN),
		&prediction.npast, prediction.lastTokenBuffer, C.int(prediction.Threads))

	resp := C.CString("")

	gotNew := C.sample(&p.model, &resp, C.int(prediction.repeatLastN),
		prediction.embed_w,
		&prediction.npast, prediction.lastTokenBuffer,

		C.float(prediction.Settings.RepeatPenalty),
		C.float(prediction.Settings.Top_p),
		C.float(prediction.Settings.Temp))

	newS := C.GoString(resp)
	C.free(unsafe.Pointer(resp))

	if 0 < gotNew {
		prediction.Predicted += newS
		return newS, nil
	}
	return "", nil
}

// Free allocated resources
func (p *BloomPrediction) Free() {
	C.helperFreeFloatArr(p.embed_w)
	C.helperFreeIntArr(p.lastTokenBuffer)
}

/*
func main() {
	fmt.Printf("---GO SOFTA---\n")

	model, loadErr := LoadModel("../aimallit/bloomz.cpp/models/ggml-model-gpt3-finnish-small-f32.bin", -1)
	if loadErr != nil {
		fmt.Printf("load failed %s\n", loadErr.Error())
		return
	}
	fmt.Printf("model OK..\n")

	pred, errPred := model.StartPredicting("Suomen kieltä käsittelevien tekoälymenetelmien kehitykselle on keskeisen tärkeää, että", 64)
	if errPred != nil {
		fmt.Printf("starting prediction failed %v\n", errPred)
		return
	}

	fmt.Printf("threadseja on threadsissä %v ja maxthread =%v\n", pred.Threads, model.MaxThreads)

	for i := 0; i < 10; i++ {
		fmt.Printf("prediction %v\nWith prediction %#v", i, pred)
		newstring, errPredict := model.Predict(&pred)
		if errPredict != nil {
			fmt.Printf("error prediction %s\n", errPredict.Error())
			return
		}
		fmt.Printf("new token:%s\n", newstring)
		fmt.Printf("total=%s%s\n", pred.Prompt, pred.Predicted)
	}

	fmt.Printf("doing model free\n")
	pred.Free()

	fmt.Printf("\nResult is still available if needed!\n\n%s%s\n\n\n", pred.Prompt, pred.Predicted)

	fmt.Printf("Doing model free\n")
	model.Free()
	fmt.Printf("\nDONE\n")
}
*/
