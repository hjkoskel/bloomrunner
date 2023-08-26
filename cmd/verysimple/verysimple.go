/*
a very simple example
*/
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/hjkoskel/bloomrunner"
)

const DEFAULTPROMPT = "Suomen kieltä käsittelevien tekoälymenetelmien kehitykselle on keskeisen tärkeää, että"

//const DEFAULTPROMPT = "Levanteri tiedusteli pieksujen hintaa"

func main() {
	pModelFile := flag.String("m", "", "model file in ggml binary format")
	pPromptFileName := flag.String("f", "", "prompt filename. If left empty uses default prompt string")
	pTokenLimit := flag.Int("max", 10, "how many tokens are run max. If negative then no limit except end token")

	pRepeatPenalty := flag.Float64("rep", 1.3, "repeat penalty factor on sampling")
	pTopP := flag.Float64("topp", 0.95, "top p sampling treshold")
	pTemp := flag.Float64("temp", 0.8, "temperature for sampling")
	flag.Parse()

	samplingSettings := bloomrunner.BloomSampleSettings{
		RepeatPenalty: float32(*pRepeatPenalty),
		Top_p:         float32(*pTopP),
		Temp:          float32(*pTemp),
	}

	if len(*pModelFile) == 0 {
		fmt.Printf("Please spesify model file\n")
		return
	}
	prompt := DEFAULTPROMPT
	if 0 < len(*pPromptFileName) {
		byt, readErr := os.ReadFile(*pPromptFileName)
		if readErr != nil {
			fmt.Printf("error reading prompt file %s err=%s", *pPromptFileName, readErr.Error())
			os.Exit(-1)
		}
		prompt = strings.TrimSpace(string(byt))
	}

	model, loadErr := bloomrunner.LoadModel(*pModelFile, -1)
	if loadErr != nil {
		fmt.Printf("load failed %s\n", loadErr.Error())
		return
	}

	pred, errPred := model.StartPredicting(prompt, 64)
	if errPred != nil {
		fmt.Printf("starting prediction failed %v\n", errPred)
		return
	}
	pred.Settings = samplingSettings //OPTIONAL!!! change settings if wanted

	fmt.Print(pred.Prompt)
	i := 0
	for i < *pTokenLimit || *pTokenLimit < 0 { //If negative go forever except end token
		newstring, errPredict := model.Predict(&pred)
		if errPredict != nil {
			fmt.Printf("error prediction %s\n", errPredict.Error())
			os.Exit(-1)
		}
		if len(newstring) == 0 {
			break //END
		}
		fmt.Print(newstring)
		i++
	}

	pred.Free()
	model.Free()

	fmt.Printf("\n\n\n-------------------------------\nResult is still available after C-memory free on prediction .. if needed!\n--------------------\n\n%s%s\n\n\n", pred.Prompt, pred.Predicted)

}
