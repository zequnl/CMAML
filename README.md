## Learning to Customize Model Structures for Few-shot Dialogue Generation Tasks

This is the implementation of our ACL 2020 paper:

**Learning to Customize Model Structures for Few-shot Dialogue Generation Tasks**. 

Yiping Song, Zequn Liu, Wei Bi, Rui Yan, Ming Zhang

https://arxiv.org/abs/1910.14326

Please cite our paper when you use this code in your work.

## Dependency
```console
❱❱❱ pip install -r requirements.txt
```
Put the [**Pre-trained glove embedding**](http://nlp.stanford.edu/data/glove.6B.zip): ***glove.6B.300d.txt*** in /vectors/.

[**Trained NLI model**](https://drive.google.com/file/d/1Qawz1pMcV0aGLVYzOgpHPgG5vLSKPOJ1/view?usp=sharing) ***pytorch_model.bin*** in /data/nli_model/.
## Experiment

The code is for the experiment of our model CMAML-Seq2SPG on [**Persona-chat**](https://arxiv.org/abs/1801.07243). The scripts for training and evaluation are "train.sh" and "test.sh".

After training, please set the "--save_model" as the model with the lowest PPL in validation set to evaluate the model.
## Acknowledgement
We use the framework of [**PAML**](https://github.com/HLTCHKUST/PAML) and the Seq2seq implementation in https://github.com/MaximumEntropy/Seq2Seq-PyTorch
