# Class Conditional Variational Transformer

Code for our paper "Transformers as Neural Augmentors: Class Conditional Sentence Generation with Variational Bayes", submitted to ICPR 2022.

**Abstract:**
Data augmentation for Natural Language Processing tasks are explored in recent years, however they are limited and hard to capture the diversity on sentence level. Besides, it is not always possible to perform data augmentation on supervised tasks. To address those problems, we propose a neural data augmentation method, which is a combination of Conditional Variational Autoencoder and encoder-decoder Transformer model. While encoding and decoding input sentence, our model captures the syntactic and semantic representation of the input language with its class condition. Following the developments in the past years on pre-trained language models, we train and evaluate our models on several benchmarks to robust the downstream tasks. We compare our method with 3 different augmentation techniques. The presented results show that, our model increases the performance of current models compared to other data augmentation techniques with a small amount of computation power.


## Training A Tokenizer

```console
python train_tokenizer.py \
       --dataframe "train.csv" \
       --cased "true" \ 
       --preprocess "true" \
       --tokenizer "space" \
       --feature_name "sentence"
```
Please take a look at arguments in [train_tokenizer.py](https://github.com/safakkbilici/Conditional-Variational-Transformer/blob/main/train_tokenizer.py) file if you want to configure the tokenizer.

Vocab and config files of your trained tokenizer will be saved under ./tokenizer directory.

## Training Class Conditional Variational Transformer

main.py will use trained tokenizer which is saved under ./tokenizer directory.

```console
python main.py \ 
       --df_train "train.csv" \
       --df_test "test.csv" \
       --preprocess "true" \
       --epochs 90 \
       --tokenizer "space" \
       --max_seq_len 128 \
       --df_sentence_name "sentence" \
       --df_target_name "target" \
       --cuda "true" \
       --batch_size 32 \
       --posterior_collapse "true" \
       --initial_learning_rate 0.0005 \
       --noise "false" \
       --n_classes 6 \
       --latent_size 32
```

Please take a look at arguments in [main.py](https://github.com/safakkbilici/Conditional-Variational-Transformer/blob/main/main.py) file if you want to configure hyperparameters of proposed model's, training configuration, or load model and resume training.

model_params.json, model.pt and optimizer.pt (scheduler.pt if used) files will be saved under main directory.

## Generating Sentences
Please use [generate.ipynb](https://github.com/safakkbilici/Conditional-Variational-Transformer/blob/main/notebooks/generate.ipynb) notebook to generate new sentences for data augmentation.

## Finetuning
We provide finetuning scripts as well at [./benchmarks/models](https://github.com/safakkbilici/Conditional-Variational-Transformer/tree/main/benchmarks/models). However, anyone can write their own finetuning code.

## Pre-training Class Conditional Variational Transformer
We haven't experimented much our pre-training objective and code. To pre-train Class Conditional Variational Transformer, we use denoising sequence-to-sequence pre-training, which is proposed by Lewis et al., 2019. 

Train a tokenizer:

```console
python train_tokenizer.py \
       --dataframe "wiki.train.tokens" \
       --cased "true" \
       --preprocess "false" \
       --tokenizer "bpe"
```

Pre-train Class Conditional Variational Transformer:

```console
python main_pretraining.py \
       --train_corpus "wiki.train.tokens" \
       --test_corpus "wiki.test.tokens" \
       --batch_size 32 \
       --epochs 150 \
       --tokenizer "bpe" \
       --max_seq_len 256 \
       --latent_size 32 \
       --initial_learning_rate 0.0005 \
       --posterior_collapse "true" \
```

## Finetuning Class Conditional Variational Transformer

After pre-training model_params.json, model.pt and optimizer.pt (scheduler.pt if used) files will be saved under main directory. Following the same training script (with a few additional arguments), you can train your pre-trained Class Conditional Variational Transformer to generate new sentences for data augmentation.

```console
python main.py \ 
       --df_train "train.csv" \
       --df_test "test.csv" \
       --preprocess "true" \
       --epochs 90 \
       --tokenizer "space" \
       --max_seq_len 128 \
       --df_sentence_name "sentence" \
       --df_target_name "target" \
       --cuda "true" \
       --batch_size 32 \
       --posterior_collapse "true" \
       --initial_learning_rate 0.0005 \
       --noise "false" \
       --n_classes 6 \
       --latent_size 32 \
       --model "model.pt" \
       --model_params "model_params.json" \
       --pretraining "true"
```

We implemented four finetuning procedure in [finetune.py](https://github.com/safakkbilici/Conditional-Variational-Transformer/blob/main/pretraining/finetune.py) file. However, we haven't give it as an argument in main.py file. If anybody want to freeze custom layers, please give a dictionary to constructor in main.py file:

```python
if args.pretraining == "true":
   layers = {"enc": [0, 1, 2], "dec": [1, 2, 3]} 
   # freeze encoder but last MHSA, freeze decoder but first MHSA
   freezer = Freezer(layers)
   model = freezer.freeze(model)
```
