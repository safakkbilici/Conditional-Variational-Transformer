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

## Training Class Conditional Variational Transformer

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

Please take a look at arguments in [main.py](https://github.com/safakkbilici/Conditional-Variational-Transformer/blob/main/main.py) file if you want to configure hyperparameters of proposed model's or training configuration.
