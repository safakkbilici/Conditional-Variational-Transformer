# Conditional (denoising) Variational Transformer

\[Prototype\] A new architecture for class conditional sentence generation to augment downstream tasks: Conditional (denoising) Variational Transformer. Public for "git clone" at gpu server and colab.

Follow-up work of our paper: "Variational Sentence Augmentation for Masked Language Modeling", M. Şafak Bilici & Mehmet Fatih Amasyali, 2021.


## Remainder For Me 

```bash
python3 main.py --df_train DF_TRAIN
                        pandas dataframe for training
                --df_test DF_TEST
                        pandas dataframe for test
                --batch_size BATCH_SIZE
                        batch size
                --df_sentence_name DF_SENTENCE_NAME
                        sentence feature name for dataframe
                --df_target_name DF_TARGET_NAME
                        target feature name for dataframe
                --preprocess PREPROCESS
                        preprocess
                --epochs EPOCHS
                        number of epochs
                --tokenizer TOKENIZER
                        tokenizer
                --initial_learning_rate INITIAL_LEARNING_RATE
                        initial learning rate for Adam
                --scheduler SCHEDULER
                        learning rate scheduler:
                        torch.optim.lr_scheduler.StepLR
                --cuda CUDA
                        device
                --scheduler_step_size SCHEDULER_STEP_SIZE
                        step size for learning rate scheduler
                --scheduler_gamma SCHEDULER_GAMMA
                        gamma factor for learning rate scheduler
                --max_seq_len MAX_SEQ_LEN
                        maximum sequence length for both encoder end decoder
                --posterior_collapse POSTERIOR_COLLAPSE
                        posterior collapse helper
                --anneal_function ANNEAL_FUNCTION
                        type of anneal function for posterior collapse
                --k K
                        k value for posterior collapse anneal function
                --x0 X0
                        x0 value for posterior collapse anneal function
                --latent_size LATENT_SIZE
                        latent size for vae
                --d_k D_K
                        dimension of key (k, q, v)
                --d_v D_V
                        dimension of value (k, q, v)
                --d_model D_MODEL
                        dimension of model
                --d_word D_WORD
                        dimension of word vectors
                --d_inner D_INNER
                        inner dimensionality of model
                --n_layers N_LAYERS
                        number of layers for both encoder and decoder
                --n_heads N_HEADS
                        number of heads for multihead self-attention
                --dropout DROPOUT
                        dropout for model
                --emb_weight_sharing EMB_WEIGHT_SHARING
                        weight sharing for embedding layers of encoder and decoder
                --trg_proj_weight_sharing TRG_PROJ_WEIGHT_SHARING
                        projection layer and decoder embedding layer weight sharing
                --n_classes N_CLASSES
                        number of classes for generating samples
                --n_generate N_GENERATE
                        number of generated samples per class
                --generate_len GENERATE_LEN
                        max len for generation
                --evaluate_per_epoch EVALUATE_PER_EPOCH
                        evaluating at nth epoch
                --model MODEL
                        model checkpoint to load
                --optimizer OPTIMIZER
                        optimizer checkpoint to load
                --model_params MODEL_PARAMS
                        model attributes to load
                --load_scheduler LOAD_SCHEDULER
                        scheduler checkpoint to load
                --noise NOISE
                        noised input to decoder

```
