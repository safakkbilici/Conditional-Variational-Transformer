from models.variational_transformer import CVAETransformer
from pretraining.pretrainer import train
from pretraining.denoising_lm import DenoisingLM, DenoisingLMDataset

from tokenizer.space_tokenizer import SpaceTokenizer
from tokenizer.bpe_tokenizer import BytePairTokenizer

from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import argparse
import json

def main(args):
    if args.tokenizer == "space":
        tokenizer = SpaceTokenizer()
        tokenizer.load("./tokenizer")

    elif args.tokenizer == "bpe":
        tokenizer = BytePairTokenizer()
        tokenizer.load("./tokenizer", "vocab")
    else:
        raise NotImplementedError()

    if args.model_params != "none" and args.model != "none":
        with open(args.model_params, 'r') as fp:
            model_params = json.load(fp)
        model = CVAETransformer(**model_params)
        print(f"Model is loaded from {args.model} and {args.model_params}")
        model.load_state_dict(torch.load(args.model))


    else:
        model = CVAETransformer(
            n_src_vocab = tokenizer.vocab_size,
            n_trg_vocab = tokenizer.vocab_size,
            src_pad_idx = tokenizer.pad_token_id,
            trg_pad_idx = tokenizer.pad_token_id,
            trg_emb_prj_weight_sharing = True if args.trg_proj_weight_sharing == "true" else False,
            emb_src_trg_weight_sharing = True if args.emb_weight_sharing == "true" else False,
            d_k = args.d_k,
            d_v = args.d_v,
            d_model = args.d_model,
            d_word_vec = args.d_word,
            d_inner = args.d_inner,
            n_layers = args.n_layers,
            n_head = args.n_heads,
            dropout = args.dropout,
            scale_emb_or_prj = 'prj',
            enc_max_seq_len = args.max_seq_len,
            latent_size = args.latent_size,
            n_position = args.max_seq_len + 1
        )

        model_params = model.serialize
        with open('model_params.json', 'w') as fp:
            json.dump(model_params, fp)

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_param}")
    print(f"CUDA: {args.cuda}")


    objective = DenoisingLM(
        tokenizer.mask_token, 
        tokenizer.mask_token_id, 
        tokenizer.end_token_id, 
        tokenizer.pad_token_id,
        max_seq_len = args.max_seq_len,
        prob_mask = args.prob_mask,
        prob_delete = args.prob_delete
    )

    train_dataset = DenoisingLMDataset(
        tokenizer = tokenizer, 
        corpus = args.train_corpus, 
        objective = objective, 
        max_seq_len = args.max_seq_len, 
        pad_token_id = tokenizer.pad_token_id,
        end_token_id = tokenizer.end_token_id,
        start_token_id = tokenizer.start_token_id
    )

    test_dataset = DenoisingLMDataset(
        tokenizer = tokenizer, 
        corpus = args.test_corpus, 
        objective = objective, 
        max_seq_len = args.max_seq_len, 
        pad_token_id = tokenizer.pad_token_id,
        end_token_id = tokenizer.end_token_id,
        start_token_id = tokenizer.start_token_id
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size = args.batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle = False,
        batch_size = args.batch_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = args.initial_learning_rate)
    if args.optimizer != "none":
        optimizer.load_state_dict(args.optimizer)

    scheduler = None
    criterion = nn.CrossEntropyLoss()
    if args.cuda == "true":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")


    print(f"Device: {device}")
    
    if args.scheduler == "true":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = args.scheduler_step_size,
            gamma = args.scheduler_gamma
        )

        if args.load_scheduler != "none":
            scheduler.load_state_dict(args.load_scheduler)


    params = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "device": device,
        "args": args,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "tokenizer": tokenizer
    }

    history = train(**params)
    torch.save(history["model"].state_dict(), "model.pt")
    torch.save(history["optimizer"].state_dict(), "optimizer.pt")
    if history["scheduler"] != None:
        torch.save(history["scheduler"].state_dict(), "scheduler.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_corpus",
                        "-train",
                        help="corpus for training",
                        type = str,
    )

    parser.add_argument("--test_corpus",
                        "-test",
                        help="corpus for test",
                        type = str,
    )

    parser.add_argument("--batch_size",
                        "-bs",
                        help="batch size",
                        type = int,
                        default = 32
    )

    parser.add_argument("--prob_delete",
                        "-pd",
                        help="delete probability",
                        type = int,
                        default = 7
    )

    parser.add_argument("--prob_mask",
                        "-pm",
                        help="mask probability",
                        type = int,
                        default = 10
    )
    
    parser.add_argument("--epochs",
                        "-e",
                        help="number of epochs",
                        type = int,
                        default = 15
    )

    parser.add_argument("--tokenizer",
                        "-tok",
                        help="tokenizer",
                        type = str,
                        default = "space"
    )

    parser.add_argument("--initial_learning_rate",
                        "-ilr",
                        help="initial learning rate for Adam",
                        type = float,
                        default = 0.0005
    )

    parser.add_argument("--scheduler",
                        "-sch",
                        help="learning rate scheduler: torch.optim.lr_scheduler.StepLR",
                        type = str,
                        default = "false"
    )

    parser.add_argument("--cuda",
                        "-cuda",
                        help="device",
                        type = str,
                        default = "true"
    )

    parser.add_argument("--scheduler_step_size",
                        "-sss",
                        help="step size for learning rate scheduler",
                        type = int,
                        default = 20
    )

    parser.add_argument("--scheduler_gamma",
                        "-scg",
                        help="gamma factor for learning rate scheduler",
                        type = float,
                        default = 0.1
    )

    parser.add_argument("--max_seq_len",
                        "-msl",
                        help="maximum sequence length for both encoder end decoder",
                        type = int,
                        default = 200
    )

    parser.add_argument("--posterior_collapse",
                        "-pc",
                        help = "posterior collapse helper",
                        type = str,
                        default = "true"
    )

    parser.add_argument("--anneal_function",
                        "-af",
                        help = "type of anneal function for posterior collapse",
                        type = str,
                        default = "logistic"
    )

    parser.add_argument("--k",
                        "-k",
                        help="k value for posterior collapse anneal function",
                        type = float,
                        default = 0.0025
    )

    parser.add_argument("--x0",
                        "-x0",
                        help="x0 value for posterior collapse anneal function",
                        type = int,
                        default = 2500
    )

    parser.add_argument("--latent_size",
                        "-ls",
                        help="latent size for vae",
                        type = int,
                        default = 16
    )

    parser.add_argument("--d_k",
                        "-dk",
                        help="dimension of key (k, q, v)",
                        type = int,
                        default = 64
    )

    parser.add_argument("--d_v",
                        "-dv",
                        help="dimension of value (k, q, v)",
                        type = int,
                        default = 64
    )

    parser.add_argument("--d_model",
                        "-dm",
                        help="dimension of model",
                        type = int,
                        default = 256
    )

    parser.add_argument("--d_word",
                        "-dw",
                        help="dimension of word vectors",
                        type = int,
                        default = 256
    )

    parser.add_argument("--d_inner",
                        "-di",
                        help="inner dimensionality of model",
                        type = int,
                        default = 128
    )

    parser.add_argument("--n_layers",
                        "-nl",
                        help="number of layers for both encoder and decoder",
                        type = int,
                        default = 3
    )

    parser.add_argument("--n_heads",
                        "-nh",
                        help="number of heads for multihead self-attention",
                        type = int,
                        default = 4
    )

    parser.add_argument("--dropout",
                        "-drp",
                        help="dropout for model",
                        type = float,
                        default = 0.1
    )

    parser.add_argument("--emb_weight_sharing",
                        "-ews",
                        help="weight sharing for embedding layers of encoder and decoder",
                        type = str,
                        default = "false"
    )

    parser.add_argument("--trg_proj_weight_sharing",
                        "-tpws",
                        help="projection layer and decoder embedding layer weight sharing",
                        type = str,
                        default = "false"
    )

    parser.add_argument("--evaluate_per_epoch",
                        "-epe",
                        help="evaluating at nth epoch",
                        type = int,
                        default = 2
    )

    parser.add_argument("--model",
                        "-ml",
                        help="model checkpoint to load",
                        type = str,
                        default = "none"
    )

    parser.add_argument("--optimizer",
                        "-opt",
                        help="optimizer checkpoint to load",
                        type = str,
                        default = "none"
    )

    parser.add_argument("--model_params",
                        "-mlp",
                        help="model attributes to load",
                        type = str,
                        default = "none"
    )

    parser.add_argument("--load_scheduler",
                        "-lsch",
                        help="scheduler checkpoint to load",
                        type = str,
                        default = "none"
    )

    args = parser.parse_args()
    main(args)
