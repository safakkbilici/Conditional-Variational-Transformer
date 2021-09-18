from models.variational_transformer import CVAETransformer
from models.trainer import train
from tokenizer.basic_tokenizer import BasicTokenizer
from tokenizer.bpe import BytePairTokenizer
from utils.dataloaders import get_dataloaders

import torch
import torch.nn as nn
import argparse


def main(args):
    if args.tokenizer == "space":
        tokenizer = BasicTokenizer()
        tokenizer.load("./tokenizer")

    elif args.tokenizer == "bpe":
        tokenizer = BytePairTokenizer()
        tokenizer.load("./tokenizer", "vocab")
    else:
        raise NotImplementedError()
    
    model = CVAETransformer(
        n_src_vocab = len(tokenizer.w2i),
        n_trg_vocab = len(tokenizer.w2i),
        src_pad_idx = tokenizer.pad_token_id,
        trg_pad_idx = tokenizer.pad_token_id,
        trg_emb_prj_weight_sharing = args.trg_proj_weight_sharing,
        emb_src_trg_weight_sharing = args.emb_weight_sharing,
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

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_param}")

    train_dataloader, test_dataloader = get_dataloaders(
        df_train = args.df_train,
        df_test = args.df_test,
        tokenizer = tokenizer,
        text_feature_name = args.df_sentence_name,
        target_feature_name = args.df_sentence_name,
        batch_size = args.batch_size,
        max_len = args.max_seq_len,
        preprocess = args.preprocess
    )

    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.initial_learning_rate)
    scheduler = None
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.cuda == False:
        device = torch.device("cpu")
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = args.scheduler_step_size,
            gamma = args.scheduler_gamma
        )
        
    params = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "device": device,
        "args": args,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader
        
    }

    history = train(**params)
    torch.save(history["model"].state_dict(), "model.pt")
    torch.save(history["optimizer"].state_dict(), "optimizer.pt")
    if history["scheduler"] != None:
        torch.save(history[""].state_dict(), "scheduler.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--df_train",
                        "-train",
                        help="pandas dataframe for training",
                        type = str,
    )

    parser.add_argument("--df_test",
                        "-test",
                        help="pandas dataframe for test",
                        type = str,
    )

    parser.add_argument("--batch_size",
                        "-bs",
                        help="batch size",
                        type = int,
                        default = 32
    )

    parser.add_argument("--df_sentence_name",
                        "-dsn",
                        help="sentence feature name for dataframe",
                        type = str,
                        default = "sentence"
    )

    parser.add_argument("--df_target_name",
                        "-dtn",
                        help="target feature name for dataframe",
                        type = str,
                        default = "target"
    )

    parser.add_argument("--preprocess",
                        "-prc",
                        help="preprocess",
                        type = bool,
                        default = False
    )
    
    parser.add_argument("--epochs",
                        "-e",
                        help="number of epochs",
                        type = int,
                        default = 250
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
                        default = 0.001
    )

    parser.add_argument("--scheduler",
                        "-sch",
                        help="learning rate scheduler: torch.optim.lr_scheduler.StepLR",
                        type = bool,
                        default = False
    )

    parser.add_argument("--cuda",
                        "-cuda",
                        help="device",
                        type = bool,
                        default = True
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
                        type = bool,
                        default = True
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
                        type = bool,
                        default = True
    )

    parser.add_argument("--trg_proj_weight_sharing",
                        "-tpws",
                        help="projection layer and decoder embedding layer weight sharing",
                        type = bool,
                        default = True
    )

    args = parser.parse_args()
    main(args)
