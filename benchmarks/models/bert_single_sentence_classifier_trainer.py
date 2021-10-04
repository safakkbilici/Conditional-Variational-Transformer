import torch
import torch.nn as nn
import argparse
import pandas as pd
from transformers import AutoTokenizer

from benchmarks.models.bert_single_sentence_classifier import (
    BERTClassifier,
    dataloader_bert_classifier,
    train_bert_classifier
)

def main(args):
    print(f"CUDA: {args.cuda}")
    if args.cuda == "true":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = BERTClassifier(args.model_name, args.n_classes, args.dropout)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.initial_learning_rate)
    
    
    df_train = pd.read_csv(args.df_train)
    df_test = pd.read_csv(args.df_test)

    df_train = df_train.rename(columns={f'args.df_sentence_name': 'sentence', 'args.df_target_name': 'target'})
    df_test = df_test.rename(columns={f'args.df_sentence_name': 'sentence', 'args.df_target_name': 'target'})

    train_dataloader, test_dataloder = dataloader_bert_classifier(
        tokenizer = tokenizer,
        batch_size = args.batch_size,
        df_train = df_train,
        df_test = df_test
    )

    results = train_bert_classifier(
        model = model,
        device = device,
        criterion = criterion,
        optimizer = optimizer,
        epochs = args.epochs,
        n_classes = args.n_classes,
        train_dataloader = train_dataloader,
        test_dataloder = test_dataloder,
        sometimes_loss = args.sometimes_loss,
        maximize = args.maximize,
        save_every = True if args.save_every=="true" else False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name",
                        "-mn",
                        help="pretrained model from huggingface hub",
                        type = str,
                        default = "bert-base-cased"
    )

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

    parser.add_argument("--epochs",
                        "-e",
                        help="number of epochs",
                        type = int,
                        default = 250
    )

    parser.add_argument("--initial_learning_rate",
                        "-ilr",
                        help="initial learning rate for Adam",
                        type = float,
                        default = 0.001
    )

    parser.add_argument("--cuda",
                        "-cuda",
                        help="device",
                        type = str,
                        default = "true"
    )

    parser.add_argument("--dropout",
                        "-drp",
                        help="dropout for model",
                        type = float,
                        default = 0.1
    )

    parser.add_argument("--n_classes",
                        "-ncl",
                        help="number of classes to be classified",
                        type = int,
                        default = 2
    )

    parser.add_argument("--sometimes_loss",
                        "-sl",
                        help="prints training loss at each sometimes loss step",
                        type = int,
                        default = 10
    )

    parser.add_argument("--maximize",
                        "-max",
                        help="monitored metric for saving best model",
                        type = str,
                        default = "acc"
    )

    parser.add_argument("--save_every",
                        "-se",
                        help="save model at each epoch, with overwrite",
                        type = str,
                        default = "true"
    )

    args = parser.parse_args()
    main(args)
