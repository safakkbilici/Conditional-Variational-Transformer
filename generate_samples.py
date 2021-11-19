import torch
import torch.nn as nn

import json

from models.variational_transformer import CVAETransformer
from models.eval import generate
from tokenizer.space_tokenizer import SpaceTokenizer
from tokenizer.bpe_tokenizer import BytePairTokenizer

import argparse

import pandas as pd

def main(args):
    if args.tokenizer == "space":
        tokenizer = BasicTokenizer()
        tokenizer.load("./tokenizer")

    elif args.tokenizer == "bpe":
        tokenizer = BytePairTokenizer()
        tokenizer.load("./tokenizer", "vocab")
    else:
        raise NotImplementedError()

    with open(f'{args.model_params}', 'r') as fp:
        model_params = json.load(fp)

    if args.cuda == "true":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = CVAETransformer(**model_params)
    model.load_state_dict(torch.load(f"{args.model}"))
    model = model.to(device)

    samples = generate(
        model = model,
        device = device,
        tokenizer = tokenizer,
        latent_size = model_params["latent_size"],
        n_classes = args.n_classes,
        n_samples_per_class = args.n_generate,
        generate_len = args.generate_len
    )

    df_generated = pd.DataFrame(samples)
    print(df_generated)
    df_generated.to_csv("generated.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        "-m",
                        help="trained model",
                        type = str,
    )

    parser.add_argument("--cuda",
                        "-c",
                        help="device",
                        type = str,
                        default = "true"
    )

    parser.add_argument("--tokenizer",
                        "-t",
                        help="tokenizer",
                        type = str,
                        default = "space"
    )

    parser.add_argument("--model_params",
                        "-mp",
                        help="model parameters as json file",
                        type = str,
                        default = "model_params.json"
    )

    parser.add_argument("--n_classes",
                        "-ncl",
                        help="number of classes for generating samples",
                        type = int,
                        default = 2
    )

    parser.add_argument("--n_generate",
                        "-ngen",
                        help="number of generated samples per class",
                        type = int,
                        default = 1
    )

    parser.add_argument("--generate_len",
                        "-ngenl",
                        help="max len for generation",
                        type = int,
                        default = 100
    )

    parser.add_argument("--evaluate_per_epoch",
                        "-epe",
                        help="evaluating at nth epoch",
                        type = int,
                        default = 2
    )

    args = parser.parse_args()
    main(args)  
