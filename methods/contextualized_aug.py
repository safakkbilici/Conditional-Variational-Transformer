import argparse
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
from tqdm.auto import tqdm
import random
import nltk
import re
import functools

from utils.preprocessing import denoise_text

def augment(args):

    augm = naw.ContextualWordEmbsAug(
        model_path=args.model_name, 
        action="insert",
        device = "cuda" if args.cuda=="true" else "cpu"
    )
    
    df = pd.read_csv(args.data)
    if args.preprocessing != "none":
        preprocessor = functools.partial(
            denoise_text,
            t = args.preprocessing_type
        )
        df.sentence = df.sentence.apply()

    augs = []
    targets = []
    with tqdm(total = args.total) as tt:
        for i in range(args.total):
            if i < len(df):  
                sentence = df.iloc[i,0]
                target = df.iloc[i, 1]
                aug = augm.augment(sentence)
                augs.append(aug)
                targets.append(target)
            else:
                randidx = random.randint(0, len(df) - 1)
                sentence = df.iloc[randidx, 0]
                target = df.iloc[randidx, 1]
                aug = augm.augment(sentence)
                augs.append(aug)
                targets.append(target)
            tt.update()

    df_2 = pd.DataFrame({"sentence": augs, "target": targets})
    df_all = pd.concat([df, df_2])
    name = args.data.split(".")[0]
    df_all.to_csv(f"{name}+_contextualized_aug", index = False)


if __name__ == "__main__":
    parser.add_argument(
        "--data",
        "-d",
        help="dataframe (sentence, target)",
        type = str,
    )

    parser.add_argument(
        "--cuda",
        "-c",
        help = "device",
        type = str,
        default = "cuda"
    )

    parser.add_argument(
        "--model_name",
        "-mn",
        help = "pretrained model",
        type = str,
        default = "roberta-base"
    )

    parser.add_argument(
        "--preprocessing",
        "-pp",
        help = "if preprocessing",
        type = "str",
        default = "true"
    )

    parser.add_argument(
        "--preprocessing_type",
        "-ppt",
        help = "type of the preprocessing",
        type = str,
        default = "denoise"
    )

    parser.add_argument(
        "--total",
        "-t",
        help = "number of augmentations",
        type = int,
        default = 10000
    )
    args = parser.parse_args()

    augment(args)
