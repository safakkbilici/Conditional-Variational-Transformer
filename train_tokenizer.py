from tokenizer.basic_tokenizer import BasicTokenizer
from tokenizer.bpe import BytePairTokenizer
from utils.preprocessing import denoise_text

import argparse
import pandas as pd
import os

def main(args):
    df = pd.read_csv(args.dataframe)
    sentence_feature = getattr(df, args.feature_name)
    
    if args.preprocess:
        sentence_feature = sentence_feature.apply(denoise_text)
        
    if args.tokenizer == "space":
        tokenizer = BasicTokenizer(
            unk_token = args.unk_token,
            start_token = args.start_token,
            end_token = args.end_token,
            pad_token = args.pad_token,
            cased = args.cased,
            feature_name = args.feature_name
        )

        tokenizer.fit(df)
        tokenizer.save("./tokenizer")

    elif args.tokenizer == "bpe":
        data = '\n'.join(sentence_feature.tolist())
        with open("./tokenizer/data.txt", "w") as f:
            f.wirte(data)

        tokenizer = BytePairTokenizer(
            unk_token = args.unk_token,
            start_token = args.start_token,
            end_token = args.end_token,
            pad_token = args.pad_token,
            vocab_size = args.vocab_size
        )

        tokenizer.train(data)
        tokenizer.save("./tokenizer", "vocab")

        os.remove("./tokenizer/data.txt")
    else:
        raise NotImplementedError()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer",
                        "-tok",
                        help="-space- for space based tokenizer, -bpe- for Byte Pair Encoding",
                        type = str,
                        default = "space"
    )

    parser.add_argument("--dataframe",
                        "-df",
                        help="/path/to/dataframe.csv",
                        type = str
    )
    

    parser.add_argument("--start_token",
                        "-st",
                        help="start token",
                        type = str,
                        default = "[START]"
    )

    parser.add_argument("--end_token",
                        "-et",
                        help="-end token",
                        type = str,
                        default = "[END]"
    )

    parser.add_argument("--pad_token",
                        "-pt",
                        help="padding token",
                        type = str,
                        default = "[PAD]"
    )

    parser.add_argument("--unk_token",
                        "-ut",
                        help="unknown token",
                        type = str,
                        default = "[UNK]"
    )

    parser.add_argument("--cased",
                        "-c",
                        help="cased or uncased",
                        type = bool,
                        default = False
    )

    parser.add_argument("--feature_name",
                        "-fn",
                        help="pandas dataframe's feature name, that contains sentences",
                        type = str,
                        default = "sentence"
    )

    parser.add_argument("--vocab_size",
                        "-vs",
                        help="pre-determined vocab size for byte pair tokenizer",
                        type = int,
                        default = 500_000
    )

    parser.add_argument("--preprocess",
                        "-pp",
                        help="preprocess",
                        type = bool,
                        default = False
    )
    
    args = parser.parse_args()
    main(args)