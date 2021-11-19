from tokenizer.base import TokenizerBase

import tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE

import json
import pandas as pd
import os

class BytePairTokenizer(TokenizerBase):
    def __init__(
            self,
            start_token = "[START]",
            end_token = "[END]",
            pad_token = "[PAD]",
            unk_token = "[UNK]",
            mask_token = "[MASK]",
            cased = True,
            vocab_size = 500_000
    ):
        super(BytePairTokenizer, self).__init__(
            start_token = start_token,
            end_token = end_token,
            pad_token = pad_token,
            unk_token = unk_token,
            mask_token = mask_token
        )


        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = Whitespace()

    def fit(self, data, feature_name = None):
        if type(data) == str:
            trainer = BpeTrainer(
                special_tokens = self.get_special_tokens(),
                vocab_size = self.vocab_size
            )

            self.tokenizer.train(files=[data], trainer=trainer)

            assert self.start_token_id == self.tokenizer.token_to_id(self.start_token)
            assert self.end_token_id == self.tokenizer.token_to_id(self.end_token)
            assert self.pad_token_id == self.tokenizer.token_to_id(self.pad_token)
            assert self.unk_token_id == self.tokenizer.token_to_id(self.unk_token)

            self.vocab_size = self.tokenizer.get_vocab_size()
            print(f"Vocab size: {self.vocab_size}")

        elif type(data) == pd.DataFrame:
            try:
                df_feature = getattr(data, "text")
            except:
                df_feature = getattr(data, feature_name)

            data = []
            for sentence in df_feature:
                data.append(sentence)

            with open("__data__.txt", "w") as f:
                f.write(data)

            trainer = BpeTrainer(
                special_tokens = self.get_special_tokens(),
                vocab_size = self.vocab_size
            )

            self.tokenizer.train(files=["__data__.txt"], trainer=trainer)

            assert self.start_token_id != self.tokenizer.token_to_id(self.start_token)
            assert self.end_token_id != self.tokenizer.token_to_id(self.end_token)
            assert self.pad_token_id != self.tokenizer.token_to_id(self.pad_token)
            assert self.unk_token_id != self.tokenizer.token_to_id(self.unk_token)

            self.vocab_size = self.tokenizer.get_vocab_size()
            print(f"Vocab size: {self.vocab_size}")
            os.remove("__data__.txt")

        
    def save(self, directory, name):
        if directory[-1] != "/":
            directory += "/"
        self.tokenizer.save(directory + name + ".json")


    def load(self, directory, name):
        if directory[-1] != "/":
            directory += "/"
        self.tokenizer = Tokenizer.from_file(directory + name + ".json")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, sentence, max_len = 512):

        encoded = []
        encoded.append(self.start_token_id)

        out = self.tokenizer.encode(sentence)
        encoded_sentence = out.ids
        encoded.extend(encoded_sentence)

        encoded.append(self.end_token_id)
        if len(encoded) > max_len:
            encoded = encoded[:max_len -1]
            encoded.append(self.end_token_id)

        elif max_len != None:
            for _ in range(len(encoded), max_len):
                encoded.append(self.pad_token_id)

        return encoded

    def decode(self, token_ids, remove_special_tokens = False):
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens = remove_special_tokens)
        return decoded
        
