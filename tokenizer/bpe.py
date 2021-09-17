import tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
import json


class BytePairTokenizer():
    def __init__(self, start_token = "[START]", end_token = "[END]",
                 unk_token = "[UNK]", pad_token = "[PAD]", vocab_size = 500_000):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.special_tokens = [self.start_token, self.end_token, self.pad_token, self.unk_token]
        
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = Whitespace()


    def train(self, data):
        trainer = BpeTrainer(special_tokens = self.special_tokens, vocab_size = self.vocab_size)

        self.tokenizer.train(files=[data], trainer=trainer)

        self.start_token_id = self.tokenizer.token_to_id(self.start_token)
        self.end_token_id = self.tokenizer.token_to_id(self.end_token)
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        self.unk_token_id = self.tokenizer.token_to_id(self.unk_token)


    def save(self, directory, name):
        if directory[-1] != "/":
            directory += "/"

        self.tokenizer.save(directory + name + ".json")
        special_tokens = {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,

            "start_token_id": self.start_token_id,
            "end_token_id": self.end_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            
        }

        with open(directory + name + "_" + "special_tokens.json", "w") as special_f:
            json.dump(special_tokens, special_f)

    def load(self, directory, name):
        if directory[-1] != "/":
            directory += "/"

        self.tokenizer = Tokenizer.from_file(directory + name + ".json")
        with open(directory + name + "_" + "special_tokens.json", "r") as special_f:
            self.special_tokens = json.load(special_f)

        self.start_token = self.special_tokens["start_token"]
        self.end_token = self.special_tokens["end_token"]
        self.pad_token = self.special_tokens["pad_token"]
        self.unk_token = self.special_tokens["unk_token"]

        self.start_token_id = self.special_tokens["start_token_id"]
        self.end_token_id = self.special_tokens["end_token_id"]
        self.pad_token_id = self.special_tokens["pad_token_id"]
        self.unk_token_id = self.special_tokens["unk_token_id"]


    def encode(self, sentence, max_len = 512, preprocess = None):
        if preprocess != None:
            sentence = preprocess(sentence)

        encoded = []
        encoded.append(self.start_token_id)

        out = self.tokenizer.encode(sentence)
        encoded_sentence = out.ids
        encoded.extend(encoded_sentence)

        encoded.append(self.end_token_id)

        if max_len != None:
            for _ in range(len(encoded), max_len):
                encoded.append(self.pad_token_id)

        return encoded

    def decode(self, token_ids, remove_special_tokens = False):
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens = remove_special_tokens)
        return decoded
