from tokenizer.base import TokenizerBase
import json
import pandas as pd

class SpaceTokenizer(TokenizerBase):
    def __init__(
            self,
            start_token = "[START]",
            end_token = "[END]",
            pad_token = "[PAD]",
            unk_token = "[UNK]",
            mask_token = "[MASK]",
            cased = True
    ):
        super(SpaceTokenizer, self).__init__(
            start_token = start_token,
            end_token = end_token,
            pad_token = pad_token,
            unk_token = unk_token,
            mask_token = mask_token
        )

        self.w2i = {
            self.start_token: self.start_token_id,
            self.end_token: self.end_token_id,
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.mask_token: self.mask_token_id
        }

        self.cased = cased


    def fit(self, data, feature_name = None):
        id_count = 5
        
        if type(data) == str:
            with open(data, "r") as f:
                data = f.read()

            data = data.split()
            for word in data:
                if self.cased == False:
                    word = word.lower()
                if word not in self.w2i.keys():
                    self.w2i[word] = id_count
                    id_count +=1
                    
        elif type(data) == pd.DataFrame:
            try:
                df_feature = getattr(data, "text")
            except:
                df_feature = getattr(data, feature_name)

            for sentence in df_feature:
                splitted = sentence.split()
                for word in splitted:
                    if self.cased == False:
                        word = word.lower()
                    if word not in self.w2i.keys():
                        self.w2i[word] = id_count
                        id_count +=1

        self.i2w = {v: k for k, v in self.w2i.items()}
        print(f"Vocab size: {len(self.w2i)}")
        self.vocab_size = len(self.w2i)

    def encode(self,sentence, max_len = 512):
        if self.cased == False:
            sentence = sentence.lower()
        s_len = 2
        encoded = []
        encoded.append(self.start_token_id)
        splitted = sentence.split()
        for word in splitted:
            s_len+=1
            if word not in self.w2i:
                encoded.append(self.unk_token_id)
            else:
                encoded.append(self.w2i[word])
            if max_len != None and s_len == max_len-1:
                break
        encoded.append(self.end_token_id)
        if max_len != None:
            for _ in range(s_len, max_len):
                encoded.append(self.pad_token_id)
        return encoded

    def decode(self, token_ids, remove_special_tokens = False):
        decoded = []
        for token_id in token_ids:
            decoded.append(self.i2w[token_id])
        if remove_special_tokens:
            decoded = self.remove_special_tokens(decoded)
        return decoded
    
    def remove_special_tokens(self, tokens, from_str = False):
        if not from_str:
            for token in self.get_special_tokens():
                tokens = list(filter((token).__ne__, tokens))
                
        else:
            tokens = tokens.split()
            for token in self.get_special_tokens():
                tokens = list(filter((token).__ne__, tokens))
            tokens = ' '.join(tokens)
        return tokens

    def save(self, directory):
        if directory[-1] != "/":
            directory += "/"
        with open(directory + "vocab.json", "w") as vocab_f:
            json.dump(self.w2i, vocab_f)

        params = dict(
            unk_token = self.unk_token,
            start_token = self.start_token,
            end_token = self.end_token,
            pad_token = self.pad_token,
            mask_token = self.mask_token,
            cased = self.cased,
            unk_token_id = self.unk_token_id,
            start_token_id = self.start_token_id,
            end_token_id = self.end_token_id,
            pad_token_id = self.pad_token_id,
            mask_token_id = self.mask_token_id
        )
        with open(directory + "params.json", "w") as params_f:
            json.dump(params, params_f)

    def load(self, directory):
        if directory[-1] != "/":
            directory += "/"

        with open(directory + "vocab.json", "r") as vocab_f:
            self.w2i = json.load(vocab_f)

        self.i2w = {v: k for k, v in self.w2i.items()}

        with open(directory + "params.json", "r") as params_f:
            params = json.load(params_f)
        self.cased = params["cased"]
        self.vocab_size = len(self.w2i)
