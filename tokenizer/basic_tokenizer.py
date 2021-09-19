import json

class BasicTokenizer():
    def __init__(self, 
                 unk_token = "[UNK]", 
                 start_token = "[START]",
                 end_token = "[END]",
                 pad_token = "[PAD]",
                 cased = True,
                 feature_name = "sentence"):
        
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.feature_name = feature_name
        self.cased = cased
        
        self.unk_token_id = 3
        self.start_token_id = 2
        self.end_token_id = 1
        self.pad_token_id = 0
        self.w2i = {
            self.start_token: self.start_token_id,
            self.end_token: self.end_token_id,
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id
        }
        self.special_tokens = list(self.w2i.keys())
        
        
    def fit(self, df):
        df_feature = getattr(df, self.feature_name)
        id_count = 4
        for sentence in df_feature:
            splitted = sentence.split()
            for word in splitted:
                if self.cased == False:
                    word = word.lower()
                if word not in self.w2i.keys():
                    self.w2i[word] = id_count
                    #print(word, id_count)
                    id_count +=1
        self.i2w = {v: k for k, v in self.w2i.items()}
        print(len(self.w2i))
        self.vocab_len = len(self.w2i)
        
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
            self.remove_special_tokens(decoded)
        return decoded
    
    def remove_special_tokens(self, tokens, from_str = False):
        if not from_str:
            for token in self.special_tokens:
                tokens = list(filter((token).__ne__, tokens))
                
        else:
            tokens = tokens.split()
            for token in self.special_tokens:
                tokens = list(filter((token).__ne__, tokens))
            tokens = ' '.join(tokens)
        return tokens


    def save(self, directory):
        if directory[-1] != "/":
            directory += "/"
        with open(directory + "vocab.json", "w") as vocab_f:
            json.dump(self.w2i, vocab_f)

        with open(directory + "special_tokens.json", "w") as special_f:
            json.dump(self.special_tokens, special_f)

        params = {
            "unk_token": self.unk_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "pad_token": self.pad_token,
            "cased": self.cased,
            "unk_token_id": self.unk_token_id,
            "start_token_id": self.start_token_id,
            "end_token_id": self.end_token_id,
            "pad_token_id": self.pad_token_id,
        }
        with open(directory + "params.json", "w") as params_f:
            json.dump(params, params_f)

    def load(self, directory):
        if directory[-1] != "/":
            directory += "/"

        with open(directory + "vocab.json", "r") as vocab_f:
            self.w2i = json.load(vocab_f)

        self.i2w = {v: k for k, v in self.w2i.items()}

        with open(directory + "special_tokens.json", "r") as special_f:
            self.special_tokens = json.load(special_f)

        with open(directory + "params.json", "r") as params_f:
            params = json.load(params_f)

        self.unk_token = params["unk_token"]
        self.start_token = params["start_token"]
        self.end_token = params["end_token"]
        self.pad_token = params["pad_token"]
        self.cased = params["cased"]
        self.unk_token_id = params["unk_token_id"]
        self.start_token_id = params["start_token_id"]
        self.end_token_id = params["end_token_id"]
        self.pad_token_id = params["pad_token_id"]
