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
        self.vocab_len = len(self.w2i)
        
    def fit(self, df):
        df_feature = getattr(df, self.feature_name)
        id_count = 4
        for sentence in df_feature:
            splitted = sentence.split()
            for word in splitted:
                if word not in self.w2i.keys():
                    if self.cased == False:
                        self.w2i[word.lower()] = id_count
                    else:
                        self.w2i[word] = id_count
                    id_count +=1
        self.i2w = {v: k for k, v in self.w2i.items()}
        
    def encode(self,sentence, max_len = 512):
        if self.cased:
            sentence = sentence.lower()
        s_len = 2
        encoded = []
        encoded.append(self.start_token_id)
        splitted = denoise_text(sentence).split()
        for word in splitted:
            s_len+=1
            if word not in self.w2i:
                encoded.append(self.unk_token_id)
            else:
                encoded.append(self.w2i[word])
        encoded.append(self.end_token_id)
        if max_len != None:
            for _ in range(s_len, max_len):
                encoded.append(self.pad_token_id)
        return encoded
    
    def decode(self, token_ids, remove_special_characters = False):
        decoded = []
        for token_id in token_ids:
            decoded.append(self.i2w[token_id])
        if remove_special_characters:
            self.remove_special_characters(decoded)
        return decoded
    
    def remove_special_characters(self, tokens, from_str = "False"):
        if not from_str:
            for token in self.special_tokens:
                tokens = list(filter((token).__ne__, tokens))
                
        else:
            tokens = tokens.split()
            for token in self.special_tokens:
                tokens = list(filter((token).__ne__, tokens))
            tokens = ' '.join(tokens)
        return tokens