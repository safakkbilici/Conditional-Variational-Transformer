class TokenizerBase(object):
    def __init__(
            self,
            unk_token = "[UNK]", 
            start_token = "[START]",
            end_token = "[END]",
            pad_token = "[PAD]",
            mask_token = "[MASK]"
    ) -> None:
        
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.start_token_id = 1
        self.end_token_id = 2
        self.pad_token_id = 3
        self.unk_token_id = 0
        self.mask_token_id = 4

        self.special_tokens = {
            self.unk_token: self.unk_token_id,
            self.start_token: self.start_token_id,
            self.end_token: self.end_token_id,
            self.pad_token: self.pad_token_id,
            self.mask_token: self.mask_token_id
        }

    def get_special_tokens(self):
        return list(self.special_tokens.keys())

    def get_special_token_ids(self):
        return list(self.special_tokens.values())
