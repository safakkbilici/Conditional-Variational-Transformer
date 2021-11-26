import numpy as np
from torch.utils.data import Dataset

def tokenize_corpus(tokenizer, corpus, max_seq_len, pad_token_id):
    with open(corpus, "r") as f:
        data = f.read().splitlines()
    ids = []
    data = data.split()
    for w in data:
        ids.extend(tokenizer.encode(w), None)

    seq_data = []
    for i in range(0, len(data) - max_seq_len, max_seq_len):
        try:
            seq_data.append([data[i:i+max_seq_len]])
        except:
            pad_len = max_seq_len - (len(data) - i)
            a = [data[i:-1]] + [pad_token_id for _ in range(pad_len)]
            # break

    return seq_data

class MaskedLanguageModeling(object):
    def __init__(
            self,
            mask_token,
            mask_token_id,
            end_token_id,
            prob = 15,
            max_seq_len = 512
    ):
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self.end_token_id = end_token_id
        self.prob = prob
        self.max_seq_len = max_seq_len

    def mask(self, tokens):
        start_id = tokens[0]
        end_idx = tokens.index(self.end_token_id)
        tokens = tokens[1:end_idx]

        token_len = len(tokens)
        token_idx_list = list(range(token_len))
        n_mask = round(token_len * self.prob / 100)

        sampled_pre_mask_tokens = np.random.choice(
            token_idx_list,
            size = n_mask,
            replace = False
        ).tolist()

        for i in sampled_pre_mask_tokens:
            tokens[i] = self.mask_id

        return tokens, sampled_pre_mask_tokens


class MLMDataLoader(Dataset):
    def __init__(
            self,
            tokenizer,
            corpus,
            mlm: MaskedLanguageModeling,
            max_seq_len,
            pad_token_id
    ):
        self.corpus_tokenized_splitted = tokenize_corpus(
            tokenizer = tokenizer,
            corpus = tokenizer,
            max_seq_len = tokenizer,
            pad_token_id = pad_token_id
        )
        self.mlm = mlm

    def __len__(self):
        return len(self.corpus_tokenized_splitted)

    def __getitem__(self, idx):
        segment = self.corpus_tokenized_splitted[idx]
        masked_segment, masked_token_ids = mlm.mask(segment)

        return {
            'segment': torch.LongTensor(masked_segment)
            'masked_token_ids': torch.Tensor(masked_token_ids)
        }
