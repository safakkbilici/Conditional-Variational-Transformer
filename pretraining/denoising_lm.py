import numpy as np
from torch.utils.data import Dataset
from tokenizer.noise import add_noise
import torch

def tokenize_corpus(tokenizer, corpus, max_seq_len, pad_token_id, end_token_id, start_token_id):
    with open(corpus, "r") as f:
        data = f.read()
    ids = []
    ids.extend(tokenizer.encode(data, None)[1:-1])
    seq_data = []
    ids = ids[1:-1]
    for i in range(0, len(ids) - max_seq_len - 2, max_seq_len - 2):
        seq_data.extend([[tokenizer.start_token_id] + ids[i:i+max_seq_len-1] + [tokenizer.end_token_id]])

    pad_len = max_seq_len - (len(ids) - (i+max_seq_len-2))
    a = [tokenizer.start_token_id] + ids[i+max_seq_len-2:-1] + [tokenizer.end_token_id]
    a = a + [tokenizer.pad_token_id for _ in range(pad_len)]
    seq_data.append(a)
        
    return seq_data

class DenoisingLM(object):
    def __init__(
            self,
            mask_token,
            mask_token_id,
            end_token_id,
            pad_token_id,
            prob_mask = 15,
            prob_delete = 15,
            max_seq_len = 512
    ):
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
        self.prob_mask = prob_mask
        self.prob_delete = prob_delete
        self.max_seq_len = max_seq_len

    def noise(self, tokens):
        tokens = add_noise(
            tokens = tokens,
            mask_id = self.mask_token_id,
            end_id = self.end_token_id,
            pad_id = self.pad_token_id,
            prob_mask = self.prob_mask,
            prob_delete = self.prob_delete
        )
        
        return tokens


class DenoisingLMDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            corpus,
            objective: DenoisingLM,
            max_seq_len,
            pad_token_id,
            end_token_id,
            start_token_id,
    ):
        self.corpus_tokenized_splitted = tokenize_corpus(
            tokenizer = tokenizer,
            corpus = corpus,
            max_seq_len = max_seq_len,
            pad_token_id = pad_token_id,
            end_token_id = end_token_id,
            start_token_id = start_token_id
        )
        self.objective = objective

    def __len__(self):
        return len(self.corpus_tokenized_splitted)

    def __getitem__(self, idx):
        segment = self.corpus_tokenized_splitted[idx]
        noised_segment = self.objective.noise(segment)

        return {
            'original_segment': torch.LongTensor(segment),
            'noised_segment': torch.LongTensor(noised_segment)
        }
