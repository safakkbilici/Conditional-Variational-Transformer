import random
import numpy as np
import itertools

def powerset(s):
    x = len(s)
    p = []
    for i in range(1 << x):
        p.append([s[j] for j in range(x) if (i & (1 << j))])

    return p

def mask(tokens, mask_id, prob = 15):
    token_len = len(tokens)
    token_idx_list = list(range(token_len))
    n_mask = round(token_len * prob / 100)
    
    sampled_pre_mask_tokens = np.random.choice(
        token_idx_list,
        size = n_mask,
        replace = False
    ).tolist()

    for i in sampled_pre_mask_tokens:
        tokens[i] = mask_id

    return mask
    

def delete(tokens, mask_id, prob = 15):
    token_len = len(tokens)
    token_idx_list = list(range(token_len))
    n_mask = round(token_len * prob / 100)

    sampled_pre_mask_tokens = np.random.choice(
        token_idx_list,
        size = n_mask,
        replace = False
    ).tolist()

    
    for i in sampled_pre_mask_tokens:
        tokens.remove(tokens[i])

    return tokens
    
    
def rotate(tokens):
    token_len = len(tokens)
    new_list = []
    token_idx_list = list(range(token_len))

    base = random.choice(token_idx_list)
    new_list.extend(tokens[base:])
    new_list.extend(tokens[:base])
    return new_list


def add_noise(tokens, start_id, end_id, pad_id):
    implemented = [
        mask,
        rotate,
        delete
    ]

    perms = powerset(implemented)
    perms_idx_list = list(range(len(perms)))
    perms_choose = np.random.choice(
        perms_idx_list,
        size = 1,
        replace = False
    ).tolist()
    selected_perms = []
    for i in perms_choose:
        selected_perms.append(perms[i])

    print(selected_perms)
