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

    return tokens
    

def delete(tokens, mask_id, prob = 15):
    token_len = len(tokens)
    token_idx_list = list(range(token_len))
    n_mask = round(token_len * prob / 100)

    sampled_pre_mask_tokens = np.random.choice(
        token_idx_list,
        size = n_mask,
        replace = False
    ).tolist()
    
    new_tokens = []
    for i in range(token_len):
        if i not in sampled_pre_mask_tokens:
            new_tokens.append(tokens[i])
        
    return new_tokens
    
    
def rotate(tokens, mask_id, prob = 15):
    token_len = len(tokens)
    new_list = []
    token_idx_list = list(range(token_len))

    base = random.choice(token_idx_list)
    new_list.extend(tokens[base:])
    new_list.extend(tokens[:base])
    return new_list


def add_noise(tokens, mask_id, end_id, pad_id, prob = 15):

    max_len = len(tokens)
    start_id = tokens[0]
    end_idx = tokens.index(end_id)
    tokens = tokens[1:end_idx]

    new_tokens = []
    
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

    for i in perms_choose:
        if perms[i]:
            for f in perms[i]:
                tokens = f(tokens, mask_id, prob)

    new_tokens.append(start_id)
    new_tokens.extend(tokens)
    new_tokens.append(end_id)
    for i in range(len(new_tokens), max_len):
        new_tokens.append(pad_id)
    return new_tokens


#tokens = [1,5,7,8,9,5,6,77,88,99,55,4,22,456,2,0,0,0,0,0]
#new_tokens = add_noise(tokens, -1, 2, 0, 15)
#print(tokens)
#print(new_tokens)
