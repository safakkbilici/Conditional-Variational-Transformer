import torch
import pandas as pd
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, TensorDataset
from utils.preprocessing import denoise_text
from tokenizer.noise import add_noise
import functools


def get_dataloaders(
        df_train,
        df_test,
        tokenizer,
        text_feature_name = "sentence",
        target_feature_name = "target",
        batch_size = 32,
        max_len = 200,
        preprocess = False,
        preprocess_type = "denoise",
        noise = True):
    
    df_train = pd.read_csv(df_train)
    df_test = pd.read_csv(df_test)

    df_train_sentence = getattr(df_train, text_feature_name)
    df_train_target = getattr(df_train, target_feature_name)

    df_test_sentence = getattr(df_test, text_feature_name)
    df_test_target = getattr(df_test, target_feature_name)

    if preprocess:
        preprocessor = functools.partial(denoise_text, t = preprocess_type)
        df_train_sentence = df_train_sentence.apply(preprocessor)
        df_test_sentence = df_test_sentence.apply(preprocess)


    data = []
    noised = []
    labels = []
    for sentence, target in zip(df_train_sentence, df_train_target):
        encoded = tokenizer.encode(sentence, max_len = max_len)
        if noise:
            encoded1 = add_noise(encoded, tokenizer.mask_token_id, tokenizer.end_token_id, tokenizer.pad_token_id)
            noised.append(encoded1)
        data.append(encoded)
        labels.append(float(target))

    data = torch.Tensor(data)
    if noise:
        noised = torch.Tensor(noised)
    labels = torch.Tensor(labels)
    if noise:
        train_data = TensorDataset(noised, data, labels)
    else:
        train_data = TensorDataset(data, data, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    
    data = []
    labels = []
    noised = []
    for sentence, target in zip(df_test_sentence, df_test_target):
        encoded = tokenizer.encode(sentence, max_len = max_len)
        if noise:
            encoded1 = add_noise(encoded, tokenizer.mask_token_id, tokenizer.end_token_id, tokenizer.pad_token_id)
            noised.append(encoded1)
        data.append(encoded)
        labels.append(float(target))

    data = torch.Tensor(data)
    if noise:
        noised = torch.Tensor(noised)
    labels = torch.Tensor(labels)
    if noise:
        test_data = TensorDataset(noised, data, labels)
    else:
        test_data = TensorDataset(data, data, labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader
