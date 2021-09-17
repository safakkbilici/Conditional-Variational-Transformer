import torch
import pandas as pd
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, TensorDataset
from utils.preprocessing import denoise_text


def get_dataloaders(
        df_train,
        df_test,
        tokenizer,
        text_feature_name = "sentence",
        target_feature_name = "target",
        batch_size = 32,
        max_len = 200,
        preprocess = False):
    
    df_train = pd.read_csv(df_train)
    df_train = pd.read_csv(df_test)

    df_train_sentence = getattr(df_train, text_feature_name)
    df_train_target = getattr(df_train, target_feature_name)

    df_test_sentence = getattr(df_test, text_feature_name)
    df_test_target = getattr(df_test, target_feature_name)

    if preprocess:
        df_train_sentence = df_train_sentence.apply(denoise_text)
        df_test_sentence = df_test_sentence.apply(denoise_text)


    ids = []
    masks = []
    labels = []
    for sentence, target in zip(df_train_sentence, df_train_target):
        encoded = tokenizer.encode(sentence, max_len = max_len)
        data.append(encoded)
        labels.append(float(target))

    data = torch.Tensor(data)
    labels = torch.Tensor(labels)
    train_data = TensorDataset(data, data, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    
    ids = []
    masks = []
    labels = []
    for sentence, target in zip(df_test_sentence, df_test_target):
        encoded = tokenizer.encode(sentence, max_len = max_len)
        data.append(encoded)
        labels.append(float(target))

    data = torch.Tensor(data)
    labels = torch.Tensor(labels)
    test_data = TensorDataset(data, data, labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader
