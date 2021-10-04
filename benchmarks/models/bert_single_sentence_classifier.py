import torch
import torch.nn as nn
from transformers import AutoModel
from tqdm.auto import tqdm
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    DataLoader,
    TensorDataset
)

import pandas as pd
import numpy as np

from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score
)

from collections import defaultdict
from copy import deepcopy

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


class BERTClassifier(nn.module):
    model_name: str
    n_classes: str

    def __init__(self, model_name, n_classes, p_dropout):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(768, n_classes)

    def forward(self, token_ids, attention_mask):
        out = self.bert(token_ids, attention_mask)
        out = out[0]
        out = out[:, 0, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
            
def dataloader_bert_classifier(tokenizer, batch_size, df_train, df_test):
    max_len = max([len(tokenizer.encode(sentence)) for sentence in tqdm(df_train.sentence)])
    print(f"Max token len for dataset: {max_len}")
    if max_len > 512:
        print(f"Max token len for BERT is 512")
        max_len = 512

    ids = []
    masks = []
    labels = []

    for idx, row in df_train.iterrows():
        encoded = tokenizer.encode_plus(
            text = row["sentence"],
            add_special_tokens = True,
            max_length = max_len,
            pad_to_max_length = True,
            return_attention_mask = True,
            truncation = True
        )

        ids.append(encoded.get('input_ids'))
        masks.append(encoded.get('attention_mask'))
        labels.append(float(row["target"]))

    ids = torch.Tensor(ids)
    masks = torch.Tensor(masks)
    labels = torch.Tensor(labels)
    train_data = TensorDataset(ids, masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    ids = []
    masks = []
    labels = []

    for idx, row in df_test.iterrows():
        encoded = tokenizer.encode_plus(
            text = row["sentence"],
            add_special_tokens = True,
            max_length = max_len,
            pad_to_max_length = True,
            return_attention_mask = True,
            truncation = True
        )

        ids.append(encoded.get('input_ids'))
        masks.append(encoded.get('attention_mask'))
        labels.append(float(row["target"]))

    ids = torch.Tensor(ids)
    masks = torch.Tensor(masks)
    labels = torch.Tensor(labels)
    test_data = TensorDataset(ids, masks, labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader
    

    
def train_bert_classifier(model, device, criterion, optimizer, epochs, n_classes, train_dataloader,
                          test_dataloader, sometimes_loss = 10, maximize = "acc", save_every = False):
    model = model.to(device)
    total = len(train_dataloader) * epochs

    results = defaultdict(list)
    results["f1_macro"] = []
    results["eval_loss"] = []
    results["acc"] = []

    f_ = np.argmax

    with tqdm(total = total) as tt:
        for epoch in range(epochs):
            train_loss, batch_count = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()

                token_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
                optimizer.zero_grad()

                out = model(token_ids.long(), attention_masks.long())
                loss = criterion(out, labels.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_count += 1

                if (batch_count+1) % sometimes_loss == 0:
                    print(f"{epoch+1}/{epochs}: {step}/{len(train_dataloader)} -> Loss: {train_loss / batch_count}")
                tt.update()


            f1_macro, accuracy, val_loss = evaluate_bert_classifier(
                model = model,
                device = device,
                criterion = criterion,
                test_dataloader = test_dataloader,
                n_classes = n_classes
            )

            print(25*"*")
            print(f"{epoch+1}/{epochs} -> F1 Macro: {f1_macro}")
            print(f"{epoch+1}/{epochs} -> Accuracy: {accuracy}")
            print(f"{epoch+1}/{epochs} -> Loss: {val_loss}")

            results["f1_macro"].append(f1_macro)
            results["eval_loss"].append(val_loss)
            results["acc"].append(accuracy)

            if results[maximize][f_(results[maximize])] == results[maximize][-1]:
                torch.save(model.state_dict(), "bert_classifier_best_model.pt")
                store = results[maximize][-1]
                print(f"Best model is stored with {maximize}: {store}")

            if save_every:
                torch.save(model.state_dict(), "bert_classifier.pt")

            print(25*"*")
    return results

def evaluate_bert_classifier(model, device, criterion, test_dataloader, n_classes):
    eval_loss, batch_count = 0, 0
    y_pred = []
    y_true = []

    with tqdm(total = len(test_dataloader), leave=False) as ee:
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                model.eval()
                token_ids, attention_masks, labels = tuple(t.to(device) for t in batch)

                out = model(token_ids.long(), attention_masks.long())
                loss = criterion(out, labels.long())
                eval_loss += loss.item()
                batch_count += 1

                preds = torch.argmax(out, dim=-1).flatten().cpu().numpy()
                labels_view = labels.view(-1).cpu().numpy()

                preds_onehot = np.zeros((preds.size, n_classes))
                gt_onehot = np.zeros((labels_view.size, n_classes))
                preds_onehot[np.arange(preds.size), preds.astype(int)] = 1
                gt_onehot[np.arange(labels_view.size), labels_view.astype(int)] = 1

                y_pred.extend(preds_onehot)
                y_true.extend(gt_onehot)

                ee.update()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro = f1_score(y_true, y_pred, average = "macro")
    acc = accuracy_score(y_true, y_pred)
    eval_loss = eval_loss / batch_count

    return macro, acc, eval_loss
