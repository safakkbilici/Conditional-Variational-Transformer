from datasets import load_dataset
import pandas as pd


def get_cola_as_csv():
    dataset = load_dataset("glue", "cola")
    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    train_df = pd.DataFrame({"sentence": train["sentence"], "target": train["label"]})
    test_df = pd.DataFrame({"sentence": test["sentence"], "target": test["label"]})
    val_df = pd.DataFrame({"sentence": val["sentence"], "target": val["label"]})

    train_df.to_csv("cola_train.csv", index = False)
    test_df.to_csv("cola_test.csv", index = False)
    val_df.to_csv("cola_val.csv", index = False)


get_cola_as_csv()
