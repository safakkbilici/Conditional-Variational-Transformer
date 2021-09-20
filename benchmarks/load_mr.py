from datasets import load_dataset
import pandas as pd


def get_mr_as_csv():
    dataset = load_dataset("rotten_tomatoes")
    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    train_df = pd.DataFrame({"sentence": train["text"], "target": train["label"]})
    test_df = pd.DataFrame({"sentence": test["text"], "target": test["label"]})
    val_df = pd.DataFrame({"sentence": val["text"], "target": val["label"]})

    train_df.to_csv("mr_train.csv", index = False)
    test_df.to_csv("mr_test.csv", index = False)
    val_df.to_csv("mr_val.csv", index = False)


get_mr_as_csv()
