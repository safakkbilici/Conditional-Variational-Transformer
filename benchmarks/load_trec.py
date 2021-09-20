from datasets import load_dataset
import pandas as pd


def get_trec_as_csv():
    dataset = load_dataset("trec")
    train = dataset["train"]
    test = dataset["test"]

    train_df = pd.DataFrame({"sentence": train["text"], "label-coarse": train["label-coarse"], "label-fine": train["label-fine"]})
    test_df = pd.DataFrame({"sentence": test["text"], "label-coarse": test["label-coarse"], "label-fine": test["label-fine"]})

    train_df.to_csv("trec_train.csv", index = False)
    test_df.to_csv("trec_test.csv", index = False)


get_trec_as_csv()
