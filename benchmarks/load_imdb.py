from datasets import load_dataset
import pandas as pd


def get_imdb_as_csv():
    dataset = load_dataset("imdb")
    train = dataset["train"]
    test = dataset["test"]

    train_df = pd.DataFrame({"sentence": train["text"], "target": train["label"]})
    test_df = pd.DataFrame({"sentence": test["text"], "target": test["label"]})

    train_df.to_csv("imdb_train.csv", index = False)
    test_df.to_csv("imdb_test.csv", index = False)


get_imdb_as_csv()
