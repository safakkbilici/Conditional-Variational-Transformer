from datasets import load_dataset
import pandas as pd


def get_agnews_as_csv():
    dataset = load_dataset("ag_news")
    train = dataset["train"]
    test = dataset["test"]

    train_df = pd.DataFrame({"sentence": train["text"], "target": train["label"]})
    test_df = pd.DataFrame({"sentence": test["text"], "target": test["label"]})

    train_df.to_csv("ag_news_train.csv", index = False)
    test_df.to_csv("ag_news_test.csv", index = False)


get_agnews_as_csv()
