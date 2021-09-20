from datasets import load_dataset
import pandas as pd


def get_yelp_as_csv():
    dataset = load_dataset("yelp_review_full")
    train = dataset["train"]
    test = dataset["test"]

    train_df = pd.DataFrame({"sentence": train["text"], "target": train["label"]})
    test_df = pd.DataFrame({"sentence": test["text"], "target": test["label"]})

    train_df.to_csv("yelp_train.csv", index = False)
    test_df.to_csv("yelp_test.csv", index = False)
    val_df.to_csv("yelp_val.csv", index = False)


get_yelp_as_csv()
