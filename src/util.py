import numpy as np
import re


def read_dataset(dataset):
    with open(dataset, "r") as f:
        words = f.read().split()

    return words


def clean_dataset(dataset):
    words = read_dataset(dataset)

    outfile = dataset.split(".txt")[0] + "_clean.csv"

    with open(outfile, "w+") as new_f:
        correct_word = ""
        for word in words:
            if word.startswith("$"):
                correct_word = word.split("$")[1]
            elif word.endswith(":"):
                correct_word = word.split(":")[0]
            else:
                new_f.write(word)
                new_f.write(",")
                new_f.write(correct_word)
                new_f.write("\n")


def split_dataset(combined_csv):
    combined_csv['split'] = np.random.randn(combined_csv.shape[0], 1)

    combined_csv['is_duplicate'] = combined_csv.duplicated()     # no duplicated rows
    msk = np.random.rand(len(combined_csv)) <= 0.8

    train = combined_csv[msk].drop(['split', 'is_duplicate'], axis=1)
    test = combined_csv[~msk].drop(['split', 'is_duplicate'], axis=1)

    return train, test
