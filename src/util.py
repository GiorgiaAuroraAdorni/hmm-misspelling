import numpy as np


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
    # lowercase
    df = combined_csv.apply(lambda x: x.astype(str).str.lower())

    # shuffle data frame
    df.reindex(np.random.permutation(df.index))

    df['split'] = np.random.randn(df.shape[0], 1)

    df['is_duplicate'] = df.duplicated()     # no duplicated rows
    msk = np.random.rand(len(df)) <= 0.8

    train = df[msk].drop(['split', 'is_duplicate'], axis=1)
    test = df[~msk].drop(['split', 'is_duplicate'], axis=1)

    return train, test
