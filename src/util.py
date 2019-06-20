from collections import defaultdict
from hmm import HMM
import pandas as pd
import numpy as np
import random
import string
import os


def read_dataset(dataset):
    with open(dataset, "r") as f:
        words = f.read().split()

    return words


def clean_dataset(directory, dataset):
    words = read_dataset(directory + dataset)

    outfile = directory + "new/" + dataset.split(".txt")[0] + "_clean.csv"

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
    df['split'] = np.random.randn(df.shape[0], 1)

    df['is_duplicate'] = df.duplicated()     # no duplicated rows
    msk = np.random.rand(len(df)) <= 0.8

    train = df[msk].drop(['split', 'is_duplicate'], axis=1)
    test = df[~msk].drop(['split', 'is_duplicate'], axis=1)

    return train, test


def perturb():
    # Create a model for the test set
    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/big_clean.txt",
              typo_ds="../data/typo/new/test.csv")

    # typos = pd.read_csv("../data/typo/new/test.csv")
    #
    # typos_dict = defaultdict(lambda: list())
    #
    # for index, row in typos.iterrows():
    #     typos_dict[row[1]].append(row[0])

    cleaned = open("../data/texts/big_clean.txt", "r")

    if not os.path.exists("../data/texts/perturbated/"):
        os.makedirs("../data/texts/perturbated/")

    perturbed = open("../data/texts/perturbated/big_perturbed.txt", "w")

    # FIXME
    ## To remove after the change of hmm.model_error
    #hmm.error_model["p"]
    # probability that a word has an edit
    p = 0.10

    for line in cleaned:
        line_words = line.split()

        for i, word in enumerate(line_words):
            n = len(word)
            x = np.random.binomial(n, p)        # x ~ Bin(p, n)  number of errors to introduce in the word

            # choose two letter to change
            indices = np.random.choice(n, x, replace=False)
            indices = -np.sort(-indices)

            for j in range(x):
                r = np.random.random()  #FIXME now choose an edit randomly

                # insert a letter in a random position (after idx)
                if r < 0.33:
                    new_letter = random.choice(string.ascii_letters)
                    word = word[0:indices[j]] + new_letter + word[indices[j] + 1:]

                # delete a letter
                elif r > 0.66:
                    word = word[0:indices[j]] + word[indices[j] + 1:]

                # substitute a letter
                else:
                    l = list(word)
                    if not l[indices[j]] in hmm.error_model["sub"]:
                        l[indices[j]] = random.choice(string.ascii_letters).lower()
                    else:
                        l[indices[j]] = np.random.choice(list(hmm.error_model["sub"][l[indices[j]]].keys()))
                    word = "".join(l)

            line_words[i] = word

        line = " ".join(line_words)
        perturbed.write(line)

    perturbed.close()
    cleaned.close()
