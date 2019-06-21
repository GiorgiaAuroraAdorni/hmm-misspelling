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

    cleaned = open("../data/texts/big_clean.txt", "r")

    if not os.path.exists("../data/texts/perturbated/"):
        os.makedirs("../data/texts/perturbated/")

    perturbed = open("../data/texts/perturbated/big_perturbed.txt", "w")

    # probability that a word has an edit
    p = hmm.error_model["p"]

    # probability of the various edit
    prob_swap = hmm.error_model["swap"]
    prob_ins = hmm.error_model["ins"]
    prob_del = hmm.error_model["del"]
    prob_sub = 1 - (prob_swap + prob_ins + prob_del)

    edit_prob = [prob_swap, prob_ins, prob_del, prob_sub]

    for i, e in enumerate(edit_prob):
        if i == 0:
            continue

        edit_prob[i] = edit_prob[i] + edit_prob[i - 1]

    def substitute(word):
        l = list(word)
        if not l[indices[j]] in hmm.error_model["sub"]:
            l[indices[j]] = random.choice(string.ascii_letters).lower()
        else:
            l[indices[j]] = np.random.choice(list(hmm.error_model["sub"][l[indices[j]]].keys()))
        return "".join(l)

    for line in cleaned:
        line_words = line.split()

        for i, word in enumerate(line_words):
            n = len(word)
            # number of errors to introduce in the word
            x = np.random.binomial(n, p)        # x ~ Bin(p, n)

            # choose two letter to change
            indices = np.random.choice(n, x, replace=False)
            indices = -np.sort(-indices)

            for j in range(x):
                r = np.random.random()

                for k, e in enumerate(edit_prob):
                    if r <= edit_prob[k]:
                        break
                value = k

                # swap if you have to do only one edit
                if value == 0 and x == 1:
                    # if the letter to switch is the last one, switch with the previous one
                    if len(indices) <= j + 1:
                        word = word[0:indices[j] - 1] + word[indices[j]] + word[indices[j] - 1] +  word[indices[j] + 1:]
                    else:
                        word = word[0:indices[j]] + word[indices[j] + 1] + word[indices[j]] + word[indices[j] + 2:]

                # insert a letter in a random position (after idx)
                elif value == 1:
                    new_letter = random.choice(string.ascii_letters)
                    word = word[0:indices[j]] + new_letter + word[indices[j] + 1:]

                # delete a letter
                elif value == 2:
                    if len(word) == 1:
                        # if the word is 1 char, don't delete the word but substitute it with another one
                        word = substitute(word)
                    else:
                        word = word[0:indices[j]] + word[indices[j] + 1:]

                # substitute a letter
                else:
                    word = substitute(word)

            line_words[i] = word

        line = " ".join(line_words)
        perturbed.write(line + '\n')

    perturbed.close()
    cleaned.close()
