from collections import Counter, defaultdict
from hmm import HMM
import numpy as np
import random
import string
import time
import csv


def read_dataset(dataset):
    with open(dataset, "r") as f:
        words = f.read().split()

    return words


def clean_dataset(directory, dataset):
    words = read_dataset(directory + dataset)

    outfile = directory + "clean/" + dataset.split(".txt")[0] + "_clean.csv"

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


def extract_model_edit_probabilities(hmm):
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

    return p, edit_prob


def perturb_word(edit_prob, indices, word, x, hmm):

    def substitute(word):
        l = list(word)
        if not l[indices[j]] in hmm.error_model["sub"]:
            l[indices[j]] = random.choice(string.ascii_letters).lower().replace(l[indices[j]], "")
        else:
            l[indices[j]] = np.random.choice(list(hmm.error_model["sub"][l[indices[j]]].keys()-l[indices[j]]))
        return "".join(l)

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
                word = word[0:indices[j] - 1] + word[indices[j]] + word[indices[j] - 1] + word[indices[j] + 1:]
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
    return word


def perturb_file(perturbed, rumor_percentage):
    # Create a model for the test set
    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/test.csv")

    with open("../data/texts/lotr_clean.txt", "r") as myfile:
        cleaned = myfile.readlines()

    p, edit_prob = extract_model_edit_probabilities(hmm)

    p = p * rumor_percentage  # introduce a certain percentage of error (10-15-20%)

    print("\n Starting pertub…")
    start = time.time()

    for line in cleaned:
        line = line.replace("\n", "")
        line_words = line.split()

        for i, word in enumerate(line_words):
            n = len(word)
            # number of errors to introduce in the word
            x = np.random.binomial(n, p)        # x ~ Bin(p, n)

            # choose two letter to change
            indices = np.random.choice(n, x, replace=False)
            indices = -np.sort(-indices)

            word = perturb_word(edit_prob, indices, word, x, hmm)

            line_words[i] = word

        line = " ".join(line_words)
        perturbed.write(line + '\n')

    end = time.time()
    perturb_time = end - start

    perturbed.close()

    print("Endend pertubation in {:6.2f} seconds \n".format(perturb_time))


def create_model_language():
    with open("../data/texts/lotr_clean.txt", "r") as myfile:
        cleaned = myfile.readlines()

    counter = Counter()
    total_word = 0

    for line in cleaned:
        for word in line.split():
            counter[word] += 1
            total_word += 1

    for el in counter:
        counter[el] /= total_word

    with open("../data/word_freq/lotr_language_model.txt", "w") as file:
        writer = csv.writer(file)

        for key, value in counter.items():
            writer.writerow([key, value])


def split_list(lst, n):
    result = list()

    while lst:
        if not len(lst[:n]) < n/2:
            result.append(" ".join(lst[:n]))
        else:
            result[-1] += " " + " ".join(lst[:n])
        lst = lst[n:]

    return result


def create_typo_dataset(typo_writer):
    # Create a model for the test set
    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    with open("../data/word_freq/lotr_language_model.txt", "r") as myfile:
        reader = csv.reader(myfile)
        words_ds = [row for row in reader]

    typo_ds = defaultdict()

    _, edit_prob = extract_model_edit_probabilities(hmm)

    print("\n Starting pertub…")
    start = time.time()

    for word in words_ds:
        word = word[0]
        orig_word = word
        typos = list()

        for idx in range(5):
            word = orig_word
            n = len(word)
            # number of errors to introduce in the word
            if n < 3:
                x = 1
            else:
                x = np.random.randint(2) + 1

            # choose two letter to change
            indices = np.random.choice(n, x, replace=False)
            indices = -np.sort(-indices)

            word = perturb_word(edit_prob, indices, word, x, hmm)

            typos.append(word)

        typo_ds[orig_word] = typos

    end = time.time()
    perturb_time = end - start

    for key, value in typo_ds.items():
        for v in value:
            typo_writer.writerow([v, key])

    print("Endend pertubation in {:6.2f} seconds \n".format(perturb_time))
