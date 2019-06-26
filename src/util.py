from collections import Counter, defaultdict
from hmm import HMM
import pandas as pd
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


def clean_sentences_dataset(ds):
    ds = ds.replace("Mr.", "Mr").replace("Mrs.", "Mrs")
    ds = ds.split(".")
    splitted = list()

    for line in ds:
        if len(line.split()) == 0:
            continue
        elif len(line.split()) > 10:
            line = split_list(line.split(), 10)

            splitted.extend(line)
        else:
            splitted.append(line)

    cleaned = [r.strip().lower().replace("'", '') for r in splitted]
    cleaned = [re.sub(r"[^a-zA-Z0-9]+", ' ', r) for r in cleaned]

    return cleaned


def extract_model_edit_probabilities(hmm):
    # probability that a word has an edit
    p = hmm.error_model["p"]

    # probability of the various edit
    prob_swap = hmm.error_model["swap"]
    prob_ins = hmm.error_model["ins"]
    prob_del = hmm.error_model["del"]
    prob_sub = hmm.error_model["sub"]

    edit_prob = [prob_swap, prob_ins, prob_del, prob_sub]

    return p, edit_prob


def perturb_word(edit_prob, indices, word, x):

    def substitute(word):
        l = list(word)
        if not l[indices[j]] in edit_prob[3]:
            l[indices[j]] = random.choice(string.ascii_letters).lower().replace(l[indices[j]], "")
        else:
            l[indices[j]] = np.random.choice(list(edit_prob[3][l[indices[j]]].keys()-l[indices[j]]))
        return "".join(l)

    for j in range(x):
        r = np.random.randint(4)

        # swap if you have to do only one edit
        if r == 0 and x == 1:
            # if the letter to switch is the last one, switch with the previous one
            if len(indices) <= j + 1:
                word = word[0:indices[j] - 1] + word[indices[j]] + word[indices[j] - 1] + word[indices[j] + 1:]
            else:
                word = word[0:indices[j]] + word[indices[j] + 1] + word[indices[j]] + word[indices[j] + 2:]

        # insert a letter in a random position (after idx)
        elif r == 1:
            if not word[indices[j]] in edit_prob[1]:
                new_letter = random.choice(string.ascii_letters).lower().replace(word[indices[j]], "")
            else:
                new_letter = np.random.choice(list(edit_prob[1][word[indices[j]]].keys()-word[indices[j]]))
            word = word[0:indices[j]] + new_letter + word[indices[j] + 1:]

        # delete a letter
        elif r == 2:
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

    print("\n Starting pertub {} …".format(perturbed.name))
    start = time.time()

    typo_counter = 0
    word_counter = 0
    perturbed_word = 0

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

            word = perturb_word(edit_prob, indices, word, x)

            line_words[i] = word

            if x != 0:
                perturbed_word += 1
            typo_counter += x
            word_counter += 1

        line = " ".join(line_words)
        perturbed.write(line + '\n')

    end = time.time()
    perturb_time = end - start

    perturbed.close()

    print("Endend pertubation in {:6.2f} seconds \n".format(perturb_time))

    typo_per_word = typo_counter / word_counter
    perturbed_word_percentace = perturbed_word / word_counter

    m = {'file': [perturbed.name], 'typo_percentage': [typo_per_word], 'perturbed_word_percentace': [perturbed_word_percentace], 'total_typo': [typo_counter], 'total_word': [word_counter], 'perturbed_word': perturbed_word}
    meta = pd.DataFrame(m)
    meta.to_csv(perturbed.name.replace('.txt', '-meta.csv'), sep=',', index=False)


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


def create_typo_dataset(typo_ds):
    typo_writer = csv.writer(typo_ds)

    # Create a model for the test set
    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    with open("../data/word_freq/lotr_language_model.txt", "r") as myfile:
        reader = csv.reader(myfile)
        words_ds = [row for row in reader]

    typo_dict = defaultdict()

    _, edit_prob = extract_model_edit_probabilities(hmm)

    print("\n Starting pertub {} …".format(typo_ds.name))
    start = time.time()

    typo_counter = 0
    word_counter = 0

    for word in words_ds:
        word = word[0]
        orig_word = word
        typos = list()

        for _ in range(5):
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

            word = perturb_word(edit_prob, indices, word, x)

            typos.append(word)

            typo_counter += x
            word_counter += 1

        typo_dict[orig_word] = typos

    end = time.time()
    perturb_time = end - start

    for key, value in typo_dict.items():
        for v in value:
            typo_writer.writerow([v, key])

    print("Endend pertubation in {:6.2f} seconds \n".format(perturb_time))

    typo_per_word = typo_counter / word_counter

    m = {'file': [typo_ds.name], 'typo_percentage': [typo_per_word], 'total_typo': [typo_counter], 'total_word': [word_counter]}
    meta = pd.DataFrame(m)
    meta.to_csv(typo_ds.name.replace('.csv', '-meta.csv'), sep=',', index=False)

    typo_ds.close()
