#!/usr/bin/python3.7
import util
import pandas as pd
import json
import csv
import os

directory = "../data/typo/"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        util.clean_dataset(directory, filename)

clean_directory = directory + "clean/"

combined_csv = pd.concat([pd.read_csv(clean_directory + f,
                                      names=['mispelled_word', 'correct_word'])
                          for f in os.listdir(clean_directory) if f.endswith(".csv")])

# Split the typo dataset into train and test
train, test = util.split_dataset(combined_csv)

train.to_csv(directory + "clean/big_train.csv", sep=',', header=None, index=False)
test.to_csv(directory + "clean/big_test.csv", sep=',', header=None, index=False)

# Read big.txt and extract a cleaned dataset
big_file = "../data/texts/big.txt"
lotr_file = "../data/texts/LordOfTheRingsBook.json"

with open(big_file, "r") as f:
    big = f.read()
    big_cleaned = util.clean_sentences_dataset(big)

with open(big_file.replace(".txt", "_clean.txt"), mode="w") as outfile:
    for r in big_cleaned:
        outfile.write("%s\n" % r)

# Read LordOfTheRingsBook.json and extract a cleaned dataset
with open(lotr_file) as json_data:
    d = json.load(json_data)
    lotr = [chapter["ChapterData"] for chapter in d]
    lotr = "\n".join(lotr)

    lotr_cleaned = util.clean_sentences_dataset(lotr)

with open("../data/texts/lotr_clean.txt", mode="w") as outfile:
    for r in lotr_cleaned:
        outfile.write("%s\n" % r)

# Create a perturbated dataset
if not os.path.exists("../data/texts/perturbated/"):
    os.makedirs("../data/texts/perturbated/")

# Create perturbed datasets for big
perturbed0a = open("../data/texts/perturbated/big_clean_perturbed-5%.txt", "w")
perturbed1a = open("../data/texts/perturbated/big_clean_perturbed-10%.txt", "w")
perturbed2a = open("../data/texts/perturbated/big_clean_perturbed-15%.txt", "w")
perturbed3a = open("../data/texts/perturbated/big_clean_perturbed-20%.txt", "w")

util.perturb_file(perturbed0a, 0.05)
util.perturb_file(perturbed1a, 0.10)
util.perturb_file(perturbed2a, 0.15)
util.perturb_file(perturbed3a, 0.20)

# Create perturbed datasets for lotr
perturbed0b = open("../data/texts/perturbated/lotr_clean_perturbed-5%.txt", "w")
perturbed1b = open("../data/texts/perturbated/lotr_clean_perturbed-10%.txt", "w")
perturbed2b = open("../data/texts/perturbated/lotr_clean_perturbed-15%.txt", "w")
perturbed3b = open("../data/texts/perturbated/lotr_clean_perturbed-20%.txt", "w")

util.perturb_file(perturbed0b, 0.05)
util.perturb_file(perturbed1b, 0.10)
util.perturb_file(perturbed2b, 0.15)
util.perturb_file(perturbed3b, 0.20)

# Create lotr model language
util.create_model_language()

# Clean big model language
frequency_alpha_gcide = pd.read_csv("../data/word_freq/frequency-alpha-gcide.txt", sep="\t", header=None)

big_model_lang = pd.DataFrame()
big_model_lang['word'] = [x.split()[1] for x in frequency_alpha_gcide[0].tolist()]
big_model_lang['freq'] = frequency_alpha_gcide[2]
big_model_lang.to_csv("../data/word_freq/big_language_model.txt", sep=',', index=False, header=None)

# Create a new perturbed typo dataset (accordingly the new language model)
typo_ds = open("../data/typo/lotr_typo.csv", mode="w")
util.create_typo_dataset(typo_ds)

# Split the typo dataset in train and test
lotr_train, lotr_test = util.split_dataset(pd.read_csv("../data/typo/lotr_typo.csv"))

lotr_train.to_csv(directory + "clean/lotr_train.csv", sep=',', header=None, index=False)
lotr_test.to_csv(directory + "clean/lotr_test.csv", sep=',', header=None, index=False)
