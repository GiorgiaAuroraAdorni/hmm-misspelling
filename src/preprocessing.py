#!/usr/bin/python3.7
import util
import pandas as pd
import json
import csv
import os

directory = "../data/typo/"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        clean_dataset(directory, filename)

clean_directory = directory + "clean/"

combined_csv = pd.concat([pd.read_csv(clean_directory + f,
                                      names=['mispelled_word', 'correct_word'])
                          for f in os.listdir(clean_directory) if f.endswith(".csv")])

# Split the typo dataset into train and test
train, test = split_dataset(combined_csv)

train.to_csv(directory + "clean/train.csv", sep=',', header=None, index=False)
test.to_csv(directory + "clean/test.csv", sep=',', header=None, index=False)

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

perturbed1 = open("../data/texts/perturbated/lotr_clean_perturbed-10%.txt", "w")
perturbed2 = open("../data/texts/perturbated/lotr_clean_perturbed-15%.txt", "w")
perturbed3 = open("../data/texts/perturbated/lotr_clean_perturbed-20%.txt", "w")

perturb_file(perturbed1, 0.10)
perturb_file(perturbed2, 0.20)
perturb_file(perturbed3, 0.30)

create_model_language()

# Create a new perturbed typo dataset (accordingly the new language model)
typo_ds = open("../data/typo/lotr_typo.csv", mode="w")
typo_writer = csv.writer(typo_ds)

create_typo_dataset(typo_writer)

typo_ds.close()

# Split the typo dataset in train and test
lotr_train, lotr_test = split_dataset(pd.read_csv("../data/typo/lotr_typo.csv"))

lotr_train.to_csv(directory + "clean/lotr_train.csv", sep=',', header=None, index=False)
lotr_test.to_csv(directory + "clean/lotr_test.csv", sep=',', header=None, index=False)
