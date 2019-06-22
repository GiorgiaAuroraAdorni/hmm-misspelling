#!/usr/bin/python3.7
from util import clean_dataset, split_dataset, perturb
import pandas as pd
import json
import os
import re

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

# Read LordOfTheRingsBook.json and extract a cleaned dataset
with open('../data/texts/LordOfTheRingsBook.json') as json_data:
    d = json.load(json_data)
    real = [chapter["ChapterData"] for chapter in d]
    real = "\n".join(real)
    real = real.replace("Mr.", "Mr").replace("Mrs.", "Mrs")
    real = real.split(".")
    real = [r.strip().lower().replace("'", '') for r in real]
    real = [re.sub(r"[^a-zA-Z0-9]+", ' ', r) for r in real]

with open("../data/texts/LordOfTheRingsBook_clean.txt", mode="w") as outfile:
    for r in real:
        outfile.write("%s\n" % r)

# Create a perturbated dataset
if not os.path.exists("../data/texts/perturbated/"):
    os.makedirs("../data/texts/perturbated/")

perturbed1 = open("../data/texts/perturbated/LordOfTheRingsBook_clean_perturbed-10%.txt", "w")
perturbed2 = open("../data/texts/perturbated/LordOfTheRingsBook_clean_perturbed-15%.txt", "w")
perturbed3 = open("../data/texts/perturbated/LordOfTheRingsBook_clean_perturbed-20%.txt", "w")

perturb(perturbed1, 0.10)
perturb(perturbed2, 0.20)
perturb(perturbed3, 0.30)
