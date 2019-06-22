#!/usr/bin/python3.7
from util import clean_dataset, split_dataset, perturb
import pandas as pd
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

# Clean the dataset big
with open("../data/texts/big.txt", "r") as f:
    real = f.read().split(".")
    real = [r.strip().lower().replace("'", "") for r in real]
    real = [re.sub(r"[^a-zA-Z0-9]+", ' ', r) for r in real]

filename = "../data/texts/big_clean.txt"
# Create a perturbated dataset
if not os.path.exists("../data/texts/perturbated/"):
    os.makedirs("../data/texts/perturbated/")

perturbed1 = open("../data/texts/perturbated/big_perturbed-10%.txt", "w")
perturbed2 = open("../data/texts/perturbated/big_perturbed-15%.txt", "w")
perturbed3 = open("../data/texts/perturbated/big_perturbed-20%.txt", "w")

perturb(perturbed1, 0.10)
perturb(perturbed2, 0.20)
perturb(perturbed3, 0.30)
