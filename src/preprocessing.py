#!/usr/bin/python3.7
from util import clean_dataset, split_dataset, perturb
import pandas as pd
import os
import re

directory = "../data/typo/"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        clean_dataset(directory, filename)

combined_csv = pd.concat([pd.read_csv(directory + f, names=['mispelled_word', 'correct_word'])
                          for f in os.listdir(directory) if f.endswith(".csv")])

# Split the typo dataset into train and test
train, test = split_dataset(combined_csv)

train.to_csv(directory + "new/train.csv", sep=',', header=None, index=False)
test.to_csv(directory + "new/test.csv", sep=',', header=None, index=False)

# Clean the dataset big
with open("../data/texts/big.txt", "r") as f:
    real = f.read().split(".")
    real = [r.strip().lower().replace("'", "") for r in real]
    real = [re.sub(r"[^a-zA-Z0-9]+", ' ', r) for r in real]

filename = "../data/texts/big_clean.txt"

with open(filename, mode="w") as outfile:
    for r in real:
        outfile.write("%s\n" % r)

# Create a perturbated dataset
perturb()
