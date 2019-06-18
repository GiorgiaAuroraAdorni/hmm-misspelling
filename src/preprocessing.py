from util import clean_dataset, split_dataset
import pandas as pd
import os

directory = "../data/typo/"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        clean_dataset(directory + filename)

combined_csv = pd.concat([pd.read_csv(directory + f, names=['mispelled_word', 'correct_word'])
                          for f in os.listdir(directory) if f.endswith(".csv")])

# lowercase
df = combined_csv.apply(lambda x: x.astype(str).str.lower())

# remove number and symbols
# regex = re.compile('[@_!-#$%^&*()<>?/\|}{~:0-9]')
# msk = df.mispelled_word.str.contains("@_!-#$%^&*()<>?/\|}{~:0-9")

#split dataset
train, test = split_dataset(df)

train.to_csv(directory + "train.csv", sep=',', header=None, index=False)
test.to_csv(directory + "test.csv", sep=',', header=None, index=False)
