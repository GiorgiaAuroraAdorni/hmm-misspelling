from hmm import HMM
import pandas as pd
import numpy as np
import random
import pprint
import time
import csv
import os


def prediction_hmm_candidate_test():
    print("### HMM Candidates - Evaluation")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=3)

    print("\n Start trainining…")
    start = time.time()

    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/lotr_intro.txt",
              typo_ds="../data/typo/new/train.csv")

    end = time.time()
    train_time = end - start

    print("Endend training in {:4.2f} seconds".format(train_time))
    print("\n Start testing…")
    start = time.time()

    real = []
    observed = []

    with open("../data/typo/new/test.csv", "r") as f:
        reader = csv.reader(f)
        obs = [row for row in reader]

        iterator = 0
        for el in obs:
            if iterator % 100 == 0:
                print(iterator)
            iterator += 1

            real.append(el[1])
            if len(hmm.candidates(el[0])) > 0:
                observed.append(hmm.candidates(el[0])[0][0])
            else:
                observed.append("") # if no word is bredicted by the model
                print(el, "No words predicted", "\n")

    end = time.time()
    test_time = end - start
    print("Endend testing in {:6.2f} seconds \n".format(test_time))

    # save prediction to csv
    d = {'real': real, 'observed': observed}
    prediction = pd.DataFrame(d)

    if not os.path.exists("../results"):
        os.makedirs("../results")

    prediction.to_csv("../results/typo_evaluation.csv", sep=',', index=False)


def evaluation_hmm_candidate_test():
    predictions = pd.read_csv("../results/typo_evaluation.csv")

    print("\n Starting evaluation…")
    start = time.time()
    predictions['count'] = np.where(predictions['real'] == predictions['observed'], True, False)

    frequencies = predictions['count'].value_counts(True)
    accuracy = frequencies[True]

    end = time.time()
    eval_time = end - start

    print("Ended evaluation in {:6.2f} seconds \n".format(eval_time))

    print("Accuracy: {:4.2f} %".format(accuracy*100))


def perturb(hmm):
    cleaned = open("../data/texts/lotr_intro.txt", "r")

    if not os.path.exists("../data/texts/perturbates/"):
        os.makedirs("../data/texts/perturbates/")

    perturbed = open("../data/texts/perturbates/lotr_intro_perturbed.txt", "w")

    for line in cleaned:
        pos = 0
        while pos < len(line):
            if line[pos].isalpha():
                char_select = line[pos]
                c = random.random()
                if c <= 0.1:
                    # Given a char selected randomly from the line,
                    # replace it with a new char chosen randomly from the given neighbours in the hmm error model
                    cran = random.choice(list(hmm.error_model["sub"][char_select.lower()].keys()))
                    initial = line[0:pos]
                    final = line[pos + 1:]
                    line = initial + cran + final
                pos += 1
            else:
                pos += 1
            if pos == len(line):
                perturbed.write(line)

    cleaned.close()
    perturbed.close()


# prediction_hmm_candidate_test()
# evaluation_hmm_candidate_test()
