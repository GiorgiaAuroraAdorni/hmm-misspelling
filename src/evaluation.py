from hmm import HMM
import pandas as pd
import numpy as np
import pprint
import time
import csv
import os

pp = pprint.PrettyPrinter(indent=4)


def prediction_hmm_candidate_test():
    print("### HMM Candidates - Evaluation")

    hmm = HMM(1, max_edits=2, max_states=5)

    print("\n Starting training…")
    start = time.time()

    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/LordOfTheRingsBook_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    end = time.time()
    train_time = end - start

    print("Endend training in {:4.2f} seconds".format(train_time))
    print("\n Starting testing…")
    start = time.time()

    real = []
    perturbed = []
    observed = [[], [], [], [], []]

    with open("../data/typo/clean/test.csv", "r") as f:
        reader = csv.reader(f)
        obs = [row for row in reader]

        iterator = 0
        for el in obs:
            if iterator % 100 == 0:
                print(iterator)
            iterator += 1

            real.append(el[1])
            perturbed.append(el[0])

            candidates = hmm.candidates(el[0])

            for idx in range(5):
                if len(candidates) < idx + 1:
                    observed[idx].append("")
                else:
                    observed[idx].append(candidates[idx][0])

    end = time.time()
    test_time = end - start
    print("Endend testing in {:6.2f} seconds \n".format(test_time))

    # save prediction to csv
    d = {'real':            real,
         'perturbed':       real,
         'first_observed':  observed[0],
         'second_observed': observed[1],
         'third_observed':  observed[2],
         'fourth_observed': observed[3],
         'fifth_observed':  observed[4]}
    prediction = pd.DataFrame(d)

    if not os.path.exists("../results"):
        os.makedirs("../results")

    prediction.to_csv("../results/typo_prediction.csv", sep=',', index=False)

    m = {'obervation': [iterator], 'train_time': [train_time], 'test_time': [test_time]}
    meta = pd.DataFrame(m)
    meta.to_csv("../results/meta_typo_prediction.csv", sep=',', index=False)


def evaluation_hmm_candidate_test():
    predictions = pd.read_csv("../results/typo_prediction.csv")
    meta = pd.read_csv("../results/meta_typo_prediction.csv")

    print("\n Starting evaluation…")
    start = time.time()

    predictions['count_first'] = np.where((predictions['real'] == predictions['first_observed']), True, False)
    predictions['count_third'] = np.where((predictions['real'] == predictions['first_observed']) |
                                          (predictions['real'] == predictions['second_observed']) |
                                          (predictions['real'] == predictions['third_observed']), True, False)
    predictions['count_fifth'] = np.where((predictions['real'] == predictions['first_observed']) |
                                          (predictions['real'] == predictions['second_observed']) |
                                          (predictions['real'] == predictions['third_observed']) |
                                          (predictions['real'] == predictions['fourth_observed']) |
                                          (predictions['real'] == predictions['fifth_observed']), True, False)

    frequencies1 = predictions['count_first'].value_counts(True)
    frequencies3 = predictions['count_third'].value_counts(True)
    frequencies5 = predictions['count_fifth'].value_counts(True)

    accuracy_top1 = frequencies1[True]
    accuracy_top3 = frequencies3[True]
    accuracy_top5 = frequencies5[True]

    end = time.time()
    eval_time = end - start

    print("Ended evaluation in {:6.2f} seconds \n".format(eval_time))

    meta['eval_time'] = eval_time
    meta['accuracy_top_1'] = accuracy_top1 * 100
    meta['accuracy_top_3'] = accuracy_top3 * 100
    meta['accuracy_top_5'] = accuracy_top5 * 100

    print("Accuracy_top_1: {:4.2f} %".format(accuracy_top1 * 100))
    print("Accuracy_top_3: {:4.2f} %".format(accuracy_top3 * 100))
    print("Accuracy_top_5: {:4.2f} %".format(accuracy_top5 * 100))

    meta = meta.round(2)
    meta.to_csv("../results/meta_typo_prediction.csv", sep=',', index=False)


def prediction_hmm_sequence_test():
    print("### HMM Sequence Prediction - Evaluation")

    # Cleaning dataset
    with open("../data/texts/big_clean.txt", "r") as f:
        real = f.readlines()
        real = [r.replace("\n", "") for r in real]

    print("\n Start training…")

    start = time.time()

    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/LordOfTheRingsBook_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    end = time.time()
    train_time = end - start

    print("Endend training in {:4.2f} seconds".format(train_time))

    print("\n Start testing…")
    start = time.time()

    observed = []

    with open("../data/texts/perturbated/LordOfTheRingsBook_clean_perturbed-10%.txt", "r") as f:
        perturbated = f.readlines()
        perturbated = [p.replace("\n", "") for p in perturbated]

        iterator = 0
        for sentence in perturbated:
            if sentence == '':
                continue
            if iterator % 20 == 0:
                pp.pprint(iterator)
            if iterator > 50:
                break

            iterator += 1
            corrected = hmm.predict_sequence(sentence)
            observed.append(corrected)

    end = time.time()
    test_time = end - start
    print("Endend testing in {:6.2f} seconds \n".format(test_time))

    # save prediction to csv
    d = {'target': real[:51], 'perturbated': perturbated[:51], 'observed': observed}
    prediction = pd.DataFrame(d)

    if not os.path.exists("../results"):
        os.makedirs("../results")

    prediction.to_csv("../results/sentence_prediction-50-10%.csv", sep=',', index=False)

    m = {'obervation': [iterator], 'train_time': [train_time], 'test_time': [test_time]}
    meta = pd.DataFrame(m)
    meta.to_csv("../results/meta_sentence_prediction-50-10%.csv", sep=',', index=False)


def evaluation_hmm_sequence_test():
    predictions = pd.read_csv("../results/sentence_prediction-50-10%.csv")
    meta = pd.read_csv("../results/meta_sentence_prediction-50-10%.csv")

    print("\n Starting evaluation…")
    start = time.time()
    predictions['exact_match'] = np.where(predictions['target'] == predictions['observed'], True, False)

    exact_match_frequencies = predictions['exact_match'].value_counts(True)

    if True not in exact_match_frequencies.keys():
        exact_match_accuracy = 0
    else:
        exact_match_accuracy = exact_match_frequencies[True]

    for index, row in predictions.iterrows():
        target = row["target"].split()
        prediction = row["observed"].split()
        noisy = row["perturbated"].split()

        total = len(target)

        perturbated = 0
        not_perturbated = 0

        correct_prediction = 0
        not_correct_prediction = 0

        correct_perturbated = 0
        not_correct_perturbated = 0

        correct_not_perturbated = 0
        not_correct_not_perturbated = 0

        for i, word in enumerate(target):
            is_perturbated = (target[i] != noisy[i])
            is_correct = (target[i] == prediction[i])

            if is_perturbated:
                perturbated += 1
            else:
                not_perturbated += 1

            if is_correct:
                correct_prediction += 1
            else:
                not_correct_prediction += 1

            if is_perturbated and is_correct:
                correct_perturbated += 1
            elif is_perturbated and not is_correct:
                not_correct_perturbated += 1
            elif not is_perturbated and is_correct:
                correct_not_perturbated += 1
            else:
                not_correct_not_perturbated += 1

        if perturbated == 0:
            predictions.loc[index, 'not_correct PREV correct'] = np.nan
            predictions.loc[index, 'not_correct PREV not_correct'] = np.nan
            predictions.loc[index, 'initial_error'] = np.nan
        else:
            predictions.loc[index, 'not_correct PREV correct'] = correct_perturbated / perturbated
            predictions.loc[index, 'not_correct PREV not_correct'] = not_correct_perturbated / perturbated
            predictions.loc[index, 'initial_error'] = perturbated / total

        if not_perturbated == 0:
            predictions.loc[index, 'correct PREV correct'] = np.nan
            predictions.loc[index, 'correct PREV not_correct'] = np.nan
        else:
            predictions.loc[index, 'correct PREV correct'] = correct_not_perturbated / not_perturbated
            predictions.loc[index, 'correct PREV not_correct'] = not_correct_not_perturbated / not_perturbated

        predictions.loc[index, 'error_rate'] = not_correct_prediction / total
        predictions.loc[index, 'correct'] = perturbated / total
        predictions.loc[index, 'not_correct'] = not_perturbated / total
        predictions.loc[index, 'accuracy'] = correct_prediction / total

        if correct_not_perturbated + correct_perturbated == 0:
            predictions.loc[index, 'precision'] = np.nan
        else:
            predictions.loc[index, 'precision'] = correct_not_perturbated / (correct_not_perturbated + correct_perturbated)

        if correct_not_perturbated + not_correct_not_perturbated == 0:
            predictions.loc[index, 'recall'] = np.nan
        else:
            predictions.loc[index, 'recall'] = correct_not_perturbated / (correct_not_perturbated + not_correct_not_perturbated)  # same of sensitivity

        if not_correct_perturbated + correct_not_perturbated == 0:
            predictions.loc[index, 'specificity'] = np.nan
        else:
            predictions.loc[index, 'specificity'] = not_correct_perturbated / (not_correct_perturbated + correct_not_perturbated)

    word_accuracy = np.mean(predictions['accuracy'])
    word_precision = np.mean(predictions['precision'])
    word_recall = np.mean(predictions['recall'])
    word_specificity = np.mean(predictions['specificity'])

    word_correct = np.nanmean(predictions['correct'])
    word_not_correct = np.nanmean(predictions['not_correct'])
    word_not_correct_PREV_correct = np.nanmean(predictions['not_correct PREV correct'])
    word_not_correct_PREV_not_correct = np.nanmean(predictions['not_correct PREV not_correct'])
    word_correct_PREV_correct = np.nanmean(predictions['correct PREV correct'])
    word_correct_PREV_not_correct = np.nanmean(predictions['correct PREV not_correct'])
    initial_error = np.nanmean(predictions['initial_error'])
    error_rate = np.nanmean(predictions['error_rate'])

    end = time.time()
    eval_time = end - start

    print("Ended evaluation in {:6.2f} seconds \n".format(eval_time))

    print("Exact match accuracy: {:4.2f} %".format(exact_match_accuracy * 100))
    print("Word accuracy: {:4.2f} %".format(word_accuracy * 100))
    print("Word precision: {:4.2f} %".format(word_precision * 100))
    print("Word recall: {:4.2f} %".format(word_recall * 100))
    print("Word specificity: {:4.2f} %".format(word_specificity * 100))

    predictions.to_csv("../results/sentence_evaluation-50-10%.csv", sep=',', index=False)

    meta['eval_time'] = eval_time
    meta['accuracy_top_1'] = word_accuracy * 100
    meta['exact_match'] = exact_match_accuracy * 100
    meta['initial_error'] = initial_error * 100
    meta['error_rate'] = error_rate * 100
    meta['precision'] = word_precision * 100
    meta['recall'] = word_recall * 100
    meta['specificity'] = word_specificity * 100

    meta['correct'] = word_correct * 100
    meta['not_correct'] = word_not_correct * 100
    meta['not_correct_PREV_correct'] = word_not_correct_PREV_correct * 100
    meta['not_correct_PREV_not_correct'] = word_not_correct_PREV_not_correct * 100
    meta['correct_PREV_correct'] = word_correct_PREV_correct * 100
    meta['correct_PREV_not_correct'] = word_correct_PREV_not_correct * 100

    meta = meta.round(2)
    meta.to_csv("../results/meta_sentence_prediction-50-10%.csv", sep=',', index=False)


prediction_hmm_candidate_test()
evaluation_hmm_candidate_test()

# prediction_hmm_sequence_test()
# evaluation_hmm_sequence_test()
