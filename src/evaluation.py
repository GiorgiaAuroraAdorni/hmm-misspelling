import pandas as pd
import numpy as np
import pprint
import time
import csv


def prediction_hmm_candidate_test(typo_ds_test, hmm, prediction_typo_filename, meta_typo_filename):
    print("### HMM Candidates - Evaluation")
    print("Starting testing…")
    start = time.time()

    real = []
    perturbed = []
    observed = [[], [], [], [], []]

    with open(typo_ds_test, "r") as f:
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
    print("Ended testing in {:6.2f} seconds".format(test_time))

    # save prediction to csv
    d = {'real': real,
         'perturbed': perturbed,
         'first_observed': observed[0],
         'second_observed': observed[1],
         'third_observed': observed[2],
         'fourth_observed': observed[3],
         'fifth_observed': observed[4]}
    prediction = pd.DataFrame(d)

    prediction.to_csv(prediction_typo_filename, sep=',', index=False)

    m = {'observation': [iterator], 'test_time': [test_time]}
    meta = pd.DataFrame(m)
    meta.to_csv(meta_typo_filename, sep=',', index=False)


def evaluation_hmm_candidate_test(prediction_typo_filename, meta_typo_filename):
    predictions = pd.read_csv(prediction_typo_filename)
    meta = pd.read_csv(meta_typo_filename)

    print("Starting evaluation…")
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

    print("Ended evaluation in {:6.2f} seconds".format(eval_time))

    meta['eval_time'] = eval_time
    meta['accuracy_top_1'] = accuracy_top1 * 100
    meta['accuracy_top_3'] = accuracy_top3 * 100
    meta['accuracy_top_5'] = accuracy_top5 * 100

    print("Accuracy_top_1: {:4.2f} %".format(accuracy_top1 * 100))
    print("Accuracy_top_3: {:4.2f} %".format(accuracy_top3 * 100))
    print("Accuracy_top_5: {:4.2f} %".format(accuracy_top5 * 100))

    meta = meta.round(2)
    meta.to_csv(meta_typo_filename, sep=',', index=False)


def prediction_hmm_sequence_test(sentences_ds, perturbed_ds, hmm, prediction_sentence_filename, meta_sentence_filename):
    print("### HMM Sequence Prediction - Evaluation")

    # Cleaning dataset
    with open(sentences_ds, "r") as f:
        real = f.readlines()
        real = [r.replace("\n", "") for r in real]

    print("Start testing…")
    start = time.time()

    observed = []

    with open(perturbed_ds, "r") as f:
        perturbed = f.readlines()
        perturbed = [p.replace("\n", "") for p in perturbed]

        iterator = 0
        for sentence in perturbed:
            if sentence == '':
                continue
            if iterator % 20 == 0:
                print(iterator)
            if iterator > 5000:
                break

            iterator += 1
            corrected = hmm.predict_sequence(sentence)
            observed.append(corrected)

    end = time.time()
    test_time = end - start
    print("Ended testing in {:6.2f} seconds".format(test_time))

    # save prediction to csv
    d = {'target': real[:iterator], 'perturbed': perturbed[:iterator], 'observed': observed}

    prediction = pd.DataFrame(d)

    prediction.to_csv(prediction_sentence_filename, sep=',', index=False)

    m = {'observation': [iterator], 'test_time': [test_time]}
    meta = pd.DataFrame(m)
    meta.to_csv(meta_sentence_filename, sep=',', index=False)


def evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds):
    predictions = pd.read_csv(prediction_sentence_filename)
    meta = pd.read_csv(meta_sentence_filename)

    print("Starting evaluation…")
    start = time.time()
    predictions['exact_match'] = np.where(predictions['target'] == predictions['observed'], True, False)

    exact_match_frequencies = predictions['exact_match'].value_counts(True)

    if True not in exact_match_frequencies.keys():
        exact_match_accuracy = 0
    else:
        exact_match_accuracy = exact_match_frequencies[True]

    for index, row in predictions.iterrows():
        real = row["target"].split()
        prediction = row["observed"].split()
        noisy = row["perturbed"].split()

        total = len(real)

        perturbed = 0
        not_perturbed = 0

        correct_prediction = 0
        not_correct_prediction = 0

        correct_perturbed = 0
        not_correct_perturbed = 0

        correct_not_perturbed = 0
        not_correct_not_perturbed = 0

        not_correct_modified_perturbed = 0

        for i, word in enumerate(real):
            is_perturbed = (real[i] != noisy[i])
            is_correct = (real[i] == prediction[i])

            if is_perturbed:
                perturbed += 1
            else:
                not_perturbed += 1

            if is_correct:
                correct_prediction += 1
            else:
                not_correct_prediction += 1

            if is_perturbed and is_correct:
                correct_perturbed += 1
            elif is_perturbed and not is_correct:
                if noisy[i] != prediction[i]:
                    not_correct_modified_perturbed += 1
                else:
                    not_correct_perturbed += 1
            elif not is_perturbed and is_correct:
                correct_not_perturbed += 1
            else:
                not_correct_not_perturbed += 1

        if perturbed == 0:
            predictions.loc[index, 'not_correct PREV correct'] = np.nan
            predictions.loc[index, 'not_correct PREV not_correct'] = np.nan
            predictions.loc[index, 'initial_error'] = np.nan
        else:
            predictions.loc[index, 'not_correct PREV correct'] = correct_perturbed / perturbed
            predictions.loc[index, 'not_correct PREV not_correct'] = not_correct_perturbed / perturbed
            predictions.loc[index, 'initial_error'] = perturbed / total

        if not_perturbed == 0:
            predictions.loc[index, 'correct PREV correct'] = np.nan
            predictions.loc[index, 'correct PREV not_correct'] = np.nan
        else:
            predictions.loc[index, 'correct PREV correct'] = correct_not_perturbed / not_perturbed
            predictions.loc[index, 'correct PREV not_correct'] = not_correct_not_perturbed / not_perturbed

        predictions.loc[index, 'not_correct_modified PREV not_correct'] = not_correct_modified_perturbed
        predictions.loc[index, 'correct'] = perturbed / total
        predictions.loc[index, 'not_correct'] = not_perturbed / total
        predictions.loc[index, 'accuracy'] = correct_prediction / total

        if correct_perturbed + not_correct_not_perturbed + not_correct_modified_perturbed == 0:
            predictions.loc[index, 'precision'] = np.nan
        else:
            predictions.loc[index, 'precision'] = correct_perturbed / (
                    correct_perturbed + not_correct_not_perturbed + not_correct_modified_perturbed)

        if correct_perturbed + not_correct_perturbed == 0:
            predictions.loc[index, 'recall'] = np.nan
        else:
            predictions.loc[index, 'recall'] = correct_perturbed / (correct_perturbed + not_correct_perturbed)

        if predictions.loc[index, 'precision'] + predictions.loc[index, 'recall'] == 0:
            predictions.loc[index, 'F1-score'] = np.nan
        else:
            predictions.loc[index, 'F1-score'] = 2 * (
                    predictions.loc[index, 'precision'] * predictions.loc[index, 'recall'] / (
                    predictions.loc[index, 'precision'] + predictions.loc[index, 'recall']))

    word_accuracy = np.mean(predictions['accuracy'])
    word_precision = np.mean(predictions['precision'])
    word_recall = np.mean(predictions['recall'])
    word_f1_score = np.mean(predictions['F1-score'])

    word_correct = np.nanmean(predictions['correct'])
    word_not_correct = np.nanmean(predictions['not_correct'])
    word_not_correct_PREV_correct = np.nanmean(predictions['not_correct PREV correct'])
    word_not_correct_PREV_not_correct = np.nanmean(predictions['not_correct PREV not_correct'])
    word_correct_PREV_correct = np.nanmean(predictions['correct PREV correct'])
    word_correct_PREV_not_correct = np.nanmean(predictions['correct PREV not_correct'])
    word_not_correct_modified_PREV_not_correct = np.nanmean(predictions['not_correct_modified PREV not_correct'])
    initial_error = np.nanmean(predictions['initial_error'])

    end = time.time()
    eval_time = end - start

    print("Ended evaluation in {:6.2f} seconds".format(eval_time))

    print("Exact match accuracy: {:4.2f} %".format(exact_match_accuracy * 100))
    print("Word accuracy: {:4.2f} %".format(word_accuracy * 100))
    print("Word precision: {:4.2f} %".format(word_precision * 100))
    print("Word recall: {:4.2f} %".format(word_recall * 100))
    print("Word F1 Score: {:4.2f} %".format(word_f1_score * 100))

    predictions.to_csv(prediction_sentence_filename, sep=',', index=False)

    meta['eval_time'] = eval_time
    meta['perturbed_ds'] = perturbed_ds
    meta['accuracy'] = word_accuracy * 100
    meta['exact_match'] = exact_match_accuracy * 100
    meta['initial_error'] = initial_error * 100
    meta['precision'] = word_precision * 100
    meta['recall'] = word_recall * 100
    meta['F1-Score'] = word_f1_score * 100
    meta['not_correct_PREV_correct'] = word_not_correct_PREV_correct * 100
    meta['correct_PREV_not_correct'] = word_correct_PREV_not_correct * 100
    meta['correct'] = word_correct * 100
    meta['not_correct'] = word_not_correct * 100
    meta['not_correct_PREV_not_correct'] = word_not_correct_PREV_not_correct * 100
    meta['not_correct_modified_PREV_not_correct'] = word_not_correct_modified_PREV_not_correct * 100
    meta['correct_PREV_correct'] = word_correct_PREV_correct * 100

    meta = meta.round(2)
    meta.to_csv(meta_sentence_filename, sep=',', index=False)
