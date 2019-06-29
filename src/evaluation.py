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

    case_1_T = 0
    case_1a_T = 0
    case_1b_T = 0
    case_2_T = 0
    case_3_T = 0
    case_4_T = 0

    for index, row in predictions.iterrows():
        real = row["target"].split()
        prediction = row["observed"].split()
        noisy = row["perturbed"].split()

        total = len(real)

        perturbed = 0
        not_perturbed = 0

        correct_prediction = 0
        not_correct_prediction = 0

        case_1 = 0   # Perturbed word not correctly provided
        case_1a = 0  # The perturbed word was not the subject of attempted correction by the model
        case_1b = 0  # The perturbed word has been the subject of attempted correction by the model but without success
        case_2 = 0   # Perturbed word correctly provided
        case_3 = 0   # Unperturbed word not correctly provided
        case_4 = 0   # Unperturbed word correctly provided

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
                case_2 += 1
            elif is_perturbed and not is_correct:
                case_1 += 1
                if noisy[i] != prediction[i]:
                    case_1b += 1
                else:
                    case_1a += 1
            elif not is_perturbed and is_correct:
                case_4 += 1
            else:
                case_3 += 1

        if perturbed == 0:
            predictions.loc[index, 'case_1'] = np.nan
            predictions.loc[index, 'case_1a'] = np.nan
            predictions.loc[index, 'case_1b'] = np.nan
            predictions.loc[index, 'case_2'] = np.nan
        else:
            predictions.loc[index, 'case_1'] = case_1 / perturbed
            predictions.loc[index, 'case_1a'] = case_1a / perturbed
            predictions.loc[index, 'case_1b'] = case_1b / perturbed
            predictions.loc[index, 'case_2'] = case_2 / perturbed

        if not_perturbed == 0:
            predictions.loc[index, 'case_3'] = np.nan
            predictions.loc[index, 'case_4'] = np.nan
        else:
            predictions.loc[index, 'case_3'] = case_3 / not_perturbed
            predictions.loc[index, 'case_4'] = case_4 / not_perturbed

        predictions.loc[index, 'detection-accuracy'] = (case_1b + case_2 + case_4) / total
        predictions.loc[index, 'correction-accuracy'] = (case_2 + case_4) / total

        # Recall
        if case_1 + case_2 == 0:
            predictions.loc[index, 'detection-recall'] = np.nan
        else:
            predictions.loc[index, 'detection-recall'] = (case_1b + case_2) / (case_1 + case_2)

        if case_1 + case_2 == 0:
            predictions.loc[index, 'correction-recall'] = np.nan
        else:
            predictions.loc[index, 'correction-recall'] = case_2 / (case_1 + case_2)

        # Precision
        if case_1b + case_2 + case_3 == 0:
            predictions.loc[index, 'detection-precision'] = np.nan
        else:
            predictions.loc[index, 'detection-precision'] = (case_1b + case_2) / (case_1b + case_2 + case_3)

        if case_2 + case_3 == 0:
            predictions.loc[index, 'correction-precision'] = np.nan
        else:
            predictions.loc[index, 'correction-precision'] = case_2 / (case_2 + case_3)

        # Specificity
        if case_3 + case_4 == 0:
            predictions.loc[index, 'specificity'] = np.nan
        else:
            predictions.loc[index, 'specificity'] = case_4 / (case_3 + case_4)

        case_1_T += case_1
        case_1a_T += case_1a
        case_1b_T += case_1b
        case_2_T += case_2
        case_3_T += case_3
        case_4_T += case_4

    detection_accuracy = np.mean(predictions['detection-accuracy'])
    detection_recall = np.mean(predictions['detection-recall'])
    detection_precision = np.mean(predictions['detection-precision'])

    correction_accuracy = np.mean(predictions['correction-accuracy'])
    correction_recall = np.mean(predictions['correction-recall'])
    correction_precision = np.mean(predictions['correction-precision'])

    specificity = np.mean(predictions['specificity'])

    end = time.time()
    eval_time = end - start

    print("Ended evaluation in {:6.2f} seconds".format(eval_time))

    print("Exact match accuracy: {:4.2f} %".format(exact_match_accuracy * 100))
    print("Word detection-accuracy: {:4.2f} %".format(detection_accuracy * 100))
    print("Word correction-accuracy: {:4.2f} %".format(correction_accuracy * 100))
    print("Word detection-recall: {:4.2f} %".format(detection_recall * 100))
    print("Word correction-recall: {:4.2f} %".format(correction_recall * 100))
    print("Word detection-precision: {:4.2f} %".format(detection_precision * 100))
    print("Word correction-precision: {:4.2f} %".format(correction_precision * 100))
    print("Word specificity: {:4.2f} %".format(specificity * 100))

    predictions.to_csv(prediction_sentence_filename, sep=',', index=False)

    meta['eval_time'] = eval_time
    meta['perturbed_ds'] = perturbed_ds

    meta['case_1'] = case_1_T
    meta['case_1a'] = case_1a_T
    meta['case_1b'] = case_1b_T
    meta['case_2'] = case_2_T
    meta['case_3'] = case_3_T
    meta['case_4'] = case_4_T

    meta['exact_match'] = exact_match_accuracy * 100
    meta['detection_accuracy'] = detection_accuracy * 100
    meta['correction_accuracy'] = correction_accuracy * 100
    meta['detection_recall'] = detection_recall * 100
    meta['correction_recall'] = correction_recall * 100
    meta['detection_precision'] = detection_precision * 100
    meta['correction_precision'] = correction_precision * 100
    meta['specificity'] = specificity * 100

    meta = meta.round(2)
    meta.to_csv(meta_sentence_filename, sep=',', index=False)
