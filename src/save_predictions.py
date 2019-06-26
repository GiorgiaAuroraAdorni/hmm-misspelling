from hmm import HMM
import evaluation as eval
import pandas as pd
import time
import os

#### EXPERIMENT 1 ####
words_ds = "../data/word_freq/big_language_model.txt"
sentences_ds = "../data/texts/big_clean.txt"
typo_ds_train = "../data/typo/clean/big_train.csv"
typo_ds_test = "../data/typo/clean/big_test.csv"
perturbed_ds_5 = "../data/texts/perturbed/big_clean_perturbed-5%.txt"
perturbed_ds_10 = "../data/texts/perturbed/big_clean_perturbed-10%.txt"
perturbed_ds_15 = "../data/texts/perturbed/big_clean_perturbed-15%.txt"
perturbed_ds_20 = "../data/texts/perturbed/big_clean_perturbed-20%.txt"

if not os.path.exists("../results/experiment1"):
    os.makedirs("../results/experiment1")
###
edit_distance = 1

print("Starting training…")
start = time.time()

hmm = HMM(1, max_edits=edit_distance, max_states=5)
hmm.train(words_ds=words_ds,
          sentences_ds=sentences_ds,
          typo_ds=typo_ds_train)

end = time.time()
train_time = end - start

print("Ended training in {:4.2f} seconds".format(train_time))

model = {"max_edits": [hmm.max_edits], "max_states": [hmm.max_states], "order": [hmm.order],
         "state_len": [hmm.state_len], "error_model_p": [hmm.error_model['p']], 'train_time': [train_time],
         'language_ds': words_ds, 'sentence_ds': sentences_ds, 'typo_ds_train': typo_ds_train,
         'typo_ds_test': typo_ds_test, 'edit_distance': edit_distance}
m = pd.DataFrame(model)
m.to_csv("../results/experiment1/1-model.csv", sep=',', index=False)

## Typo:
# On test set
prediction_typo_test_filename = "../results/experiment1/1-typo_prediction-test.csv"
meta_typo_test_filename = "../results/experiment1/1-meta_typo_prediction-test.csv"

eval.prediction_hmm_candidate_test(typo_ds_test, hmm, prediction_typo_test_filename, meta_typo_test_filename)
eval.evaluation_hmm_candidate_test(prediction_typo_test_filename, meta_typo_test_filename)

# On train set
prediction_typo_train_filename = "../results/experiment1/1-typo_prediction-train.csv"
meta_typo_train_filename = "../results/experiment1/1-meta_typo_prediction-train.csv"

eval.prediction_hmm_candidate_test(typo_ds_train, hmm, prediction_typo_train_filename, meta_typo_train_filename)
eval.evaluation_hmm_candidate_test(prediction_typo_train_filename, meta_typo_train_filename)

## Sentence
# Perturbation 5%
prediction_sentence_filename = "../results/experiment1/1-sentence_prediction-5%.csv"
meta_sentence_filename = "../results/experiment1/1-meta_sentence_prediction-5%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_5, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_5)

# Perturbation 10%
prediction_sentence_filename = "../results/experiment1/1-sentence_prediction-10%.csv"
meta_sentence_filename = "../results/experiment1/1-meta_sentence_prediction-10%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_10, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_10)

# Perturbation 15%
prediction_sentence_filename = "../results/experiment1/1-sentence_prediction-15%.csv"
meta_sentence_filename = "../results/experiment1/1-meta_sentence_prediction-15%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_15, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_15)

# Perturbation 20%
prediction_sentence_filename = "../results/experiment1/1-sentence_prediction-20%.csv"
meta_sentence_filename = "../results/experiment1/1-meta_sentence_prediction-20%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_20, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_20)

###
edit_distance = 2
print("Starting training…")
start = time.time()

hmm = HMM(1, max_edits=edit_distance, max_states=5)
hmm.train(words_ds=words_ds,
          sentences_ds=sentences_ds,
          typo_ds=typo_ds_train)

end = time.time()
train_time = end - start

print("Ended training in {:4.2f} seconds".format(train_time))

model = {"max_edits": [hmm.max_edits], "max_states": [hmm.max_states], "order": [hmm.order],
         "state_len": [hmm.state_len], "error_model_p": [hmm.error_model['p']], 'train_time': [train_time],
         'language_ds': words_ds, 'sentence_ds': sentences_ds, 'typo_ds_train': typo_ds_train,
         'typo_ds_test': typo_ds_test, 'edit_distance': edit_distance}
m = pd.DataFrame(model)
m.to_csv("../results/experiment1/2-model.csv", sep=',', index=False)

## Typo:
# On test set
prediction_typo_test_filename = "../results/experiment1/2-typo_prediction-test.csv"
meta_typo_test_filename = "../results/experiment1/2-meta_typo_prediction-test.csv"

eval.prediction_hmm_candidate_test(typo_ds_test, hmm, prediction_typo_test_filename, meta_typo_test_filename)
eval.evaluation_hmm_candidate_test(prediction_typo_test_filename, meta_typo_test_filename)

# On train set
# prediction_typo_train_filename = "../results/experiment1/2-typo_prediction-train.csv"
# meta_typo_train_filename = "../results/experiment1/2-meta_typo_prediction-train.csv"

# eval.prediction_hmm_candidate_test(typo_ds_train, hmm, prediction_typo_train_filename, meta_typo_train_filename)
# eval.evaluation_hmm_candidate_test(prediction_typo_train_filename, meta_typo_train_filename)

## Sentence
# Perturbation 5%
prediction_sentence_filename = "../results/experiment1/2-sentence_prediction-5%.csv"
meta_sentence_filename = "../results/experiment1/2-meta_sentence_prediction-5%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_5, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_5)

# Perturbation 10%
prediction_sentence_filename = "../results/experiment1/2-sentence_prediction-10%.csv"
meta_sentence_filename = "../results/experiment1/2-meta_sentence_prediction-10%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_10, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_10)

# Perturbation 15%
prediction_sentence_filename = "../results/experiment1/2-sentence_prediction-15%.csv"
meta_sentence_filename = "../results/experiment1/2-meta_sentence_prediction-15%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_15, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_15)

# Perturbation 20%
prediction_sentence_filename = "../results/experiment1/2-sentence_prediction-20%.csv"
meta_sentence_filename = "../results/experiment1/2-meta_sentence_prediction-20%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_20, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_20)

#### EXPERIMENT 3 ####
words_ds = "../data/word_freq/lotr_language_model.txt"
sentences_ds = "../data/texts/lotr_clean.txt"
typo_ds_train = "../data/typo/clean/lotr_train.csv"
typo_ds_test = "../data/typo/clean/lotr_test.csv"
perturbed_ds_5 = "../data/texts/perturbed/lotr_clean_perturbed-5%.txt"
perturbed_ds_10 = "../data/texts/perturbed/lotr_clean_perturbed-10%.txt"
perturbed_ds_15 = "../data/texts/perturbed/lotr_clean_perturbed-15%.txt"
perturbed_ds_20 = "../data/texts/perturbed/lotr_clean_perturbed-20%.txt"

if not os.path.exists("../results/experiment3"):
    os.makedirs("../results/experiment3")

###
edit_distance = 1

print("Starting training…")
start = time.time()

hmm = HMM(1, max_edits=edit_distance, max_states=5)
hmm.train(words_ds=words_ds,
          sentences_ds=sentences_ds,
          typo_ds=typo_ds_train)

end = time.time()
train_time = end - start

print("Ended training in {:4.2f} seconds".format(train_time))

model = {"max_edits": [hmm.max_edits], "max_states": [hmm.max_states], "order": [hmm.order],
         "state_len": [hmm.state_len], "error_model_p": [hmm.error_model['p']], 'train_time': [train_time],
         'language_ds': words_ds, 'sentence_ds': sentences_ds, 'typo_ds_train': typo_ds_train,
         'typo_ds_test': typo_ds_test, 'edit_distance': edit_distance}
m = pd.DataFrame(model)
m.to_csv("../results/experiment3/1-model.csv", sep=',', index=False)

## Typo:
# On test set
prediction_typo_test_filename = "../results/experiment3/1-typo_prediction-test.csv"
meta_typo_test_filename = "../results/experiment3/1-meta_typo_prediction-test.csv"

eval.prediction_hmm_candidate_test(typo_ds_test, hmm, prediction_typo_test_filename, meta_typo_test_filename)
eval.evaluation_hmm_candidate_test(prediction_typo_test_filename, meta_typo_test_filename)

# On train set
prediction_typo_train_filename = "../results/experiment3/1-typo_prediction-train.csv"
meta_typo_train_filename = "../results/experiment3/1-meta_typo_prediction-train.csv"

eval.prediction_hmm_candidate_test(typo_ds_train, hmm, prediction_typo_train_filename, meta_typo_train_filename)
eval.evaluation_hmm_candidate_test(prediction_typo_train_filename, meta_typo_train_filename)

## Sentence
# Perturbation 5%
prediction_sentence_filename = "../results/experiment3/1-sentence_prediction-5%.csv"
meta_sentence_filename = "../results/experiment3/1-meta_sentence_prediction-5%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_5, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_5)

# Perturbation 10%
prediction_sentence_filename = "../results/experiment3/1-sentence_prediction-10%.csv"
meta_sentence_filename = "../results/experiment3/1-meta_sentence_prediction-10%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_10, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_10)

# Perturbation 15%
prediction_sentence_filename = "../results/experiment3/1-sentence_prediction-15%.csv"
meta_sentence_filename = "../results/experiment3/1-meta_sentence_prediction-15%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_15, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_15)

# Perturbation 20%
prediction_sentence_filename = "../results/experiment3/1-sentence_prediction-20%.csv"
meta_sentence_filename = "../results/experiment3/1-meta_sentence_prediction-20%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_20, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_20)

###
edit_distance = 2
print("Starting training…")
start = time.time()

hmm = HMM(1, max_edits=edit_distance, max_states=5)
hmm.train(words_ds=words_ds,
          sentences_ds=sentences_ds,
          typo_ds=typo_ds_train)

end = time.time()
train_time = end - start

print("Ended training in {:4.2f} seconds".format(train_time))

model = {"max_edits": [hmm.max_edits], "max_states": [hmm.max_states], "order": [hmm.order],
         "state_len": [hmm.state_len], "error_model_p": [hmm.error_model['p']], 'train_time': [train_time],
         'language_ds': words_ds, 'sentence_ds': sentences_ds, 'typo_ds_train': typo_ds_train,
         'typo_ds_test': typo_ds_test, 'edit_distance': edit_distance}
m = pd.DataFrame(model)
m.to_csv("../results/experiment3/2-model.csv", sep=',', index=False)

## Typo:
# On test set
prediction_typo_test_filename = "../results/experiment3/2-typo_prediction-test.csv"
meta_typo_test_filename = "../results/experiment3/2-meta_typo_prediction-test.csv"

eval.prediction_hmm_candidate_test(typo_ds_test, hmm, prediction_typo_test_filename, meta_typo_test_filename)
eval.evaluation_hmm_candidate_test(prediction_typo_test_filename, meta_typo_test_filename)

# On train set
# prediction_typo_train_filename = "../results/experiment3/2-typo_prediction-train.csv"
# meta_typo_train_filename = "../results/experiment3/2-meta_typo_prediction-train.csv"

# eval.prediction_hmm_candidate_test(typo_ds_train, hmm, prediction_typo_train_filename, meta_typo_train_filename)
# eval.evaluation_hmm_candidate_test(prediction_typo_train_filename, meta_typo_train_filename)

## Sentence
# Perturbation 5%
prediction_sentence_filename = "../results/experiment3/2-sentence_prediction-5%.csv"
meta_sentence_filename = "../results/experiment3/2-meta_sentence_prediction-5%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_5, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_5)

# Perturbation 10%
prediction_sentence_filename = "../results/experiment3/2-sentence_prediction-10%.csv"
meta_sentence_filename = "../results/experiment3/2-meta_sentence_prediction-10%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_10, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_10)

# Perturbation 15%
prediction_sentence_filename = "../results/experiment3/2-sentence_prediction-15%.csv"
meta_sentence_filename = "../results/experiment3/2-meta_sentence_prediction-15%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_15, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_15)

# Perturbation 20%
prediction_sentence_filename = "../results/experiment3/2-sentence_prediction-20%.csv"
meta_sentence_filename = "../results/experiment3/2-meta_sentence_prediction-20%.csv"

eval.prediction_hmm_sequence_test(sentences_ds, perturbed_ds_20, hmm, prediction_sentence_filename,
                                  meta_sentence_filename)
eval.evaluation_hmm_sequence_test(prediction_sentence_filename, meta_sentence_filename, perturbed_ds_20)
