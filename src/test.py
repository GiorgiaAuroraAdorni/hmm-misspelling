import matplotlib.pyplot as plt
from markov import Markov
from hmm import HMM
import time

import pprint

pp = pprint.PrettyPrinter(indent=4)

def markov_test():
    print("### Markov Test")
    m = Markov(3, "word")
    m.train("../data/texts/lotr_clean.txt")

    generated = m.generate(10)
    pp.pprint(generated)
    print("\n")


def hmm_candidate_test():
    print("### HMM Candidates Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=5)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    pp.pprint("Typed: bools")
    start = time.time()
    x = hmm.candidates("bools")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: peculair")
    start = time.time()
    x = hmm.candidates("peculair")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: migt")
    start = time.time()
    x = hmm.candidates("migt")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: littele")
    start = time.time()
    x = hmm.candidates("littele")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: hoem")
    start = time.time()
    x = hmm.candidates("hoem")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: tome")
    start = time.time()
    x = hmm.candidates("tome")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    print("\n")


def hmm_build_trellis_test():
    print("### HMM Build Trellis Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    sentence = "becasue shee hes siad tat she woud sendd it o thhe dai".split()
    hmm.init_trellis()
    hmm.build_trellis(sentence[0])
    hmm.build_trellis(sentence[1])
    hmm.build_trellis(sentence[2])
    hmm.build_trellis(sentence[3])
    print("\n")
    plt.show()


def hmm_predict_sequence_test():
    print("### HMM Predict Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/lotr_train.txt")

    pp.pprint("#1")
    sentence = "wpen mr bilbo bagginx of bag end announcwd that he"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#2")
    sentence = "now beclml a local legend and it wos popultrly believed"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#3")
    sentence = "was too much of f goof thing it seemed unfair"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#4")
    sentence = "so fap trouble had not come and as mr baggins"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)


def gen_test():
    print("### HMM Candidates Test")

    hmm = HMM(1, max_edits=1, max_states=5)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    pp.pprint("Typed: andd")
    start = time.time()
    x = hmm.candidates("andd")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)


# markov_test()

hmm_candidate_test()
# hmm_build_trellis_test()
# hmm_predict_sequence_test()
# gen_test()
