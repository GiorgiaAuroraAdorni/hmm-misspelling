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

    hmm = HMM(1, max_edits=2, max_states=10)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    pp.pprint("Typed: boogs")
    start = time.time()
    x = hmm.candidates("boogs")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: ben")
    start = time.time()
    x = hmm.candidates("ben")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: ambigos")
    start = time.time()
    x = hmm.candidates("ambigos")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("Typed: ambigous")
    start = time.time()
    x = hmm.candidates("ambigous")
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
              typo_ds="../data/typo/clean/train.csv")

    pp.pprint("#1")
    sentence = "becasue shee hes said tat"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#2")
    sentence = "Pierre cae up t hin ad caugt hom by te ams"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#3")
    sentence = "Toady evenqs mawks an epovh tge gteates eioch im pur jistoty"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#4")
    sentence = "the psojuct ghtenyerg ebook of the adventures wv sherlock hslmes by sir jrthur conan doyld 15 in our series by sir arthur conan doyee copyfight laws are changing all over the world"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    correct = " ".join(correct)
    pp.pprint("Corrected: " + correct)


def gen_test():
    print("### HMM Candidates Test")

    hmm = HMM(1, max_edits=2, max_states=5)
    hmm.train(words_ds="../data/word_freq/lotr_language_model.txt",
              sentences_ds="../data/texts/lotr_clean.txt",
              typo_ds="../data/typo/clean/train.csv")

    pp.pprint("Typed: a")
    start = time.time()
    x = hmm.candidates("a")
    end = time.time()
    pred_time = end - start
    pp.pprint("Time: " + str(pred_time))
    pp.pprint(x)

    pp.pprint("#3")
    sentence = "a regular warren by all aucounts"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    pp.pprint("Corrected: " + correct)
    plt.show()


# markov_test()

# hmm_candidate_test()
# hmm_build_trellis_test()
hmm_predict_sequence_test()
# gen_test()
