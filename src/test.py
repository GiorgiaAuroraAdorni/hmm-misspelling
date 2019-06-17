from markov import Markov
from hmm import HMM
import pprint

import matplotlib.pyplot as plt

def markov_test():
    print("### Markov Test")
    m = Markov(3, "word")
    m.train("../data/texts/lotr_intro.txt")

    generated = m.generate(10)
    print(generated + "\n")


def hmm_candidate_test():
    print("### HMM Candidates Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits = 2, max_states = 3)
    hmm.train(words_ds = "../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds = "../data/texts/lotr_intro.txt", 
              typo_ds = "../data/typo/typo-corpus-r1.csv")

    pp.pprint("Typed: hoem")
    x = hmm.candidates("hoem")
    pp.pprint(x)

    pp.pprint("Typed: tome")
    x = hmm.candidates("tome")
    pp.pprint(x)

    pp.pprint("Typed: ambigos")
    x = hmm.candidates("ambigos")
    pp.pprint(x)

    pp.pprint("Typed: ambigous")
    x = hmm.candidates("ambigous")
    pp.pprint(x)

def hmm_predict_test():
    print("### HMM Predict Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits = 2, max_states = 3)
    hmm.train(words_ds = "../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds = "../data/texts/big.txt", 
              typo_ds = "../data/typo/typo-corpus-r1.csv")

    sentence = "becasue shee hes siad tat she woud sendd it o thhe dai".split()
    hmm.init_trellis()
    hmm.predict(sentence[0])
    hmm.predict(sentence[1])
    hmm.predict(sentence[2])
    hmm.predict(sentence[3])
    plt.show()


#markov_test()
#hmm_candidate_test()
hmm_predict_test()