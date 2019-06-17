from markov import Markov
from hmm import HMM
import pprint


def markov_test():
    print("### Markov Test")
    m = Markov(3, "word")
    m.train("../data/texts/lotr_intro.txt")

    generated = m.generate(10)
    print(generated + "\n")


def hmm_test():
    print("### HMM Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits = 2, max_states = 3)
    hmm.train(words_ds = "../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds = "../data/texts/lotr_intro.txt", 
              typo_ds = "../data/typo/typo-corpus-r1.csv")
    pp.pprint("Typed: ambigos")
    x = hmm.candidates("ambigos")
    #pp.pprint(x)

    pp.pprint("Typed: ambigous")
    x = hmm.candidates("ambigous")
    #pp.pprint(x)

    x = hmm.predict("hoem")
    pp.pprint(x)




#markov_test()
hmm_test()