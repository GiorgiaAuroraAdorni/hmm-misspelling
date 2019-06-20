import matplotlib.pyplot as plt
from markov import Markov
from hmm import HMM
import pprint


def markov_test():
    print("### Markov Test")
    pp = pprint.PrettyPrinter(indent=4)
    m = Markov(3, "word")
    m.train("../data/texts/lotr_intro.txt")

    generated = m.generate(10)
    pp.pprint(generated)
    pp.pprint("\n")


def hmm_candidate_test():
    print("### HMM Candidates Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/big.txt",
              typo_ds="../data/typo/new/train.csv")

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
    
    pp.pprint("\n")


def hmm_build_trellis_test():
    print("### HMM Build Trellis Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/big.txt",
              typo_ds="../data/typo/new/train.csv")

    sentence = "becasue shee hes siad tat she woud sendd it o thhe dai".split()
    hmm.init_trellis()
    hmm.build_trellis(sentence[0])
    hmm.build_trellis(sentence[1])
    hmm.build_trellis(sentence[2])
    hmm.build_trellis(sentence[3])
    pp.pprint("\n")
    plt.show()


def hmm_predict_sequence_test():
    print("### HMM Predict Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=3)
    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/big_clean.txt",
              typo_ds="../data/typo/new/train.csv")

    pp.pprint("#1")
    sentence = "becasue shee hes said tat"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#2")
    sentence = "Pierre cae up t hin ad caugt hom by te ams"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#3")
    sentence = "Toady evenqs mawks an epovh tge gteates eioch im pur jistoty"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    pp.pprint("Corrected: " + correct)

    pp.pprint("#4")
    sentence = "the psojuct ghtenyerg ebook of the adventures wv sherlock hslmes by sir jrthur conan doyld 15 in our series by sir arthur conan doyee copyfight laws are changing all over the world"
    pp.pprint("Sentence: " + sentence)
    correct = hmm.predict_sequence(sentence)
    pp.pprint("Corrected: " + correct)

    plt.show()

def gen_test():
    print("### HMM Candidates Test")
    pp = pprint.PrettyPrinter(indent=4)

    hmm = HMM(1, max_edits=2, max_states=10)
    hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
              sentences_ds="../data/texts/big.txt",
              typo_ds="../data/typo/new/train.csv")

    pp.pprint("Typed: 's'")
    x = hmm.candidates("'s'")
    pp.pprint(x)


# markov_test()
# hmm_candidate_test()
# hmm_build_trellis_test()
# hmm_predict_sequence_test()
gen_test()