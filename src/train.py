from hmm import HMM

hmm = HMM(1, max_edits=2, max_states=3)
hmm.train(words_ds="../data/word_freq/frequency-alpha-gcide.txt",
          sentences_ds="../data/texts/big.txt",
          typo_ds="../data/typo/new/train.csv")

hmm.save("../data/hmm.pickle")

HMM.load("../data/hmm.pickle")