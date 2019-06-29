import os
import sys
from hmm import HMM

data_dir = sys.argv[1]
output_dir = sys.argv[2]

hmm = HMM(1, max_edits=2, max_states=3)
hmm.train(words_ds=os.path.join(data_dir, "word_freq", "lotr_language_model.txt"),
          sentences_ds=os.path.join(data_dir, "texts", "lotr_clean.txt"),
          typo_ds=os.path.join(data_dir, "typo", "clean", "lotr_train.csv"))

hmm_file = os.path.join(output_dir, "hmm.pickle")
hmm.save(hmm_file)

# Check that the model can be loaded
HMM.load(hmm_file)
