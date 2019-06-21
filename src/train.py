import os
import sys
from hmm import HMM

data_dir = sys.argv[1]
output_file = sys.argv[2]

hmm = HMM(1, max_edits=2, max_states=3)
hmm.train(words_ds=os.path.join(data_dir, "word_freq", "frequency-alpha-gcide.txt"),
          sentences_ds=os.path.join(data_dir, "texts", "big.txt"),
          typo_ds=os.path.join(data_dir, "typo", "new", "train.csv"))

hmm.save(output_file)

# Check that the model can be loaded
HMM.load(output_file)
