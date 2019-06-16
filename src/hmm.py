import random
import csv
from collections import Counter, OrderedDict, defaultdict
import pprint

class HMM:

    def __init__(self, order, max_edits, max_states):

        # HMM parameters
        self.order = 1                      
        self.state_len = self.order + 1     
        self.max_edits = max_edits
        self.max_states = max_states

        # HMM structure
        self.graph = {}
        self.memory = {}

        # Probablity models
        self.language_model = Counter()
        self.error_model = {}

        return

    def train(self, words_ds, sentences_ds, typo_ds):

        # Training the idden markov chain
        with open(sentences_ds, "r") as f:
            words = f.read().split()

        for i in range(0, len(words) - self.state_len):
            state = words[i : i + self.order][0]
            next_s = words[i + self.order]

            if state in self.graph:
                self.graph[state]["next"].append(next_s)
            else:
                self.graph[state] = {"next": [], "obs": []}
                self.graph[state]["next"] = [next_s]
        
        # Importing the language model
        with open(words_ds, "r") as f:
            next(f)
            for line in f:
                elem = line.split()
                self.language_model[elem[1]] = float(elem[3])/100

        # Training error model
        with open(typo_ds, "r") as f:
            reader = csv.reader(f)
            obs = [row for row in reader]
            self.error_model = {"sub": defaultdict(lambda: Counter()), "ins": 0, "del": 0}
                    

        c_ins = 0
        c_del = 0
        c_sub = 0

        for elem in obs:
            typo = elem[0]
            correct = elem[1]
    
            if len(typo) > len(correct):
                self.error_model["ins"] += 1
            elif len(typo) < len(correct):
                self.error_model["del"] += 1
            
            l = zip(typo, correct)
            for i, j in l:
                if i != j:
                    self.error_model["sub"][i][j] += 1
                
            if correct in self.graph:
                self.graph[correct]["obs"].append(typo)

        ## Normalization 
        total = len(obs)
        for key in self.error_model["sub"]:
            for subkey in self.error_model["sub"][key]:
                self.error_model["sub"][key][subkey] /= total

        self.error_model["ins"] /= total
        self.error_model["del"] /= total

    def reset(self):
        self.memory = {}


    def predict(self, word):
        states = self.candidates(word, )
        
        return " ".join(result[self.order :])


    def predict_sequence(self, sequence):
        self.memory = {}
        # iterate on predict


    def edits(self, word, n = 1):

        if n == 1:
            letters    = "abcdefghijklmnopqrstuvwxyz"
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
            deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
            inserts    = [L + c + R               for L, R in splits for c in letters]

            return set(deletes + transposes + replaces + inserts)

        elif n == 2:
            return [e2 for e1 in self.edits(word, 1) 
                       for e2 in self.edits(e1, 1)]
        else:
            return [e2 for e1 in self.edits(word, 1) 
                       for e2 in self.edits(e1, n-1)]


    def known(self, words): return set(w for w in words if w in self.language_model)


    def P(self, word): return self.language_model[word]


    def candidates(self, word): 
        cand = {}
        for i in range(1, self.max_edits+1):
            cand[i] = self.known(self.edits(word, i))

        tmp = defaultdict(float)

        for k, v in cand.items():
            for c in v:
                prob = 1
                
                if len(word) > len(c):
                    prob *= self.error_model["ins"]
                
                elif len(word) < len(c):
                    prob *= self.error_model["del"]

                l = zip(word, c)
                for i, j in l:
                    if i != j:
                        prob *= self.error_model["sub"][i][j]

                prob *= self.P(c)
                tmp[c] = prob

        tmp = OrderedDict(sorted(tmp.items(), key = lambda t: t[1], reverse = True))
        
        return list(tmp.items())[: self.max_states]

