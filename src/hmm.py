import random
import csv
from collections import Counter, OrderedDict, defaultdict
import networkx
import pprint

pp = pprint.PrettyPrinter(indent=4)

class HMM:

    def __init__(self, order, max_edits, max_states):

        # HMM parameters
        self.order = 1                      
        self.state_len = self.order + 1     
        self.max_edits = max_edits
        self.max_states = max_states

        # HMM structure
        self.graph = {}
        self.memory = None

        # Probability models
        self.language_model = Counter()
        self.error_model = {}

        return

    def train(self, words_ds, sentences_ds, typo_ds):

        # Training the hidden markov chain
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

        # Training the error model
        with open(typo_ds, "r") as f:
            reader = csv.reader(f)
            obs = [row for row in reader]
            self.error_model = {"sub": defaultdict(lambda: Counter()), "ins": 0, "del": 0}

        c_sub = Counter()
        for elem in obs:
            typo = elem[0]
            correct = elem[1]

            # Counting accidental insertions and deletions
            if len(typo) > len(correct):
                self.error_model["ins"] += 1
            elif len(typo) < len(correct):
                self.error_model["del"] += 1

            # Editing typed string to account for accidental insertions and deletions to align letters 
            # (to get the correct probabilities of substituted letters):
            # Ex: steet = st$eet (accidental deletion, index accounted by special char $)
            #     mapes = maps (accidental insertion, index accounted by removing the extra char)

            if len(typo) != len(correct):
                edit_sequence = self.diff(typo, correct)
                for op in edit_sequence:
                    index_typo = op["position_typo"]
                    if op["operation"] == "insert":
                        typo = typo[:index_typo] + "$" + typo[index_typo:]
                    else:
                        typo = typo[:index_typo] + typo[index_typo + 1:]
            
            # Counting the frequency of substitutions between letters
            l = zip(typo, correct)
            for i, j in l:
                if i == "$":
                    continue
                c_sub[j] += 1
                self.error_model["sub"][i][j] += 1
            
            if correct in self.graph:
                self.graph[correct]["obs"].append(typo)

        # Removing special characters
        special = '[@_!#$%^&*()<>?/\|}{~:]'
        keys = [k for k,v in self.error_model["sub"].items() if k in special]
        for k in keys:
            self.error_model["sub"].pop(k, None)

        # Normalization
        for key in self.error_model["sub"]:
            for subkey in self.error_model["sub"][key]:
                self.error_model["sub"][key][subkey] /= c_sub[key]

        total = len(obs)
        self.error_model["ins"] /= total
        self.error_model["del"] /= total

    def reset(self):
        self.memory = None

    def predict(self, word):
        states = self.candidates(word)
        if not self.memory:
            self.memory = {}

        pp.pprint(self.memory)


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
                typo = word
                prob = 1

                insertions = 0
                deletions = 0

                # Editing typo to account for accidental insertions or deletions
                if len(typo) != len(c):
                    edit_sequence = self.diff(typo, c)
                    for op in edit_sequence:
                        index_typo = op["position_typo"]
                        if op["operation"] == "insert":
                            typo = typo[:index_typo] + "$" + typo[index_typo:]
                            insertions += 1
                        else:
                            typo = typo[:index_typo] + typo[index_typo + 1:]
                            deletions += 1
                
                # Factoring in insertion or deletion probabilities
                for i in range(0, insertions):
                    prob *= self.error_model["ins"]
                for i in range(0, deletions):
                    prob *= self.error_model["del"]

                # Factoring in substitution probabilities
                l = zip(typo, c)
                for i, j in l:
                    if i == "$":
                        continue
                    prob *= self.error_model["sub"][i][j]
                
                prob *= self.P(c)
                tmp[c] = prob

        tmp = OrderedDict(sorted(tmp.items(), key = lambda t: t[1], reverse = True))
        
        return list(tmp.items())[: self.max_states]

   
    def diff(self, e, f, i=0, j=0):
        #  Returns a minimal list of differences between 2 lists e and f
        #  requring O(min(len(e),len(f))) space and O(min(len(e),len(f)) * D)
        #  worst-case execution time where D is the number of differences.
        #  Documented at http://blog.robertelder.org/diff-algorithm/

        N,M,L,Z = len(e),len(f),len(e)+len(f),2*min(len(e),len(f))+2
        if N > 0 and M > 0:
            w,g,p = N-M,[0]*Z,[0]*Z
            for h in range(0, (L//2+(L%2!=0))+1):
                for r in range(0, 2):
                    c,d,o,m = (g,p,1,1) if r==0 else (p,g,0,-1)
                    for k in range(-(h-2*max(0,h-M)), h-2*max(0,h-N)+1, 2):
                        a = c[(k+1)%Z] if (k==-h or k!=h and c[(k-1)%Z]<c[(k+1)%Z]) else c[(k-1)%Z]+1
                        b = a-k
                        s,t = a,b
                        while a<N and b<M and e[(1-o)*N+m*a+(o-1)]==f[(1-o)*M+m*b+(o-1)]:
                            a,b = a+1,b+1
                        c[k%Z],z=a,-(k-w)
                        if L%2==o and z>=-(h-o) and z<=h-o and c[k%Z]+d[z%Z] >= N:
                            D,x,y,u,v = (2*h-1,s,t,a,b) if o==1 else (2*h,N-a,M-b,N-s,M-t)
                            if D > 1 or (x != u and y != v):
                                return self.diff(e[0:x],f[0:y],i,j)+self.diff(e[u:N],f[v:M],i+u,j+v)
                            elif M > N:
                                return self.diff([],f[N:M],i+N,j+N)
                            elif M < N:
                                return self.diff(e[M:N],[],i+M,j+M)
                            else:
                                return []
        elif N > 0: 
            return [{"operation": "delete", "position_typo": i+n} for n in range(0,N)]
        else:
            return [{"operation": "insert", "position_typo": i,"position_correct":j+n} for n in range(0,M)]