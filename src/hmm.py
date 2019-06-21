from networkx.drawing.nx_agraph import graphviz_layout
from collections import Counter, OrderedDict, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import edlib as el
import random
import pprint
import csv
import pickle
import re

pp = pprint.PrettyPrinter(indent=4)
DEBUG = False


class HMM:

    def __init__(self, order, max_edits, max_states):  # FIXME order parameter not used

        # HMM parameters
        self.order = 1
        self.state_len = self.order + 1
        self.max_edits = max_edits
        self.max_states = max_states

        # HMM structure
        self.graph = defaultdict(self._graph_init)
        self.trellis = nx.DiGraph()
        self.trellis_depth = 0

        # Probability models
        self.language_model = Counter()
        self.error_model = {}

        return

    @staticmethod
    def load(file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def _graph_init():
        return defaultdict(list)

    def train(self, words_ds, sentences_ds, typo_ds):

        # Training the hidden markov chain
        with open(sentences_ds, "r", encoding="utf-8") as f:
            words = f.read().split()

        for i in range(0, len(words) - self.state_len):
            state = words[i: i + self.order][0]
            next_s = words[i + self.order]

            self.graph[state]["next"].append(next_s)

        # Importing the language model
        with open(words_ds, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                elem = line.split()
                self.language_model[elem[1].lower()] = float(elem[3]) / 100

        # Training the error model
        with open(typo_ds, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            obs = [row for row in reader]
            self.error_model = {"sub": defaultdict(lambda: defaultdict(lambda: 1e-4)), "swap": 0, "ins": 0, "del": 0, "p": 0}

        c_sub = Counter()
        correct_character_count = 0

        for elem in obs:
            typo = edited_typo = elem[0]
            correct = elem[1]

            # Editing typed string to account for accidental insertions, deletions and swaps to align letters, counting them:
            # Ex: steet = st$eet (accidental deletion, index accounted by special char $)
            #     mapes = maps (accidental insertion, index accounted by removing the extra char)
            #     omeh = $ome (deletion + insertion)

            edit_info = el.align(correct, typo, task = "path")
            cigar = edit_info["cigar"]
            self.error_model["p"] += edit_info["editDistance"]
            correct_character_count += len(correct)

            # If the alphabet of both typo and correct is equal to the length of correct, it means its the same alphabet as correct's, so it's a swap
            # If it's a swap it's not any other error in this model
            if edit_info["alphabetLength"] == len(correct):
                l = zip(edited_typo, correct)
                for i, j in l:
                    if i != j:
                        self.error_model["swap"] += 1
            else:
                edited_typo = typo
                pos = 0
                for idx, op in re.findall('(\d+)([IDX=])?', cigar):
                    idx = int(idx)
                    pos += idx
                    if op == "I":
                        self.error_model["ins"] += 1
                        edited_typo = edited_typo[:pos - 1] + "$" + edited_typo[pos - 1:]
                    elif op == "D":
                        self.error_model["del"] += 1
                        edited_typo = edited_typo[:pos - 1] + edited_typo[pos:]
                
                # Counting the frequency of substitutions and swaps between letters
                # A typed word contains a swap between two adjacent letters if given two adjacent letters x, y of the typed word, the 
                l = zip(edited_typo, correct)
                for i, j in l:
                    if i == "$":
                        continue
                    c_sub[j] += 1
                    self.error_model["sub"][i][j] += 1

            if correct in self.graph:
                self.graph[correct]["obs"].append(typo)

        # Removing special characters
        special = '[@_!#$%^&*()<>?/\\|}{~:]'
        keys = [k for k, v in self.error_model["sub"].items() if k in special]
        for k in keys:
            self.error_model["sub"].pop(k, None)

        # Normalization
        avg = sum(c for c in c_sub.values()) / len(c_sub)
        for key in self.error_model["sub"]:
            for subkey in self.error_model["sub"][key]:
                if c_sub[key] == 0:
                    # The letter (key) doesn't appear in the dataset as a correct letter, dividing by the mean of c_sub
                    
                    self.error_model["sub"][key][subkey] /= avg
                else:
                    self.error_model["sub"][key][subkey] /= c_sub[key]

        total = len(obs)
        self.error_model["ins"] /= total
        self.error_model["del"] /= total
        self.error_model["swap"] /= total
        self.error_model["p"] /= correct_character_count

    def init_trellis(self):
        self.trellis.clear()
        self.trellis_depth = 1
        self.trellis.add_node(0, name="")

    def empty_trellis(self):
        if len(self.trellis) == 1:
            return True
        else:
            return False

    def build_trellis(self, word):
        states = self.candidates(word)
        states = [state[0] for state in states]
        if self.empty_trellis():
            for state in states:

                N_obs = len(self.graph[state]["obs"])
                obs_freq = self.graph[state]["obs"].count(word)
                if N_obs == 0 or obs_freq == 0:
                    obs_prob = 1e-6
                else:
                    obs_prob = self.graph[state]["obs"].count(word) / N_obs

                init_prob = self.language_model[state]
                p = obs_prob * init_prob

                new_id = len(self.trellis)
                self.trellis.add_node(new_id, name=state, depth=self.trellis_depth)
                self.trellis.add_edge(0, new_id, weight=p)
        else:
            # Get leaf nodes representing last states

            leaves = [x for x, v in self.trellis.nodes(data=True)
                      if self.trellis.out_degree(x) == 0
                      and v["depth"] == self.trellis_depth - 1]

            for state in states:
                p = {}
                for leaf_id in leaves:
                    leaf = self.trellis.node[leaf_id]["name"]

                    # Emission probability of observation word for the current state
                    N_obs = len(self.graph[state]["obs"])
                    obs_freq = self.graph[state]["obs"].count(word)
                    if N_obs == 0 or obs_freq == 0:
                        obs_prob = 0.1e-6
                    else:
                        obs_prob = obs_freq / N_obs

                    # Transition probability from the leaf state (previous one) to the current state
                    N_trans = len(self.graph[leaf_id]["next"])
                    trans_freq = self.graph[leaf_id]["next"].count(state)
                    if N_trans == 0 or trans_freq == 0:
                        trans_prob = 1e-6
                    else:
                        trans_prob = trans_freq / N_trans

                    # Previous state probability
                    predecessor = list(self.trellis.predecessors(leaf_id))
                    # At one time there's always a single predecessor
                    predecessor = predecessor[0]
                    prev_state_prob = self.trellis.edges[predecessor, leaf_id]["weight"]

                    p[leaf_id] = obs_prob * trans_prob * prev_state_prob
  
                # Connecting a state to a leaf only if leaf->state is the path with the local maximal probability
                max_key = max(p, key=p.get)
                new_id = len(self.trellis)
                self.trellis.add_node(new_id, name=state, depth=self.trellis_depth)
                self.trellis.add_edge(max_key, new_id, weight=p[max_key])

        self.trellis_depth += 1

        if DEBUG:
            plt.figure()
            G = self.trellis
            # FIXME: labels are very ugly
            labels = {e[0]: e[1]["name"] + " " + str(e[0]) for e in G.nodes(data=True)}
            topological_node = list(reversed(list(nx.topological_sort(G))))

            pos = graphviz_layout(G, prog='dot')
            nx.draw(G, pos=pos, labels=labels)
            # nx.draw_networkx_edge_labels(G, pos)

    def most_likely_sequence(self):
        leaves = [x for x in self.trellis.nodes()
                  if self.trellis.out_degree(x) == 0]

        leaves = [x for x, v in self.trellis.nodes(data=True)
                  if self.trellis.out_degree(x) == 0
                  and v["depth"] == self.trellis_depth - 1]

        p = {}
        for leaf_id in leaves:
            # Previous state probability
            predecessor = list(self.trellis.predecessors(leaf_id))
            predecessor = predecessor[0]
            prev_state_prob = self.trellis.edges[predecessor, leaf_id]["weight"]
            p[leaf_id] = prev_state_prob

        # Finding global maximum probability between last leaf states (Viterbi)
        max_key = max(p, key=p.get)
        seq = nx.shortest_path(self.trellis, source=0, target=max_key)
        corrected_words = []

        seq.pop(0)
        for i in seq:
            corrected_words.append(self.trellis.nodes[i]["name"])

        string = " ".join(corrected_words)
        return string

    def predict_sequence(self, sequence):
        self.init_trellis()
        words = sequence.split()

        for word in words:
            self.build_trellis(word)

        return self.most_likely_sequence()

    def edits(self, word, n=1):

        if n == 1:
            letters = "abcdefghijklmnopqrstuvwxyz"
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [L + R[1:] for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
            replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
            inserts = [L + c + R for L, R in splits for c in letters]

            return set(deletes + transposes + replaces + inserts)

        elif n == 2:
            return [e2 for e1 in self.edits(word, 1)
                    for e2 in self.edits(e1, 1)]
        else:
            return [e2 for e1 in self.edits(word, 1)
                    for e2 in self.edits(e1, n - 1)]

    def known(self, words):
        return set(w for w in words if w in self.language_model)

    def P(self, word):
        if word in self.language_model:
            return self.language_model[word]
        else:
            return 1e-6

    def candidates(self, word):
        cand = {}
        for i in range(1, self.max_edits + 1):
            # If the word is not in the language model, leave the word as the only candidate
            c = self.known(self.edits(word, i))
            if not c:
                c = set([word])
            cand[i] = c
    
        tmp = defaultdict(float)

        for k, v in cand.items():
            for c in v:
                typo = word
                prob = 1

                insertions = 0
                deletions = 0

                edit_info = el.align(c, typo, task = "path")
                cigar = edit_info["cigar"]

                # If it's a swap it's not anything else
                if edit_info["alphabetLength"] == len(c):
                    l = zip(typo, c)
                    for i, j in l:
                        if i != j:
                            prob *= self.error_model["swap"]
                else:
                    # Editing typo to account for accidental insertions or deletions
                    pos = 0
                    for idx, op in re.findall('(\d+)([IDX=])?', cigar):
                        idx = int(idx)
                        pos += idx
                        if op == "I":
                            insertions += 1
                            typo = typo[:pos - 1] + "$" + typo[pos - 1:]
                        elif op == "D":
                            deletions += 1
                            typo = typo[:pos - 1] + typo[pos:]


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
  
        tmp = OrderedDict(sorted(tmp.items(), key=lambda t: t[1], reverse=True))

        return list(tmp.items())[: self.max_states]
