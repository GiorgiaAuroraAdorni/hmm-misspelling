import multiprocessing
from collections import Counter, OrderedDict, defaultdict
import itertools
import networkx as nx
import edlib as el
import pprint
import csv
import pickle
import re

pp = pprint.PrettyPrinter(indent=4)


def process_init(_hmm):
    global hmm

    # Store a copy of the HMM model in each process
    hmm = _hmm


def process_map(input):
    global hmm

    word, i, max_states, pid, nprocesses = input

    candidates = hmm.known(hmm.edits(word, i, pid, nprocesses))

    n_candidates = len(candidates)
    results = [(c, hmm.compute_probability(typed=word, intended=c, n_candidates=n_candidates)) for c in candidates]

    results = sorted(results, key=lambda c: c[1], reverse=True)

    # Conjecture: If each process returns `max_states` distinct results, then
    # merging and deduplicating all the results will yield at least `max_states`
    # globally distinct results.
    return results[:max_states]


class HMM:

    def __init__(self, order, max_edits, max_states):

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

        # Multiprocessing
        # Do not initialize the pool here because it'd then throw an exception
        # in save(...) since it can't be pickled.
        self.pool = None

        return

    @staticmethod
    def load(file):
        with open(file, "rb") as f:
            model = pickle.load(f)

        # Hide the latency of setting up the worker processes by starting them
        # before a request comes in.
        model.setup_multiprocessing()

        return model

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def setup_multiprocessing(self):
        if self.pool is None:
            self.pool = multiprocessing.Pool(initializer=process_init, initargs=[self])

    def _graph_init(self):
        return defaultdict(list)

    def _default_sub_probability(self):
        return 1e-4

    def _error_model_sub_init(self):
        return defaultdict(self._default_sub_probability)

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
            reader = csv.reader(f)
            lines = [row for row in reader]

            for line in lines:
                word = line[0]
                self.language_model[word] = float(line[1])

        # Training the error model
        with open(typo_ds, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            obs = [row for row in reader]
            self.error_model = {"sub": defaultdict(self._error_model_sub_init),
                                "swap": defaultdict(self._error_model_sub_init),
                                "ins": defaultdict(self._error_model_sub_init),
                                "del": defaultdict(self._error_model_sub_init),
                                "p": 0}

        ngram_counter = Counter()

        correct_character_count = 0

        for elem in obs:

            typo = edited_typo = elem[0]
            correct = elem[1]

            special = '[@_!#$%^&*()<>?/\\|}{~:]'
            if any([x for x in correct if x in special]):
                continue
            if any([x for x in typo if x in special]):
                continue

            # Editing typed string to account for accidental insertions, deletions and swaps to align letters, counting them:
            # Ex: steet = st$eet (accidental deletion, index accounted by special char $)
            #     mapes = maps (accidental insertion, index accounted by removing the extra char)
            #     omeh = $ome (deletion + insertion)

            edit_info = el.align(correct, typo, task="path")
            cigar = edit_info["cigar"]
            self.error_model["p"] += edit_info["editDistance"]
            correct_character_count += len(correct)

            # If typo and correct share the same letters, are of the same length, and the cigar has one sequence of 1 deletion, 1 match and 1 insertion. that means there are only swap errors in the typo
            if set(correct) == set(typo) and len(correct) == len(typo) and "1D1=1I" in cigar:
                l = zip(edited_typo, correct)

                already_swapped = False
                for i, j in l:
                    if i != j and not already_swapped:
                        self.error_model["swap"][j][i] += 1
                        already_swapped = True
                    else:
                        already_swapped = False

            else:
                pos = -1
                edited_typo = typo

                for idx, op in re.findall('(\d+)([IDX=])?', cigar):
                    idx = int(idx)
                    pos += idx

                    if op == "I":
                        if pos == 1 or pos > len(correct):
                            prev = "$"
                        else:
                            prev = correct[pos - 1]

                        self.error_model["del"][prev][correct[pos]] += 1

                        edited_typo = edited_typo[:pos] + "$" * idx + edited_typo[pos:]
                    elif op == "D":
                        if pos == 1 or pos > len(correct):
                            prev = "$"
                        else:
                            prev = edited_typo[pos - 1]

                        self.error_model["ins"][prev][edited_typo[pos]] += 1
                        edited_typo = edited_typo[:pos - idx + 1] + edited_typo[pos:]
                        pos -= idx

                l = zip(edited_typo, correct)
                for i, j in l:
                    if i == "$":
                        continue
                    self.error_model["sub"][i][j] += 1

                ngrams = self.find_ngrams(correct, 1) + self.find_ngrams(correct, 2)

                for gram in ngrams:
                    ngram_counter[gram] += 1

            if correct in self.graph:
                self.graph[correct]["obs"].append(typo)

        # Normalization
        unigrams_counter = [v for k, v in ngram_counter.items() if len(k) == 1]
        avg_uni = sum(unigrams_counter) / len(unigrams_counter)
        for key in self.error_model["sub"]:
            for subkey in self.error_model["sub"][key]:
                if ngram_counter[key] == 0:
                    # The letter (key) doesn't appear in the dataset as a correct letter, dividing by the mean of unigram_counter

                    self.error_model["sub"][key][subkey] /= avg_uni
                else:
                    self.error_model["sub"][key][subkey] /= ngram_counter[key]

        for key in self.error_model["ins"]:
            for subkey in self.error_model["ins"][key]:
                if ngram_counter[key] == 0:
                    self.error_model["ins"][key][subkey] /= avg_uni
                else:
                    self.error_model["ins"][key][subkey] /= ngram_counter[key]

        bigrams_counter = [v for k, v in ngram_counter.items() if len(k) == 2]
        avg_bi = sum(bigrams_counter) / len(bigrams_counter)
        for key in self.error_model["del"]:
            for subkey in self.error_model["del"][key]:
                if ngram_counter[key + subkey] == 0:
                    self.error_model["del"][key][subkey] /= avg_bi
                else:
                    self.error_model["del"][key][subkey] /= ngram_counter[key + subkey]

        for key in self.error_model["swap"]:
            for subkey in self.error_model["swap"][key]:
                if ngram_counter[key + subkey] == 0:
                    self.error_model["swap"][key][subkey] /= avg_bi
                else:
                    self.error_model["swap"][key][subkey] /= ngram_counter[key + subkey]

        self.error_model["p"] /= correct_character_count

    def init_trellis(self):
        self.trellis.clear()
        self.trellis_depth = 1
        self.trellis.add_node(0, name="•")

    def empty_trellis(self):
        if len(self.trellis) == 1:
            return True
        else:
            return False

    def build_trellis(self, word):
        states = self.candidates(word)
        if self.empty_trellis():
            for state, probability in states:
                # probability is P(intended|typed) = P(typed|intended)P(intended) where intended = state, typed = word
                # We can use it as is

                new_id = len(self.trellis)
                self.trellis.add_node(new_id, name=state, depth=self.trellis_depth)
                self.trellis.add_edge(0, new_id, weight=probability)
        else:
            # Get leaf nodes representing last states
            leaves = [x for x, v in self.trellis.nodes(data=True)
                      if self.trellis.out_degree(x) == 0
                      and v["depth"] == self.trellis_depth - 1]

            for state, probability in states:
                p = {}
                # probability is P(intended|typed) = P(typed|intended)P(intended) where intended = state, typed = word
                # The emission probability of observation word for the current state is just P(typed|intended), extract
                # it dividing by P(intended).
                obs_prob = probability / self.P(state)

                for leaf_id in leaves:
                    leaf = self.trellis.node[leaf_id]["name"]

                    # Transition probability from the leaf state (previous one) to the current state
                    N_trans = len(self.graph[leaf]["next"])
                    trans_freq = self.graph[leaf]["next"].count(state)
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

        # TODO: re-add a DEBUG flag if you need to enable this
        # self.plot_trellis()

    def most_likely_sequence(self, output_str=True):

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

        if output_str:
            out = " ".join(corrected_words)
        else:
            # Return the list of corrected words and the list of node indices
            out = corrected_words, seq

        return out

    def predict_sequence(self, sequence, output_str=True):

        self.init_trellis()

        if isinstance(sequence, str):
            words = sequence.split()
        else:
            words = sequence

        for word in words:
            self.build_trellis(word)

        return self.most_likely_sequence(output_str)

    def edits(self, word, n=1, pid=None, nprocesses=None):
        if n == 1:
            if pid is not None:
                # Operate on a portion of the possible splits
                total = len(word) + 1

                count = total // nprocesses

                begin = pid * count
                end = begin + count

                if pid == nprocesses - 1:
                    end += total % nprocesses

                split_range = range(begin, end)
            else:
                split_range = range(len(word) + 1)

            letters = "abcdefghijklmnopqrstuvwxyz"
            splits = [(word[:i], word[i:]) for i in split_range]

            deletes = (L + R[1:] for L, R in splits if R)
            transposes = (L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1)
            replaces = (L + c + R[1:] for L, R in splits if R for c in letters)
            inserts = (L + c + R for L, R in splits for c in letters)

            return itertools.chain(deletes, transposes, replaces, inserts)
        else:
            return itertools.chain.from_iterable(self.edits(e1, n - 1) for e1 in self.edits(word, 1, pid, nprocesses))

    def known(self, words):
        return set(w for w in words if w in self.language_model)

    def P(self, word):
        if word in self.language_model:
            return self.language_model[word]
        else:
            return 1e-6

    def compute_probability(self, typed, intended, n_candidates):

        edit_info = el.align(intended, typed, task="path")
        cigar = edit_info["cigar"]

        if not self.known([typed]):
            # Correcting non-word errors - typed word is not in the vocabulary

            prob = 1

            edit_info = el.align(intended, typed, task="path")
            cigar = edit_info["cigar"]

            # If it's a swap it's not anything else
            if set(intended) == set(typed) and \
                    len(intended) == len(typed) and \
                    "1D1=1I" in cigar:

                l = zip(typed, intended)
                already_swapped = False

                for i, j in l:
                    if i != j and not already_swapped:
                        prob *= self.error_model["swap"][j][i]
                        already_swapped = True
                    else:
                        already_swapped = False

            else:
                # Editing typed to account for accidental insertions or deletions
                # Also factoring in insertion or deletion probabilities
                pos = -1
                edited = typed

                for idx, op in re.findall(r'''(\d+)([IDX=])?''', cigar):
                    idx = int(idx)
                    pos += idx

                    if op == "I":
                        if pos == 1 or pos > len(intended):
                            prev = "#"
                        else:
                            prev = intended[pos - 1]

                        edited = edited[:pos] + "$" * idx + edited[pos:]
                        prob *= self.error_model["del"][prev][intended[pos]]

                    elif op == "D":
                        if pos == 1 or pos > len(intended):
                            prev = "#"
                        else:
                            prev = intended[pos - 1]

                        prob *= self.error_model["ins"][prev][edited[pos]]

                        edited = edited[:pos - idx + 1] + edited[pos:]
                        pos -= idx

                # Factoring in substitution probabilities
                l = zip(edited, intended)
                for i, j in l:
                    if i == "$":
                        continue
                    prob *= self.error_model["sub"][i][j]

                # Boosting parameter to rank higher up candidates at shorter edit distances
                parameter = 1 / (int(edit_info["editDistance"]) + 1)
                prob *= self.P(intended) * parameter

        else:
            # Correcting real-word errors - typed word is in the vocabulary

            # Probability of mistaking a word for another, assumed to vary for different tasks
            alpha = 0.98

            prob = 1

            if typed == intended:
                const = alpha
                prob *= const
            else:
                # If typed != intended, redistribute 1 - alpha evenly for all other candidate corrections of the noisy channel
                const = (1 - alpha) / n_candidates

            # If it's a swap it's not anything else
            if set(intended) == set(typed) and \
                    len(intended) == len(typed) and \
                    "1D1=1I" in cigar:

                l = zip(typed, intended)
                already_swapped = False
                for i, j in l:
                    if i != j and not already_swapped:
                        prob *= const
                        already_swapped = True
                    else:
                        already_swapped = False

            else:
                # Editing typo to account for accidental insertions or deletions
                # Also factoring in insertion or deletion probabilities
                pos = -1
                edited = typed

                for idx, op in re.findall(r'''(\d+)([IDX=])?''', cigar):
                    idx = int(idx)
                    pos += idx

                    if op == "I":
                        if pos == 1 or pos > len(intended):
                            prev = "$"
                        else:
                            prev = intended[pos - 1]

                        edited = edited[:pos - 1] + "$" + edited[pos - 1:]
                        prob *= const

                    elif op == "D":
                        if pos == 1 or pos > len(intended):
                            prev = "$"
                        else:
                            prev = intended[pos - 1]

                        edited = edited[:pos - 1] + edited[pos:]
                        prob *= const
                        pos -= idx

                # Factoring in substitution probabilities
                l = zip(edited, intended)
                for i, j in l:
                    if i == "$":
                        continue
                    if i != j:
                        prob *= const

                parameter = 1 / (int(edit_info["editDistance"]) + 1)
                prob *= self.P(intended) * parameter

        return prob

    def candidates(self, word, max_states=None):
        self.setup_multiprocessing()

        word = self.reduce_lengthening(word.lower())

        if max_states is None:
            max_states = self.max_states

        results = dict()
        for i in range(1, self.max_edits + 1):
            nprocesses = multiprocessing.cpu_count()
            input = [(word, i, max_states, pid, nprocesses) for pid in range(nprocesses)]

            subprocesses = self.pool.imap_unordered(process_map, input)

            for subprocess in subprocesses:
                results.update(subprocess)

        results = sorted(results.items(), key=lambda c: c[1], reverse=True)

        # If no word was found not in the language model, leave the typo as the only candidate
        if len(results) == 0:
            results = [(word, 1)]

        return results[:max_states]

    def reduce_lengthening(self, word):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", word)

    def find_ngrams(self, input_list, n):
        return list("".join(x) for x in zip(*[input_list[i:] for i in range(n)]))

    def plot_trellis(self, highlight_path=None, show=True):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout

        plt.figure(1)
        G = self.trellis

        labels = {e[0]: e[1]["name"] for e in G.nodes(data=True)}
        pos = graphviz_layout(G, prog='dot')
        edge_color = None

        if highlight_path is not None:
            i = 0
            edge_color = []

            for edge_from, edge_to in G.edges():
                if (i == 0                  and edge_from == 0                     and edge_to == highlight_path[0]) or \
                   (i < len(highlight_path) and edge_from == highlight_path[i - 1] and edge_to == highlight_path[i]):

                    edge_color.append('r')
                    i += 1
                else:
                    edge_color.append('k')

        nx.draw(G, pos=pos, labels=labels, edge_color=edge_color, node_size=500, node_color='w')

        if show:
            # show() should not be called when opening plots from the GUI, since
            # it blocks the whole application. The GUI code takes care of
            # correctly showing the figure.
            plt.show()
