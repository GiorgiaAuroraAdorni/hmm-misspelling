import random

class Markov:

    def __init__(self, order, mode):
        self.order = order                  # Order of the Markov chain
        self.state_len = self.order + 1     # Length of the grouping
        self.text = None                    # Training text
        self.graph = {}                     # State graph
        self.mode = mode                    # Word or ngram
        return

    def train(self, filename):

        with open(filename, "r") as f:
            if self.mode == "word":
                self.text = f.read().split()
            # TODO: splitting text by ngram

        # Getting state and subsequent element
        for i in range(0, len(self.text) - self.state_len):
            key = tuple(self.text[i : i + self.order])
            value = self.text[i + self.order]

            if key in self.graph:
                self.graph[key].append(value)
            else:
                self.graph[key] = [value]


    def generate(self, length, starting_word = None):
        if not starting_word:
            index = random.randint(0, len(self.text) - self.order)
            result = self.text[index : index + self.order]
        # TODO: given starting_word case

        for i in range(length):
            state = tuple(result[len(result) - self.order :])
            next_word = random.choice(self.graph[state])
            result.append(next_word)

        return " ".join(result[self.order :])