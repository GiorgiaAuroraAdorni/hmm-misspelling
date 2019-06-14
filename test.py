from markov import Markov

m = Markov(4, "word")
m.train("data/lotr_intro.txt")

generated = m.generate(100)
print(generated)