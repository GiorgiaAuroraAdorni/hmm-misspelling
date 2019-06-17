def read_dataset(dataset):
    with open(dataset, "r") as f:
        words = f.read().split()

    return words


def preprocessing1(dataset, cleaned_dataset):
    words = read_dataset(dataset)

    with open(cleaned_dataset, "w+") as new_f:
        key = ""
        for word in words:
            if word.endswith(":"):
                key = word
            else:
                new_f.write(key.split(":")[0])
                new_f.write(",")
                new_f.write(word)
                new_f.write("\n")


def preprocessing2(dataset, cleaned_dataset):
    words = read_dataset(dataset)

    with open(cleaned_dataset, "w+") as new_f:
        key = ""
        for word in words:
            if word.startswith("$"):
                key = word
            else:
                new_f.write(key.split("$")[1])
                new_f.write(",")
                new_f.write(word)
                new_f.write("\n")


aspell_ds = "../data/typo/spelling-corrector/aspell.txt"
wikipedia_ds = "../data/typo/spelling-corrector/wikipedia.txt"
spell_testset1_ds = "../data/typo/spelling-corrector/spell-testset1.txt"
spell_testset2_ds = "../data/typo/spelling-corrector/spell-testset2.txt"
birkbeck_misp_ds = "../data/typo/birkbeck-misp.txt"

aspell_ds_clean = "../data/typo/spelling-corrector/aspell_clean.txt"
wikipedia_ds_clean = "../data/typo/spelling-corrector/wikipedia_clean.txt"
spell_testset1_clean = "../data/typo/spelling-corrector/spell-testset1_clean.txt"
spell_testset2_clean = "../data/typo/spelling-corrector/spell-testset2_clean.txt"
birkbeck_misp_ds_clean = "../data/typo/birkbeck-misp_clean.txt"

preprocessing1(aspell_ds, aspell_ds_clean)
preprocessing1(wikipedia_ds, wikipedia_ds_clean)
preprocessing1(spell_testset1_ds, spell_testset1_clean)
preprocessing1(spell_testset2_ds, spell_testset2_clean)
preprocessing2(birkbeck_misp_ds, birkbeck_misp_ds_clean)
