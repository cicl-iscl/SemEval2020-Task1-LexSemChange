import sys
from collections import Counter
from pandas.core.common import flatten

DIVIDER = "---------------------------------------------"

def load_corpus_hash(filename):
    """
        Loads a corpus into a list. Keeps only the tokens with HASHTAGS!

        :param filename: the name of the file containing the corpus
        :returns tokenized: a list of sentences, where each sentence is a list of words [["I", "eat", "apples"], ["she", "sleeps"]]
    """
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
    tokenized = flatten([line.strip().split() for line in content]) #filter(lambda x: (x % 13 == 0), my_list)
    tokenized = list(filter(lambda x : ("#" in x), tokenized))
    return tokenized

def write_to_file(liste, filename):
    with open(filename, "w+", encoding="utf8") as f:
        for i in liste:
            f.write(i + '\n')

def changed_sense(all_unique_words, cor1_words, cor2_words, k):
    """
        Determines whether there has been a change in a word's sense(s). It does not matter whether the word
        has gained or lost sense. A word is classified as gaining a sense, if the sense is never attested in corpus1,
        but attested at least k times in corpus2 (and the other way around).

        :param all_clusters: a list of unique clusters
        :param cor1_words:  a dictionary of {cluster1_label: num_occurrences_in_corpus1}
        :param cor2_words:  a dictionary of {cluster1_label: num_occurrences_in_corpus1}
        :param k: the number of times a word has to occur in order to be classified as having changed
        :return True: if a word has changed senses, False otherwise
    """
    n_occur = 0
    if len(all_unique_words) == 1:
        print("No change as combined corpora have only 1 cluster in total")
        return False

    words_changed = []
    words_unchanged = []

    for word in all_unique_words:
        # if word appears in both corpora, ignore it
        if word in cor1_words and word in cor2_words:
            words_unchanged.append(word)
            print("word ", word, " is in BOTH", cor1_words.get(word))
            continue

        # if the word lost a sense (in C1 but not in C2)
        if word in cor1_words and word not in cor2_words:
            print("word ", word, " is in corpus1", cor1_words.get(word))
            n_occur = cor1_words.get(word)

        # if the word gained a sense (not in C1 but in C2)
        elif word in cor2_words and word not in cor1_words:
            print("word ",word, " is in corpus2", cor2_words.get(word))
            n_occur = cor2_words.get(word)

        if n_occur >= k:
            print("word ", word, " occurs ", n_occur, " times.", " k: ", k)
            words_changed.append(word)
    return words_changed, words_unchanged

def print_analysis(c1_occurrences, c2_occurrences, unique_keys,changed_senses, unchanged_senses, changed_words):

    # Print what we've got so far
    print(DIVIDER)
    print("Corpus 1:")
    print(DIVIDER)
    print(c1_occurrences)
    print(DIVIDER)
    for i in c1_occurrences.items():
        print(i)
    print(DIVIDER)

    print("Corpus 2:")
    print(DIVIDER)
    print(c2_occurrences)
    print(DIVIDER)
    for i in c2_occurrences.items():
        print(i)


    print(DIVIDER)
    print("Unique SENSES from both corpora: ", len(unique_keys))
    print(DIVIDER)
    for i in unique_keys:
        print("Corpus 1: ", i, c1_occurrences.get(i), "\t\t\tCorpus 2: ",  i, c2_occurrences.get(i))
    print(DIVIDER)


    print(DIVIDER)
    print("Printing senses that CHANGED:")
    print(DIVIDER)
    for i in changed_senses:
        print("CHANGED:", i)
    print(DIVIDER)
    print("Number of senses that changed:",
          len(changed_senses))
    print(DIVIDER)

    print("Printing senses that stayed UNCHANGED:")
    print(DIVIDER)
    for i in unchanged_senses:
        print("UNchanged:", i)
    print(DIVIDER)
    print("Number of senses that did not change:", len(unchanged_senses))
    print(DIVIDER)

    print("Change in WORDS:")
    print(DIVIDER)
    for token in changed_words:
        print("Word: ", token, " changed.")
    print(DIVIDER)
    print("Number of words that changed:", len(changed_words))
    print(DIVIDER)



if __name__ == "__main__":
    try:
        k = int(sys.argv[1])

        # 1. Load the corpora
        corpus1 = load_corpus_hash("latin/corpus1/corpus1.txt") #'../starting_kit/trial_data_public/corpora/{}/corpus{}/corpus{}.txt'.format("latin",1,1)) #
        corpus2 = load_corpus_hash("latin/corpus2/corpus2.txt") #'../starting_kit/trial_data_public/corpora/{}/corpus{}/corpus{}.txt'.format("latin",2,2)) #

        # 2. Get the num of occurrences of each Sprint(unique_without_hash)ENSE (not word!!!) {'quis#2': 550}
        c1_occurrences = Counter(corpus1)
        c2_occurrences = Counter(corpus2)


        # 3. Print the unique keys
        unique_keys = sorted(list(set(list(c1_occurrences.keys()) + list(c2_occurrences.keys())))) # [volo#2, volo#1, verum#1, verum#3]
        unique_without_hash = sorted(list(set([token[:token.index("#")] for token in unique_keys]))) # [volo, verum]

        # 4. Write unique keys to file
        write_to_file(unique_keys, "out/unique_keys_hash.txt")

        # 5. Check which senses changed
        # k = 2 # TODO: change k
        changed_senses, unchanged_senses = changed_sense(unique_keys, c1_occurrences, c2_occurrences, k)

        changed_without_hash = [token[:token.index("#")] for token in changed_senses]

        changed_words = [token for token in unique_without_hash if token in changed_without_hash]


        print_analysis(c1_occurrences,c2_occurrences, unique_keys, changed_senses, unchanged_senses, changed_words)

        write_to_file(changed_senses, "out/changed_senses_k{}.txt".format(k))
        write_to_file(unchanged_senses, "out/unchanged_senses_k{}.txt".format(k))
        write_to_file(changed_words, "out/changed_WORDS_k{}.txt".format(k))
    except IndexError:
        print("Please enter a k that should be used in the changed_sense() method. It has to be smaller than 9.")

