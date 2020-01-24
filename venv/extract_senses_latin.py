from cluster_fixed import collect_all_occurrences, load_corpus
from collections import Counter
from pandas.core.common import flatten

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
    results = {}
    for word in all_unique_words:
        # if word appears in both corpora, ignore it
        if word in cor1_words and word in cor2_words:
            results[word] = "in_both"
            print("word ", word, " is in BOTH", cor1_words.get(word))

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
            results[word] = "changed"
    return results

if __name__ == "__main__":

    # 1. Load the corpora
    corpus1 = load_corpus('../starting_kit/trial_data_public/corpora/{}/corpus{}/corpus{}.txt'.format("latin",1,1)) #"latin/corpus1/corpus1.txt"
    corpus2 = load_corpus('../starting_kit/trial_data_public/corpora/{}/corpus{}/corpus{}.txt'.format("latin",2,2)) # "latin/corpus2/corpus2.txt"

    # 2. Get the num of occurrences of each word {'sum': 1104, 'et': 698, 'in': 675, 'quis#2': 550}
    c1_occurrences = Counter(list(flatten(corpus1)))
    c2_occurrences = Counter(list(flatten(corpus2)))

    # 3. Select words with hash tags {'dico#2', 148, 'Brutus#2', 2, 'volo#1', 40}
    keys_hash_c1 = {k: v for k, v in c1_occurrences.items() if '#' in k}
    keys_hash_c2 = {k: v for k, v in c2_occurrences.items() if '#' in k}

    # Print what we've got so far
    print("---------------------------------------------")
    print("Corpus 1:")
    print(c1_occurrences)
    print("---------------------------------------------")
    for i in keys_hash_c1.items():
        print(i)
    print("---------------------------------------------")

    print("Corpus 2:")
    print("---------------------------------------------")
    for i in keys_hash_c2.items():
        print(i)

    # 4. Print the unique keys
    unique_keys = sorted(list(set(list(keys_hash_c1.keys()) + list(keys_hash_c2.keys()))))
    print("---------------------------------------------")
    print("Unique keys from both corpora: ", len(unique_keys))
    for i in unique_keys:
        print(i)
    print("---------------------------------------------")

    # 5. Write unique keys to file
    write_to_file(unique_keys, "out/unique_keys_hash.txt")

    # 6. Check if changed sense
    k = 2
    res = changed_sense(unique_keys, keys_hash_c1, keys_hash_c2, k)
    print("---------------------------------------------")
    print("Printing words that changed senses:")
    for k in res.keys():
        if res.get(k) == "changed":
            print("CHANGED:", k, res.get(k))
    print("Number of words that changed senses:", len(res)) # TODO: this is not correct because some words (with diff senses occur several times)
    print("---------------------------------------------")


