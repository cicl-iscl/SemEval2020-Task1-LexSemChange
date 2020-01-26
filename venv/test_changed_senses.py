from collections import Counter



def changed_sense(all_clusters, cor1_clusters, cor2_clusters, k, threshold):
    """
        Determines whether there has been a change in a word's sense(s). It does not matter whether the word
        has gained or lost sense. A word is classified as gaining a sense, if the sense is never attested in corpus1,
        but attested at least k times in corpus2 (and the other way around).

        :param all_clusters: a list of unique clusters
        :param cor1_clusters:  a dictionary of {cluster1_label: num_occurrences_in_corpus1}
        :param cor2_clusters:  a dictionary of {cluster2_label: num_occurrences_in_corpus1}
        :param k: the number of times a word has to occur in order to be classified as having changed
        :return True: if a word has changed senses, False otherwise
    """
    words_changed = []
    words_unchanged = []
    n_occur = 0

    if len(all_clusters) == 1:
        print("No change as combined corpora have only 1 cluster in total")
        return words_unchanged.append(all_clusters[0]), words_changed


    for label in all_clusters:
        # if word appears in both corpora, ignore it
        if label in cor1_clusters and label in cor2_clusters:
            if cor2_clusters.get(label) > threshold and cor1_clusters.get(label) > threshold:
                print("Label ", label, " is above the threshold in both corpora: ", cor1_clusters.get(label))
                words_unchanged.append(label)
                continue
            if cor1_clusters.get(label) == cor2_clusters.get(label): # no change if the number of data points in both is the same
                print("Label ", label, " has equal number of datapoints", cor1_clusters.get(label))
                words_unchanged.append(label)
                continue
            # IF the number of data points in one of the corpora is below the threshold, count it as change IF K
            if cor2_clusters.get(label) <= threshold:
                n_occur = cor1_clusters.get(label)
            elif cor1_clusters.get(label) <= threshold:
                n_occur = cor2_clusters.get(label)
        # if the word lost a sense (in C1 but not in C2)
        elif label in cor1_clusters and label not in cor2_clusters:
            print("Label ", label, " is in corpus1", cor1_clusters.get(label))
            n_occur = cor1_clusters.get(label)
        # if the word gained a sense (not in C1 but in C2)
        elif label in cor2_clusters and label not in cor1_clusters:
            print("Label ", label, " is in corpus2", cor2_clusters.get(label))
            n_occur = cor2_clusters.get(label)
        if n_occur >= k:
            print("Label ", label, " occurs ", n_occur, " times IN only one of the corpora.", " k: ", k)
            words_changed.append(label)
    return words_changed, words_unchanged

if __name__ == "__main__":

    """ Testing changed_senses with a threshold"""
    k = 1
    unique_labels = [0, 1, 2, 3, 4]
    c1_labels = Counter([0, 0, 0, 1, 2, 2, 2, 3, 4, 4]) # [0, 0, 0, 1, 3, 4, 4])
    c2_labels = Counter([0, 0, 1, 2, 3, 3, 3, 4, 4]) # [0, 0, 1, 3, 3, 3, 4, 4])
    # changed_sense(all_unique_words, cor1_words, cor2_words, k, threshold):
    bla = changed_sense(unique_labels, c1_labels, c2_labels, k, threshold=1) # TODO: Problem: what if C1: 3 and C2:2 and threshold = 2
    print(bla)