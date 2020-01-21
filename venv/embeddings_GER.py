import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torch
from elmoformanylangs import Embedder
import sys
import numpy
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

numpy.set_printoptions(threshold=sys.maxsize)
import math
from sklearn.cluster import KMeans


# preprocess English historical corpus to get sentences into lists of lists like this:
# sents = [['s1_word1', 's1_word2', 's1_word3', 's1_word4'],
# ['s2_word1', 's2_word2', 's2_word3', 's2_word4', 's2_word5', 's2_word6', 's2_word7']]
# argument path: path to the historical corpus text file: 'path/to/file.txt'
def preprocess(path):
    sents = []
    with open(path) as f:
        sentences = [elem for elem in f.read().split('\n') if elem]
        for sentence in sentences:
            sents.append(sentence.split())

    return sents


"""
    Removes # from words in the corpus 
    Args:
        orig_corpus: corpus in the format of list of lists: [['s1_word1', 's1_word2', 's1_word3', 's1_word4'],
                        ['s2_word1', 's2_word2', 's2_word3', 's2_word4', 's2_word5', 's2_word6', 's2_word7']]
"""
def clean_corpus(orig_corpus):
    clean_corpus =[]
    for line in orig_corpus:
        sent = [token[:token.index("#")] if "#" in token else token for token in line]
        clean_corpus.append(sent)
        #print(sent)
    return clean_corpus


def collect_all_occurrences(corpus):
    indices = {}
    for i, sent in enumerate(corpus):
        for ind, word in enumerate(sent):
            idx = sent.index(word)
            # if word does not exist, create an entry and add a list of indices as value
            if word not in indices:
                indices[word] = [(i, idx)]
            # if word already exists in the dictionary, add the new indices to the list of indices
            else:
                indices[word].append((i, idx))
            #print("index of current sentence: ", i)
            #print("index of ", word, " in ", sent, " is ", idx)
    return indices


# use PCA and t-SNE to reduce the 1,024 dimensions which are output from ELMo
# down to 2 so that we can review the outputs from the model
# returns the reduced embedding space of shape (no. of words, 2)
def reduce_dimensions(embeddings_array):
    print("unreduced array has dimensions: ", embeddings_array.shape)

    pca = PCA(n_components=50)  # reduce down to 50 dim
    y = pca.fit_transform(embeddings_array)
    reduced = TSNE(n_components=3).fit_transform(y)  # further reduce to 2 dim using t-SNE
    print("finished reduction")
    # print("reduced emb space: ", reduced)
    print("reduced space dimensions: ", reduced.shape)

    return reduced


# args: word_to_plot - string of the word that should be plotted
# path_corpus - path of the corpus that contains the word
# filename_plot - name under which the plot will be saved
# filename_datapoints_words - text file that will contain the data points of all the words from the corpus,
# format: x1   y1   word1
#         x2   y2   word2
# filename_datapoints_specific_word - text file that will contain the data points of only the word_to_plot (same format as before)
# reduced - embedding vectors reduced to 2 dimensions
def plot_word(word_to_plot, path_corpus, filename_plot, filename_datapoints_words, filename_datapoints_specific_word,
              reduced):
    all_tokens = []  # list of the words of the specified corpus
    with open(path_corpus, 'r', encoding="utf8") as f:
        for line in f:
            for word in line.split():
                all_tokens.append(word)

    # creates a text file containing the two dimensional data point of each word:
    # x1   y1   word1
    # x2   y2   word2
    # ...
    count = 0
    with open(filename_datapoints_words, "w+", encoding="utf8") as f:
        for i in reduced:
            string = str(i[0]) + "\t" + str(i[1]) + "\t" + all_tokens[count]
            f.write(string + '\n')
            count += 1

    # plot only a single word, e.g. "walk"
    indices = [i for i, x in enumerate(all_tokens) if x == word_to_plot]

    with open(filename_datapoints_specific_word, "w+", encoding="utf8") as f_word:
        for i in indices:
            string = str(reduced[i][0]) + "\t" + str(reduced[i][1]) + "\t" + all_tokens[i]
            f_word.write(string + '\n')

    # plot
    x_dim = np.loadtxt(filename_datapoints_specific_word, usecols=0)
    y_dim = np.loadtxt(filename_datapoints_specific_word, usecols=1)
    labels = np.loadtxt(filename_datapoints_specific_word, dtype=str, usecols=2)
    ax = plt.subplot()
    ax.scatter(x_dim, y_dim)

    for i, text in enumerate(labels):
        ax.annotate(text, (x_dim[i], y_dim[i]))

    print("saving the plot...")
    plt.title("Context Embeddings for \"" + word_to_plot + "\"")
    plt.xlabel("x_dimension")
    plt.ylabel("y_dimension")
    # save the plot as it's not possible to show the plot when running on the server
    plt.savefig(filename_plot)
    plt.clf()


# args: word_to_extract - string of the word that should be extracted
# path_corpus - path of the corpus that contains the word
# filename_datapoints_words - text file that will contain the data points of all the words from the corpus,
# format: x1   y1   word1
#         x2   y2   word2
# filename_datapoints_specific_word - text file that will contain the data points of only the word_to_plot (same format as before)
# reduced - embedding vectors reduced to 2 dimensions
def extract_reduced_dim_single_word(word_to_extract, path_corpus, filename_datapoints_words,
                                    filename_datapoints_specific_word, reduced):
    all_tokens = []  # list of the words of the specified corpus
    with open(path_corpus, 'r', encoding="utf8") as f:
        for line in f:
            for word in line.split():
                all_tokens.append(word)

    # creates a text file containing the two dimensional data point of each word:
    # x1   y1   word1
    # x2   y2   word2
    # ...
    count = 0
    with open(filename_datapoints_words, "w+", encoding="utf8") as f:
        for i in reduced:
            string = str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\t" + all_tokens[
                count]  # CHANGE: ADDED str(i[2]) as we have 3 dim now
            f.write(string + '\n')
            count += 1

    # plot only a single word, e.g. "walk"
    indices = [i for i, x in enumerate(all_tokens) if x == word_to_extract]

    datapoints = []
    with open(filename_datapoints_specific_word, "w+", encoding="utf8") as f_word:
        for i in indices:
            dp = [reduced[i][0], reduced[i][1]]
            datapoints.append(dp)
    word_array = np.array(datapoints)

    return word_array


# args: word_to_extract - string of the word that should be extracted
# path_corpus - path of the corpus that contains the word
# unreduced - embedding vectors
def extract_unreduced_dim_single_word(word_to_extract, path_corpus, unreduced):
    all_tokens = []  # list of the words of the specified corpus
    with open(path_corpus, 'r', encoding="utf8") as f:
        for line in f:
            for word in line.split():
                all_tokens.append(word)

    # get only a single word, e.g. "walk"
    indices = [i for i, x in enumerate(all_tokens) if x == word_to_extract]
    datapoints = []
    for i in indices:
        dp = [unreduced[i], unreduced[i]]
        datapoints.append(dp)
    word_array = np.array(datapoints)

    # print(word_array)
    print("word array's shape of single word: ", word_array.shape)

    return word_array


# function to find a distance between a point and a line in a 2d-space (helper function for get_k())
def calc_distance(x1, y1, a, b, c):
    d = abs(a * x1 + b * y1 + c) / math.sqrt(a * a + b * b)
    return d


# Determine the optimal number of k for a single word. Creates an elbow plot, and automatically determines the
# bend in the plot.
# The idea comes from Youtube Video of Bhavesh Bhatt "Finding K in K-means Clustering Automatically", see:
# https://www.youtube.com/watch?v=IEBsrUQ4eMc
# args: word - single word that will be clustered
# emb_single_word - the reduced embedding (shape: (number of occurrences of the word, 2))
# range_upper_bound - the maximum k that will be clustered for (it will always start with k=1, k=2, ..., k=range_upper_bound - 1)
def get_k(word, emb_single_word, range_upper_bound):
    dist_points_from_cluster_center = []

    # reshape the array into 2d array (which is needed to fit the model)
    nsamples, _ = emb_single_word.shape
    #d2_emb_single_word = emb_single_word.reshape((nsamples, nx * ny))

    # in case there are less occurrences of the word in the corpus, e.g. Gott appears only 3 times in GER2 corpus:
    # the range where we can cluster can be maximally 3 clusters then.
    # So check if range_upper_bound is bigger than nsamples. If yes: set range_upper_bound to nsamples
    if nsamples == 1:
        return 1

    if (range_upper_bound > nsamples):
        range_upper_bound = nsamples + 1
    # create the actual range
    K = range(1, range_upper_bound)

    print("looking for the best k...")

    for no_of_clusters in K:
        k_model = KMeans(n_clusters=no_of_clusters)
        k_model.fit(emb_single_word)
        dist_points_from_cluster_center.append(k_model.inertia_)

    # it is possible to only have one cluster, so we need a value for 0 as well. Default is to double the value for k=1
    distance_at_zero = dist_points_from_cluster_center[0] * 1.3  # get k=1 distance and multiply it with the given value
    dist_points_from_cluster_center.insert(0, distance_at_zero)  # add it at the beginning of the list created above
    K = range(0, range_upper_bound)  # adjust range K

    # plot the elbow graph
    # plt.plot(K, dist_points_from_cluster_center)
    # plt.savefig("elbow-plot-for-" + word + ".png")
    # plt.clf()

    # draw a line so the elbow line will get a "hypotenuse" and calculate the distance from each elbow-point to the hypotenuse
    # --> where the longest distance is -> this is the optimal k
    # (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0
    # https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
    a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[range_upper_bound - 1]
    b = K[range_upper_bound - 1] - K[0]
    c1 = K[0] * dist_points_from_cluster_center[range_upper_bound - 1]
    c2 = K[range_upper_bound - 1] * dist_points_from_cluster_center[0]
    c = c1 - c2
    distance_of_points_from_line = []
    for k in range(0, range_upper_bound):
        distance_of_points_from_line.append(calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))

    # plot the three lines: elbow, distance_of_points_from_line, and hypotenuse
    plt.plot(K, dist_points_from_cluster_center)
    plt.plot(K, distance_of_points_from_line)
    plt.plot([K[0], K[range_upper_bound - 1]], [dist_points_from_cluster_center[0],
                                                dist_points_from_cluster_center[range_upper_bound - 1]], 'ro-')
    plt.savefig("./out/max-dist-from-line-at-k-for-" + word + ".png")
    plt.clf()

    return distance_of_points_from_line.index(max(distance_of_points_from_line))


# create a set that contains all unique words from corpus1 and corpus2 of a language
def get_unique_words_per_language(corpus_combined):
    unique_words = set()
    with open(corpus_combined, 'r', encoding='utf8') as f:
        sentences = [elem for elem in f.read().split('\n') if elem]
        for sentence in sentences:
            for w in sentence.split():
                unique_words.add(w)

    return unique_words


# creates a dict that contains each unique words as keys and a list of all sentence indexes where the word appears as value
def create_word_index_dict(combined_corpus_path, unique_word_set):
    unique_word_list = list(unique_word_set)
    word_indexes = dict()
    # create a dictionary with the format: {unique_word1: [], unique_word1: [], ...}
    for unique_word in unique_word_list:
        word_indexes[str(unique_word)] = list()

    with open(combined_corpus_path, 'r', encoding='utf8') as f:
        sentences = [elem for elem in f.read().split('\n') if elem]
        sentence_count = 0
        for sentence in sentences:
            for word in sentence.split():
                word_indexes[word].append(sentence_count)
            sentence_count += 1

    return word_indexes


"""
    Determines the optimal amount of clusters for each word and clusters accordingly.
    Saves the elbow plots and the labels for each word and each corpus in the directory ./out (automatically created)
    Args:
        e:          embeddings
        language:   one of those -> EN, GER, LAT, SWE (language abbreviation)
        path_corp: path of the historic corpus of that language
        word:       the individual word that is investigated
"""
def process_corpus(e, language, id, path_corp, word):
    # the list of lists which store the sentences (enter path on tesniere server)

    # in case it is latin the # symbols have to be removed from the corpus
    if (language == "LAT"):
        preprocess_sentences_corp = preprocess(path_corp)
        sentences_corp = clean_corpus(preprocess_sentences_corp)
    else:
        sentences_corp = preprocess(path_corp)

    # retrieve the indexes of the corpus where the word appears (sentence + word position)
    word_ind_corp = collect_all_occurrences(sentences_corp)

    # list with only sentences that contain the target word
    sentences_with_word = [sentences_corp[tup[0]] for tup in word_ind_corp[word]]

    embs_corp = e.sents2elmo(sentences_with_word)  # will return a list of numpy arrays, each with the shape=(seq_len, embedding_size)
    print("embs_corp length: ", len(embs_corp))
    print("element shapes: ", embs_corp[0].shape)

    # embed_word is a list of one-dimensional arrays: (occurrences of the word, 1024)
    embed_word = [embs_corp[i][tup[1]] for i, tup in enumerate(word_ind_corp[word])]
    print("embed_word length: ", len(embed_word))
    print("element shapes: ", embed_word[0].shape)

    # convert embed_word to numpy array
    embed_word_array = np.asarray(embed_word)
    print("type of embed_word_array: ", type(embed_word_array))
    print("shape: ", embed_word_array.shape)


    # cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_corp_word = get_k(word + "_" + language + id, embed_word_array, 11)
    print("optimal k for " + word + " in " + language + id +": ", optimal_k_corp_word)

    # reshape the array into 2d array (which is needed to fit the model)
    #nsamples, nx, ny = embed_word_array.shape
    #d2_embed_word_array = embed_word_array.reshape((nsamples, nx * ny))

    # cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans_corp_word = KMeans(n_clusters=optimal_k_corp_word).fit(embed_word_array)
    labels_corp_word = list(
        kmeans_corp_word.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_corp_word = set(labels_corp_word)

    # save the labels into a text file
    with open("./out/labels_" + word + "_" + language + id + ".txt", "w+", encoding="utf8") as f:
        f.write(str(labels_corp_word))

    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_corp_word = dict()
    for l in labels_corp_word:
        if l in clusters_corp_word:
            clusters_corp_word[l] += 1
        else:
            clusters_corp_word[l] = 1
    print("Cluster dictionary " + language + id + " " + word + ":")
    print("length: ", len(clusters_corp_word))
    print(clusters_corp_word)

    return labels_corp_word, set_labels_corp_word


def changed_sense(all_clusters, cor1_clusters, cor2_clusters, k):
    n_occur = 0
    if len(all_clusters) == 1:
        print("No change as combined corpora have only 1 cluster in total")
        return False

    for label in all_clusters:
        # if word appears in both corpora, ignore it
        if label in cor1_clusters and label in cor2_clusters:
            continue
        # if the word lost a sense (in C1 but not in C2)
        if label in cor1_clusters and label not in cor2_clusters:
            print("Label ", label, "is in cor1", cor1_clusters.get(label))
            n_occur = cor1_clusters.get(label)
        # if the word gained a sense (not in C1 but in C2)
        elif label in cor2_clusters and label not in cor1_clusters:
            print("Label ", label, "is in cor2", cor2_clusters.get(label))
            n_occur = cor2_clusters.get(label)
        if n_occur >= k:
            print("Label ", label, "occurs ", n_occur, " times.", " k is ", k)
            return True
    return False

if __name__ == '__main__':
    # following this tutorial for the pre-trained embeddings: https://github.com/HIT-SCIR/ELMoForManyLangs

    # ## ENGLISH #######################################################################################################
    # arg 1: the absolute path from the repo top dir to you model dir (path on tesniere server)
    # arg 2: default batch_size: 64
    e_GER = Embedder('../models/142-german-model', batch_size=64)

    corpus_old = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt')
    corpus_new = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt')
    joined_corpus = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt')

    target_words_GER = ["Gott", "und", "haben", "ändern"]

    results = dict()

    for word in target_words_GER:
        print("\nSTARTING THE ANALYSIS FOR TARGET WORD: " + word + " ===============================================================")
        # args: embedding, language abbreviation, corpus path, individual target word
        labels1, set_labels1 = process_corpus(e_GER, "GER", "1",
                                '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt',
                                word)
        labels2, set_labels2 = process_corpus(e_GER, "GER", "2",
                              '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt',
                              word)
        labels_comb, set_labels_comb = process_corpus(e_GER, "GER", "combined",
               '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt',
                                 word)

        # 5.Create a dictionary of indices of each word in the corpus
        ind_joined = collect_all_occurrences(joined_corpus)  # returns a dictionary with ALL WORDS and their indices
        #print("Printing the indices at which ", word, " occurs...")
        #for x in ind_joined[word]:
        #    print(x)

        # 6.Get the sentences that contain the word we want
        sentences_with_word = [joined_corpus[tup[0]] for tup in
                               ind_joined[word]]  # tup is (0,4) where 0 is the sent ind and 4 is the index of the word

        # 11.1 Determine where sentences from corpus2 start in sentences_with_word
        #tmp = [(1, 2), (2, 3), (3, 3), (4, 3)]
        #joined = [["a", "b"], ["a", "b", "f"], ["r"], ["bla"], ["z", "t", "b", "f"]]
        #corpus_old = [["a", "b"], ["a", "b", "f"], ["r"]]
        start_ind = 0
        for idx, t in enumerate(ind_joined[word]):  # ind_joined[word]):
            if t[0] > len(corpus_old) - 1:
                start_ind = idx
                print("Index:", t[0], "length of old corpus:", len(corpus_old), " last index in old corpus:",
                      len(corpus_old) - 1)
                break

        labels_corpus1 = labels_comb[0:start_ind]  # [0, 0, 1, 1, 1, 1, 0, 3, 3]
        print("labels corpus 1: ")
        print(labels_corpus1)
        labels_corpus2 = labels_comb[start_ind:]  # [0, 0, 0, 2, 2, 2, 0, 3, 3, 3]
        print("labels corpus 2: ")
        print(labels_corpus2)
        # TODO: problem: the corpora where joined before collecting the indices of the words, so need to check the indices

        # 12. Use the Counter object to count clusters
        cor1_clusters = Counter(labels_corpus1)
        cor2_clusters = Counter(labels_corpus2)

        # get kmeans.labels_
        # all_clusters = kmeans.labels_
        all_clusters = set_labels_comb  # [0, 1, 2, 3]
        # TODO: CHANGE k
        k = 5
        if changed_sense(all_clusters, cor1_clusters, cor2_clusters, k):
            print("Changed sense")
            results[word] = "changed sense"
        else:
            print("No change in senses detected")
            results[word] = "no changed in senses"

    print("results for GERMAN: ", results)

