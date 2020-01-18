import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torch
from elmoformanylangs import Embedder
import sys
import numpy
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
# filename_datapoints_words - text file that will be created to contain the data points of all the words from the corpus,
# format: x1   y1   word1
#         x2   y2   word2
# filename_datapoints_specific_word - text file that will contain the data points of only the word_to_plot (same format as before)
# reduced - embedding vectors reduced to 2 dimensions
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

    print(word_array)
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
    nsamples, nx, ny = emb_single_word.shape
    d2_emb_single_word = emb_single_word.reshape((nsamples, nx * ny))

    # in case there are less occurrences of the word in the corpus, e.g. Gott appears only 3 times in GER2 corpus:
    # the range where we can cluster can be maximally 3 clusters then.
    # So check if range_upper_bound is bigger than nsamples. If yes: set range_upper_bound to nsamples
    if (range_upper_bound > nsamples):
        range_upper_bound = nsamples
    # create the actual range
    K = range(1, range_upper_bound)

    print("looking for the best k...")

    for no_of_clusters in K:
        k_model = KMeans(n_clusters=no_of_clusters)
        k_model.fit(d2_emb_single_word)
        dist_points_from_cluster_center.append(k_model.inertia_)

    # it is possible to only have one cluster, so we need a value for 0 as well. Default is to double the value for k=1
    distance_at_zero = dist_points_from_cluster_center[0] * 2  # get k=1 distance and double it
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
        path_corp1: path of the historic corpus of that language
        word:       the individual word that is investigated
"""


def process_historic_corpus(e, language, path_corp1, word):
    # ## EN1 (HISTORIC CORPUS) ----------------------------------------------------------------------------------------
    # the list of lists which store the sentences (enter path on tesniere server)
    sentences_corp1 = preprocess(path_corp1)
    embs_corp1 = e.sents2elmo(
        sentences_corp1)  # will return a list of numpy arrays, each with the shape=(seq_len, embedding_size)
    # shape: (number of sentences in corpus, number of words in sentence, 1024)
    embs_array_corp1 = numpy.concatenate(embs_corp1, axis=0)

    unreduced_dim_word_corp1 = extract_unreduced_dim_single_word(word, path_corp1, embs_array_corp1)

    # cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_corp1_word = get_k(word + "_" + language + "1", unreduced_dim_word_corp1, 11)
    print("optimal k for " + word + " in " + language + "1: ", optimal_k_corp1_word)

    # reshape the array into 2d array (which is needed to fit the model)
    nsamples, nx, ny = unreduced_dim_word_corp1.shape
    d2_unreduced_dim_word_corp1 = unreduced_dim_word_corp1.reshape((nsamples, nx * ny))

    # cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans_corp1_word = KMeans(n_clusters=optimal_k_corp1_word).fit(d2_unreduced_dim_word_corp1)
    labels_corp1_word = list(
        kmeans_corp1_word.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    # set_labels_corp_word = set(labels_corp1_word)

    # save the labels into a text file
    with open("./out/labels_" + word + "_" + language + "1.txt", "w+", encoding="utf8") as f:
        f.write(str(labels_corp1_word))

    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_corp1_word = dict()
    for l in labels_corp1_word:
        if l in clusters_corp1_word:
            clusters_corp1_word[l] += 1
        else:
            clusters_corp1_word[l] = 1
    print("Cluster dictionary " + language + "1 " + " " + word + ":")
    print("length: ", len(clusters_corp1_word))
    print(clusters_corp1_word)


"""
    Determines the optimal amount of clusters for each word and clusters accordingly.
    Saves the elbow plots and the labels for each word and each corpus in the directory ./out (automatically created)
    Args:
        e:          embeddings
        language:   one of those -> EN, GER, LAT, SWE (language abbreviation)
        path_corp1: path of the modern corpus of that language
        word:       the individual word that is investigated
"""


def process_modern_corpus(e, language, path_corp2, word):
    # ## EN2 (MODERN CORPUS) ----------------------------------------------------------------------------------------
    # the list of lists which store the sentences (enter path on tesniere server)
    sentences_corp2 = preprocess(path_corp2)
    embs_corp2 = e.sents2elmo(
        sentences_corp2)  # will return a list of numpy arrays, each with the shape=(seq_len, embedding_size)
    # shape: (number of sentences in corpus, number of words in sentence, 1024)
    embs_array_corp2 = numpy.concatenate(embs_corp2, axis=0)

    unreduced_dim_word_corp2 = extract_unreduced_dim_single_word(word, path_corp2, embs_array_corp2)

    # cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_corp2_word = get_k(word + "_" + language + "2", unreduced_dim_word_corp2, 11)
    print("optimal k for " + word + " in " + language + "2: ", optimal_k_corp2_word)

    # reshape the array into 2d array (which is needed to fit the model)
    nsamples, nx, ny = unreduced_dim_word_corp2.shape
    d2_unreduced_dim_word_corp2 = unreduced_dim_word_corp2.reshape((nsamples, nx * ny))

    # cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans_corp2_word = KMeans(n_clusters=optimal_k_corp2_word).fit(d2_unreduced_dim_word_corp2)
    labels_corp2_word = list(
        kmeans_corp2_word.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    # set_labels_corp_word = set(labels_corp2_word)

    # save the labels into a text file
    with open("./out/labels_" + word + "_" + language + "2.txt", "w+", encoding="utf8") as f:
        f.write(str(labels_corp2_word))

    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_corp2_word = dict()
    for l in labels_corp2_word:
        if l in clusters_corp2_word:
            clusters_corp2_word[l] += 1
        else:
            clusters_corp2_word[l] = 1
    print("Cluster dictionary " + language + "2 " + " " + word + ":")
    print("length: ", len(clusters_corp2_word))
    print(clusters_corp2_word)


"""
    Determines the optimal amount of clusters for each word and clusters accordingly.
    Saves the elbow plots and the labels for each word and each corpus in the directory ./out (automatically created)
    Args:
        e:          embeddings
        language:   one of those -> EN, GER, LAT, SWE (language abbreviation)
        path_corp1: path of the combined corpus of that language (single txt-file containing both corpora, 1 appended to 2)
        word:       the individual word that is investigated
"""


def process_combined_corpora(e, language, path_corp_combined, word):
    # ## EN1+2 (COMBINED CORPUS) ----------------------------------------------------------------------------------------
    # the list of lists which store the sentences (enter path on tesniere server)
    sentences_corp_combined = preprocess(path_corp_combined)
    embs_corp_combined = e.sents2elmo(
        sentences_corp_combined)  # will return a list of numpy arrays, each with the shape=(seq_len, embedding_size)
    # shape: (number of sentences in corpus, number of words in sentence, 1024)
    embs_array_corp_combined = numpy.concatenate(embs_corp_combined, axis=0)

    unreduced_dim_word_corp_combined = extract_unreduced_dim_single_word(word, path_corp_combined,
                                                                         embs_array_corp_combined)

    # cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_corp_combined_word = get_k(word + "_" + language + "_combined", unreduced_dim_word_corp_combined, 11)
    print("optimal k for " + word + " in " + language + "_combined: ", optimal_k_corp_combined_word)

    # reshape the array into 2d array (which is needed to fit the model)
    nsamples, nx, ny = unreduced_dim_word_corp_combined.shape
    d2_unreduced_dim_word_corp_combined = unreduced_dim_word_corp_combined.reshape((nsamples, nx * ny))

    # cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans_corp_combined_word = KMeans(n_clusters=optimal_k_corp_combined_word).fit(d2_unreduced_dim_word_corp_combined)
    labels_corp_combined_word = list(
        kmeans_corp_combined_word.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    # set_labels_corp_word = set(labels_corp_combined_word)

    # save the labels into a text file
    with open("./out/labels_" + word + "_" + language + "_combined.txt", "w+", encoding="utf8") as f:
        f.write(str(labels_corp_combined_word))

    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_corp_combined_word = dict()
    for l in labels_corp_combined_word:
        if l in clusters_corp_combined_word:
            clusters_corp_combined_word[l] += 1
        else:
            clusters_corp_combined_word[l] = 1
    print("Cluster dictionary " + language + "_combined " + " " + word + ":")
    print("length: ", len(clusters_corp_combined_word))
    print(clusters_corp_combined_word)


if __name__ == '__main__':
    # following this tutorial for the pre-trained embeddings: https://github.com/HIT-SCIR/ELMoForManyLangs

    # ## ENGLISH #######################################################################################################
    # arg 1: the absolute path from the repo top dir to you model dir (path on tesniere server)
    # arg 2: default batch_size: 64
    e_EN = Embedder('/home/pia/train_elmo/SemEval2020/144-english-model', batch_size=64)

    target_words_EN = ["walk", "distance", "small", "god"]
    for w in target_words_EN:
        # args: embedding, language abbreviation, corpus path, individual target word
        process_historic_corpus(e_EN, "EN",
                                '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus1/corpus1.txt',
                                w)
        process_modern_corpus(e_EN, "EN",
                              '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus2/corpus2.txt',
                              w)
        process_combined_corpora(e_EN, "EN",
                                 '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus1+2/corpus_EN_combined.txt',
                                 w)


    # ## GERMAN ########################################################################################################
    # arg 1: the absolute path from the repo top dir to you model dir (path on tesniere server)
    # arg 2: default batch_size: 64
    e_DE = Embedder('/home/pia/train_elmo/SemEval2020/142-german-model', batch_size=64)

    target_words_GER = ["Gott", "und", "haben", "ändern"]
    for w in target_words_GER:
        # args: embedding, language abbreviation, corpus path, individual target word
        process_historic_corpus(e_GER, "GER",
                                '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt',
                                w)
        process_modern_corpus(e_GER, "GER",
                              '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt',
                              w)
        process_combined_corpora(e_GER, "GER",
                                 '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_EN_combined.txt',
                                 w)