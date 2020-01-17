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


# preprocess German historical corpus to get sentences into lists of lists like this:
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
    pca = PCA(n_components=50)  # reduce down to 50 dim
    y = pca.fit_transform(embeddings_array)
    reduced = TSNE(n_components=3).fit_transform(y)  # further reduce to 2 dim using t-SNE
    print("finished reduction")
    #print("reduced emb space: ", reduced)
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
def plot_word(word_to_plot, path_corpus, filename_plot, filename_datapoints_words, filename_datapoints_specific_word, reduced):
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

    # plot only a single word, e.g. "liegen"
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
def extract_reduced_dim_single_word(word_to_extract, path_corpus, filename_datapoints_words, filename_datapoints_specific_word, reduced):
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
            string = str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\t" + all_tokens[count]  # CHANGE: ADDED str(i[2]) as we have 3 dim now
            f.write(string + '\n')
            count += 1

    # plot only a single word, e.g. "liegen"
    indices = [i for i, x in enumerate(all_tokens) if x == word_to_extract]

    datapoints = []
    with open(filename_datapoints_specific_word, "w+", encoding="utf8") as f_word:
        for i in indices:
            dp = [reduced[i][0], reduced[i][1]]
            datapoints.append(dp)
    word_array = np.array(datapoints)

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
# reduced_emb_single_word - the reduced embedding (shape: (number of occurrences of the word, 2))
# range_upper_bound - the maximum k that will be clustered for (it will always start with k=1, k=2, ..., k=range_upper_bound - 1)
def get_k(word, reduced_emb_single_word, range_upper_bound):
    dist_points_from_cluster_center = []
    K = range(1, range_upper_bound)
    for no_of_clusters in K:
        k_model = KMeans(n_clusters=no_of_clusters)
        k_model.fit(reduced_emb_single_word)
        dist_points_from_cluster_center.append(k_model.inertia_)

    # it is possible to only have one cluster, so we need a value for 0 as well. Default is to double the value for k=1
    distance_at_zero = dist_points_from_cluster_center[0] * 2  # get k=1 distance and double it
    dist_points_from_cluster_center.insert(0, distance_at_zero)  # add it at the beginning of the list created above
    K = range(0, 11)  # adjust range K

    # plot the elbow graph
    #plt.plot(K, dist_points_from_cluster_center)
    #plt.savefig("elbow-plot-for-" + word + ".png")
    #plt.clf()

    # draw a line so the elbow line will get a "hypotenuse" and calculate the distance from each elbow-point to the hypotenuse
    # --> where the longest distance is -> this is the optimal k
    # (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0
    # https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
    a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[range_upper_bound-1]
    b = K[range_upper_bound-1] - K[0]
    c1 = K[0] * dist_points_from_cluster_center[range_upper_bound-1]
    c2 = K[range_upper_bound-1] * dist_points_from_cluster_center[0]
    c = c1 - c2
    distance_of_points_from_line = []
    for k in range(0, 11):
        distance_of_points_from_line.append(calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))

    # plot the three lines: elbow, distance_of_points_from_line, and hypotenuse
    plt.plot(K, dist_points_from_cluster_center)
    plt.plot(K, distance_of_points_from_line)
    plt.plot([K[0], K[range_upper_bound-1]], [dist_points_from_cluster_center[0],
                            dist_points_from_cluster_center[range_upper_bound-1]], 'ro-')
    plt.savefig("max-dist-from-line-at-k-for-" + word + ".png")
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


if __name__ == '__main__':
    # following this tutorial with the German language: https://github.com/HIT-SCIR/ELMoForManyLangs

    # ## GERMAN ########################################################################################################
    # arg 1: the absolute path from the repo top dir to you model dir (path on tesniere server)
    # arg 2: default batch_size: 64
    e_GER = Embedder('/home/pia/train_elmo/SemEval2020/142-german-model', batch_size=64)


    # ## GER1 (HISTORIC CORPUS) ----------------------------------------------------------------------------------------
    # the list of lists which store the sentences (enter path on tesniere server)
    sentences_GER1 = preprocess('/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt')
    embs_GER1 = e_GER.sents2elmo(sentences_GER1)  # will return a list of numpy arrays, each with the shape=(seq_len, embedding_size)
                                                  # shape: (number of sentences in corpus, number of words in sentence, 1024)
    embs_array_GER1 = numpy.concatenate(embs_GER1, axis=0)
    # print("type of embs_array: ", type(embs_array_GER1))
    reduced_GER1 = reduce_dimensions(embs_array_GER1)  # reduce the dimensions of the embeddings to 2

    
    # plot the space of a single word and save it as a png
    plot_word("liegen", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt',
              'german-corp1-liegen.png', 'emb_reduced_GER1.txt', 'emb_reduced_GER1_liegen.txt', reduced_GER1)

    # get the reduced emb arrays of a single word
    reduced_dim_liegen_GER1 = extract_reduced_dim_single_word(
        "liegen", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt',
        'emb_reduced_GER1.txt', 'emb_reduced_GER1_liegen.txt', reduced_GER1)
    print("dimensions of reduced array for liegen GER1: ", reduced_dim_liegen_GER1.shape)

    # cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_GER1_liegen = get_k("liegen_GER1", reduced_dim_liegen_GER1, 11)
    print("optimal k for liegen in GER1: ", optimal_k_GER1_liegen)

    # cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans_GER1_liegen = KMeans(n_clusters=optimal_k_GER1_liegen).fit(reduced_dim_liegen_GER1)
    labels_GER1_liegen = list(kmeans_GER1_liegen.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_GER1_liegen = set(labels_GER1_liegen)
    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_GER1_liegen = dict()
    for l in labels_GER1_liegen:
        if l in clusters_GER1_liegen:
            clusters_GER1_liegen[l] += 1
        else:
            clusters_GER1_liegen[l] = 1
    print("Cluster dictionary GER1 liegen:")
    print("length: ", len(clusters_GER1_liegen))
    print(clusters_GER1_liegen)


    # ## GER2 (MODERN CORPUS) ------------------------------------------------------------------------------------------
    sentences_GER2 = preprocess('/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt')
    embs_GER2 = e_GER.sents2elmo(sentences_GER2)
    embs_array_GER2 = numpy.concatenate(embs_GER2, axis=0)
    reduced_GER2 = reduce_dimensions(embs_array_GER2)

    plot_word("liegen", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt',
              'german-corp2-liegen.png', 'emb_reduced_GER2.txt', 'emb_reduced_GER2_liegen.txt', reduced_GER2)

    reduced_dim_liegen_GER2 = extract_reduced_dim_single_word(
        "liegen", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt',
        'emb_reduced_GER2.txt', 'emb_reduced_GER2_liegen.txt', reduced_GER2)
    print("dimensions of reduced array for liegen GER2: ", reduced_dim_liegen_GER2.shape)

    optimal_k_GER2_liegen = get_k("liegen_GER2", reduced_dim_liegen_GER2, 11)
    print("optimal k for liegen in GER2: ", optimal_k_GER2_liegen)

    kmeans_GER2_liegen = KMeans(n_clusters=optimal_k_GER2_liegen).fit(reduced_dim_liegen_GER2)
    labels_GER2_liegen = list(kmeans_GER2_liegen.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_GER2_liegen = set(labels_GER2_liegen)
    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_GER2_liegen = dict()
    for l in labels_GER2_liegen:
        if l in clusters_GER2_liegen:
            clusters_GER2_liegen[l] += 1
        else:
            clusters_GER2_liegen[l] = 1
    print("Cluster dictionary GER2 liegen:")
    print("length: ", len(clusters_GER2_liegen))
    print(clusters_GER2_liegen)


    # ## GER2 + GER2 (BOTH CORPORA COMBINED) ---------------------------------------------------------------------------
    sentences_GER_combined = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt')
    embs_GER_combined = e_GER.sents2elmo(sentences_GER_combined)
    embs_array_GER_combined = numpy.concatenate(embs_GER_combined, axis=0)
    reduced_GER_combined = reduce_dimensions(embs_array_GER_combined)

    reduced_dim_liegen_GER_combined = extract_reduced_dim_single_word(
        "liegen", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt',
        'emb_reduced_GER_combined.txt', 'emb_reduced_GER_combined_liegen.txt', reduced_GER_combined)
    print("dimensions of reduced array for liegen GER_combined: ", reduced_dim_liegen_GER_combined.shape)

    optimal_k_GER_combined_liegen = get_k("liegen_GER_combined", reduced_dim_liegen_GER_combined, 11)
    print("optimal k for liegen in GER_combined: ", optimal_k_GER_combined_liegen)

    kmeans_GER_combined_liegen = KMeans(n_clusters=optimal_k_GER_combined_liegen).fit(reduced_dim_liegen_GER_combined)
    labels_GER_combined_liegen = list(
        kmeans_GER_combined_liegen.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_GER_combined_liegen = set(labels_GER_combined_liegen)
    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_GER_combined_liegen = dict()
    for l in labels_GER_combined_liegen:
        if l in clusters_GER_combined_liegen:
            clusters_GER_combined_liegen[l] += 1
        else:
            clusters_GER_combined_liegen[l] = 1
    print("Cluster dictionary GER_combined liegen:")
    print("length: ", len(clusters_GER_combined_liegen))
    print(clusters_GER_combined_liegen)


    # TO DO: run code again -> now having three dimensions (check if line 119 works properly)
    # THEN check the plots and the number of clusters



    # ALIGNING CLUSTERS
    unique_words_GER = get_unique_words_per_language(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt')
    #print(unique_words)
    word_index_GER_combined = create_word_index_dict(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt',
        unique_words)
    #print("word indexes:")
    #print(word_index_GER_combined)









    # NEXT:
    # - map/align clusters with one another
    # - classify words (not like below)
    # - do task 2 ranking
    # - let it all run on the server using tmux (control terminal can be detached from process on server with this)






    """
    # classify words as change or no change ------NOT LIKE THAT, we need to take k into account (released Feb 19th)-----
    classification_GER = dict()
    if len(clusters_GER1_liegen) == len(clusters_GER2_liegen):
        classification_GER["liegen"] = 0
    else:
        classification_GER["liegen"] = 1
    print("classification for words in German:")
    print(classification_GER)
    """

    """
    # TEST WITH THE WORD D (shortened article) OF GERMAN
    # ## GER1 (HISTORIC CORPUS) ----------------------------------------------------------------------------------------
    # the list of lists which store the sentences (enter path on tesniere server)
    sentences_GER1 = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt')
    embs_GER1 = e_GER.sents2elmo(
        sentences_GER1)  # will return a list of numpy arrays, each with the shape=(seq_len, embedding_size)
    # shape: (number of sentences in corpus, number of words in sentence, 1024)
    embs_array_GER1 = numpy.concatenate(embs_GER1, axis=0)
    # print("type of embs_array: ", type(embs_array_GER1))
    reduced_GER1 = reduce_dimensions(embs_array_GER1)  # reduce the dimensions of the embeddings to 2

    # plot the space of a single word and save it as a png
    # plot_word("d", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt',
    #          'german-corp1-d.png', 'emb_reduced_GER1.txt', 'emb_reduced_GER1_d.txt', reduced_GER1)

    # get the reduced emb arrays of a single word
    reduced_dim_d_GER1 = extract_reduced_dim_single_word(
        "d", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1.txt',
        'emb_reduced_GER1.txt', 'emb_reduced_GER1_d.txt', reduced_GER1)
    print("dimensions of reduced array for d GER1: ", reduced_dim_d_GER1.shape)

    # cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_GER1_d = get_k("d_GER1", reduced_dim_d_GER1, 11)
    print("optimal k for d in GER1: ", optimal_k_GER1_d)

    # cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans_GER1_d = KMeans(n_clusters=optimal_k_GER1_d).fit(reduced_dim_d_GER1)
    labels_GER1_d = list(
        kmeans_GER1_d.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_GER1_d = set(labels_GER1_d)
    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_GER1_d = dict()
    for l in labels_GER1_d:
        if l in clusters_GER1_d:
            clusters_GER1_d[l] += 1
        else:
            clusters_GER1_d[l] = 1
    print("Cluster dictionary GER1 d:")
    print("length: ", len(clusters_GER1_d))
    print(clusters_GER1_d)

    # ## GER2 (MODERN CORPUS) ------------------------------------------------------------------------------------------
    sentences_GER2 = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt')
    embs_GER2 = e_GER.sents2elmo(sentences_GER2)
    embs_array_GER2 = numpy.concatenate(embs_GER2, axis=0)
    reduced_GER2 = reduce_dimensions(embs_array_GER2)

    # plot the space of a single word and save it as a png
    # plot_word("d", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt',
    #          'german-corp2-d.png', 'emb_reduced_GER2.txt', 'emb_reduced_GER2_d.txt', reduced_GER2)

    reduced_dim_d_GER2 = extract_reduced_dim_single_word(
        "d", '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus2/corpus2.txt',
        'emb_reduced_GER2.txt', 'emb_reduced_GER2_d.txt', reduced_GER2)
    print("dimensions of reduced array for d GER2: ", reduced_dim_d_GER2.shape)

    optimal_k_GER2_d = get_k("d_GER2", reduced_dim_d_GER2, 11)
    print("optimal k for d in GER2: ", optimal_k_GER2_d)

    kmeans_GER2_d = KMeans(n_clusters=optimal_k_GER2_d).fit(reduced_dim_d_GER2)
    labels_GER2_d = list(
        kmeans_GER2_d.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_GER2_d = set(labels_GER2_d)
    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_GER2_d = dict()
    for l in labels_GER2_d:
        if l in clusters_GER2_d:
            clusters_GER2_d[l] += 1
        else:
            clusters_GER2_d[l] = 1
    print("Cluster dictionary GER2 d:")
    print("length: ", len(clusters_GER2_d))
    print(clusters_GER2_d)

    # ## GER2 + GER2 (BOTH CORPORA COMBINED) ---------------------------------------------------------------------------
    sentences_GER_combined = preprocess(
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt')
    embs_GER_combined = e_GER.sents2elmo(sentences_GER_combined)
    embs_array_GER_combined = numpy.concatenate(embs_GER_combined, axis=0)
    reduced_GER_combined = reduce_dimensions(embs_array_GER_combined)

    reduced_dim_d_GER_combined = extract_reduced_dim_single_word(
        "d",
        '/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1+2/corpus_GER_combined.txt',
        'emb_reduced_GER_combined.txt', 'emb_reduced_GER_combined_d.txt', reduced_GER_combined)
    print("dimensions of reduced array for d GER_combined: ", reduced_dim_d_GER_combined.shape)

    optimal_k_GER_combined_d = get_k("d_GER_combined", reduced_dim_d_GER_combined, 11)
    print("optimal k for d in GER_combined: ", optimal_k_GER_combined_d)

    kmeans_GER_combined_d = KMeans(n_clusters=optimal_k_GER_combined_d).fit(reduced_dim_d_GER_combined)
    labels_GER_combined_d = list(
        kmeans_GER_combined_d.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    set_labels_GER_combined_d = set(labels_GER_combined_d)
    # create a dict where the clusters and their size is stored, format is clusterLabel: amount, eg. 0: 5, 1: 19, 2: 4
    clusters_GER_combined_d = dict()
    for l in labels_GER_combined_d:
        if l in clusters_GER_combined_d:
            clusters_GER_combined_d[l] += 1
        else:
            clusters_GER_combined_d[l] = 1
    print("Cluster dictionary GER_combined d:")
    print("length: ", len(clusters_GER_combined_d))
    print(clusters_GER_combined_d)
    """