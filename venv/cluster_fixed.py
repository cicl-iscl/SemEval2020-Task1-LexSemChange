import sys
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from elmoformanylangs import Embedder
from sklearn.cluster import KMeans
from silhouette_score import get_silhouette_scores, get_silhouette_score_with_plot
np.set_printoptions(threshold=sys.maxsize)
import tensorflow_hub as hub
import tensorflow as tf
torch.cuda.empty_cache()

DIVIDER = "---------------------------------------------"

def load_corpus(filename):
    """
        Loads a corpus into a list.

        :param filename: the name of the file containing the corpus
        :returns tokenized: a list of sentences, where each sentence is a list of words [["I", "eat", "apples"], ["she", "sleeps"]]
    """
    print(filename)
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
    tokenized = [line.strip().split() for line in content]
    return tokenized

def load_targets(filename):
    """
        Loads target words from a file into a list.

        :parameter filename: the name of the file containing the target words (one per line)
        :returns target_words: a list of target words
    """
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
    target_words = [line.strip() for line in content]
    return target_words


def clean_corpus(orig_corpus):
    """
        Removes # from words a corpus.

        :parameter:
            orig_corpus: corpus in the format of list of lists: [['s1_word1', 's1_word2', 's1_word3', 's1_word4'],
                            ['s2_word1', 's2_word2', 's2_word3', 's2_word4', 's2_word5', 's2_word6', 's2_word7']]
        :returns:
            clean_corpus: corpus in which words with hash signs have been removed (dico#2 -> dico)
    """
    clean_corpus =[]
    for line in orig_corpus:
        sent = [token[:token.index("#")] if "#" in token else token for token in line]
        clean_corpus.append(sent)
    return clean_corpus


def collect_all_occurrences(corpus):
    """
        Creates a dictionary of {word : [(sentence_index, word_index)]} pairs.
        Example: {"foo" : [(0,1),(5,12),(9,4)]}

        :param corpus: tokenized corpus as a list
        :return indices: dictionary of word : indices pairs
    """
    indices = {}
    for i, sent in enumerate(corpus):
        for ind, word in enumerate(sent):
            # TODO: there was a problem here because some sentences have this word several times! but this only gets the first index!
            if ind == 0:
                idx = sent.index(word)
            else:
                idx = sent.index(word, ind)

            # if word does not exist, create an entry and add a list of indices as value
            if word not in indices:
                indices[word] = [(i, idx)]
            # if word already exists in the dictionary, add the new indices to the list of indices
            else:
                indices[word].append((i, idx))
            #print("index of current sentence: ", i)
            #print("index of ", word, " in ", sent, " is ", idx)
    return indices


def reduce_dimensions(embeddings_array):
    """
        Reduces the 1024 dimensions (output by ELMo) of an array of embeddings using PCA and t-SNE
        down to 2 so that we can review the outputs from the model.

        :param embeddings_array: an array of embeddings
        :return reduced: reduced embedding space of shape (num_words, 2)
    """
    print("Unreduced array has dimensions: ", embeddings_array.shape)
    pca = PCA(n_components=50)                       # reduce down to 50 dim
    y = pca.fit_transform(embeddings_array)
    reduced = TSNE(n_components=3).fit_transform(y)  # further reduce to 2 dim using t-SNE
    print("Finished reduction")
    # print("reduced emb space: ", reduced)
    print("Reduced space dimensions: ", reduced.shape)

    return reduced



def plot_word(word_to_plot, path_corpus, filename_plot, filename_datapoints_words, filename_datapoints_specific_word,
              reduced):
    """
        Plots the data points for a specific word.

        :param word_to_plot: string of the word that should be plotted
        :param path_corpus: path of the corpus that contains the word
        :param filename_plot: name under which the plot will be saved
        :param filename_datapoints_words: text file that will contain the data points of all the words from the corpus
                                            format: x1   y1   word1
                                                    x2   y2   word2
        :param filename_datapoints_specific_word: text file that will contain the data points of
                                                 only the word_to_plot (same format as before)
        :param reduced: embedding vectors reduced to 2 dimensions

    """
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

    print("Saving the plot...")
    plt.title("Context embeddings for \"" + word_to_plot + "\"")
    plt.xlabel("x_dimension")
    plt.ylabel("y_dimension")
    # save the plot as it's not possible to show the plot when running on the server
    plt.savefig(filename_plot)
    plt.clf()


def extract_reduced_dim_single_word(word_to_extract, path_corpus, filename_datapoints_words,
                                    filename_datapoints_specific_word, reduced):
    """
      Extracts reduced dimensions for a single word.

    :param word_to_extract: string of the word that should be extracted
    :param path_corpus: path of the corpus that contains the word
    :param filename_datapoints_words: text file that will contain the data points of all the words from the corpus,
                                      format: x1   y1   word1
                                              x2   y2   word2
    :param filename_datapoints_specific_word: text file that will contain the data points of only the word_to_plot (same format as before)
    :param reduced: embedding vectors reduced to 2 dimensions
    :return:
    """
    sentences = load_corpus(path_corpus)  # TODO: I changed the code to use load_corpus and list comprehension
    all_tokens = [token for sentence in sentences for token in sentence]

    # creates a text file containing the two dimensional data point of each word:
    # x1   y1   word1
    # x2   y2   word2
    # ...
    count = 0
    with open(filename_datapoints_words, "w+", encoding="utf8") as f:
        for i in reduced:
            string = str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\t" + all_tokens[count]
            # CHANGE: ADDED str(i[2]) as we have 3 dim now
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
# TODO: extract reduced and unreduced are basically the same with one difference: there is a write to file call for filename_datapoints_words


def extract_unreduced_dim_single_word(word_to_extract, path_corpus, unreduced):
    """
        :param word_to_extract: string of the word that should be extracted
        :param path_corpus: path of the corpus that contains the word
        :param unreduced: embedding vectors
        :return word_array: an array with unreduced dimensions (for a single word)
    """
    sentences = load_corpus(path_corpus)  # TODO: I changed the code to use load_corpus and list comprehension
    all_tokens = [token for sentence in sentences for token in sentence]

    # get only a single word, e.g. "walk"
    indices = [i for i, x in enumerate(all_tokens) if x == word_to_extract]
    datapoints = []
    for i in indices:
        dp = [unreduced[i], unreduced[i]]
        datapoints.append(dp)
    word_array = np.array(datapoints)

    # print(word_array)
    print("Word array's shape of single word: ", word_array.shape)

    return word_array



def calc_distance(x1, y1, a, b, c):
    """
        Finds the distance between a point and a line in a Dd-space (helper function for get_k())
        :param x1:
        :param y1:
        :param a:
        :param b:
        :param c:
        :return:
    """
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
    distance_at_zero = dist_points_from_cluster_center[0] * 1.3  # get k=1 distance and multiply by the value given
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



"""
    Determines the optimal amount of clusters for each word and clusters accordingly.
    Saves the elbow plots and the labels for each word and each corpus in the directory ./out (automatically created)
    Args:
        e:          embeddings
        language:   one of those -> EN, GER, LAT, SWE (language abbreviation)
        path_corp: path of the historic corpus of that language
        word:       the individual word that is investigated
"""
def cluster(language, corpus_id, word, embeddings, save_to_file = False):

    # 1. Cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_corp_word = get_k(word + "_" + language + corpus_id, embeddings, 11)
    """
    silhouette_scores = get_silhouette_scores(embeddings)
    max = 0
    optimal_k_corp_word = 0
    for i in silhouette_scores.keys(): # TODO: Change to a more efficient way
        if silhouette_scores[i] > max:
            max = silhouette_scores[i]
            optimal_k_corp_word = i
        print(i, silhouette_scores[i])
        """

    print("The best score is ")
    print("\nOptimal k for " + word + " in " + language + " " + corpus_id + ": ", optimal_k_corp_word, "\n")

    # 2. Cluster again with optimal k and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    kmeans = KMeans(n_clusters=optimal_k_corp_word).fit(embeddings)
    labels_corp_word = list(kmeans.labels_)  # returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    # 3. Save the labels into a text file
    if save_to_file:
        with open("./out/labels_" + word + "_" + language + "_" + corpus_id + ".txt", "w+", encoding="utf8") as f:
            f.write(str(labels_corp_word).replace("[","").replace("]",""))

    return labels_corp_word

def get_sentence_embeddings(occurrences_of_all_words, all_sentences, elmo, word, corpus_id):
    """
        Get Elmo embeddings for a specific word.

        :param occurrences_of_all_words: a dictionary of {word : [(sentence_index, word_index)]} pairs.
        :param all_sentences: the list of sentences
        :param elmo: the Elmo model
        :param word: the target word
        :param corpus_id: the name of the corpus
        :return sentence_embeddings: an array with embeddings for the whole sentence
    """
    # occurrences_of_word = ind_joined -> basically, all words and their indices of occurrence
    # {"work" : [(1,4), (10,3)],"he" : [(0,3), (54,11)]}

    # 1. Get a list of sentences that contain the target word
    sentences_with_word = [all_sentences[tup[0]] for tup in occurrences_of_all_words[word]]

    # 2. Get the embeddings for all the sentences that contain the target word
    sentence_embeddings = elmo.sents2elmo(sentences_with_word)  # list of numpy arrays, each with the shape = (seq_len, embedding_size)
    print("\nSentence_embeddings length of {}: ".format(corpus_id), len(sentence_embeddings))
    print("element 0 shapes: ", sentence_embeddings[0].shape, "\n")

    return sentence_embeddings


def get_word_embeddings(sentence_embed, word_indices, corpus_id):
    """
        Get Elmo embeddings for a specific word.

        :param occurrences_of_all_words: a dictionary of {word : [(sentence_index, word_index)]} pairs.
        :param all_sentences: the list of sentences
        :param elmo: the Elmo model
        :param word: the target word
        :param corpus_id: the name of the corpus
        :return sentence_embeddings: an array with embeddings for the whole sentence
        :return word_embeddings: an array with embeddings just for the target word (occurrences of the word, 1024)
    """
    # 3. Get the individual word_embeddings (word_embeddings  is a list of 1D arrays: (occurrences of the word, 1024))
    word_embeddings = [sentence_embed[i][tup[1]] for i, tup in enumerate(word_indices)]
    print("word_embeddings length of {}: ".format(corpus_id), len(word_embeddings))
    print("element 0 shapes: ", word_embeddings[0].shape, "\n")

    # 4. Convert word_embeddings to a numpy array
    word_embeddings = np.asarray(word_embeddings)
    #embed_word_array = np.asarray(word_embeddings)
    print("Type of word_embeddings in {} after change to np array: ".format(corpus_id), type(word_embeddings))
    print("Shape: ", word_embeddings.shape, "\n")

    return word_embeddings

# TODO: context with element removed or empty string?
def get_context_embeddings(sentence_embed, word_indices, corpus_id):
    """
        Get Elmo embeddings of contexts around a word
        :param sentence_embed: the list of embeddings for sentences (list of numpy arrays)
        :param corpus_id: the name of the corpus
        :return sentence_embeddings: an array with context embeddings for sentence

    """
    # 3. Get the individual word_embeddings (word_embeddings  is a list of 1D arrays: (occurrences of the word, 1024))
    context_embeddings = [sentence_embed[i].pop(tup[1]) for i, tup in enumerate(word_indices)]
    print("context_embeddings length of {}: ".format(corpus_id), len(context_embeddings))
    print("element 0 shapes: ", context_embeddings[0].shape, "\n")

    # 4. Convert word_embeddings to a numpy array
    context_embeddings = np.asarray(context_embeddings)
    #embed_word_array = np.asarray(word_embeddings)
    print("Type of context_embeddings in {} after change to np array: ".format(corpus_id), type(context_embeddings))
    print("Shape: ", context_embeddings.shape, "\n")

    return context_embeddings

# TODO: change this later! Divide into three separate methdos: get_context_embeddings, get_word_embeddings, get_sentence_embeddings
def get_embeddings(occurrences_of_all_words, all_sentences, elmo, word, corpus_id):
    """
        Get Elmo embeddings for a specific word.

        :param occurrences_of_all_words: a dictionary of {word : [(sentence_index, word_index)]} pairs.
        :param all_sentences: the list of sentences
        :param elmo: the Elmo model
        :param word: the target word
        :param corpus_id: the name of the corpus
        :return sentence_embeddings: an array with embeddings for the whole sentence
        :return word_embeddings: an array with embeddings just for the target word (occurrences of the word, 1024)
    """
    # occurrences_of_word = ind_joined -> basically, all words and their indices of occurrence
    # {"work" : [(1,4), (10,3)],"he" : [(0,3), (54,11)]}

    # 1. Get a list of sentences that contain the target word
    sentences_with_word = [all_sentences[tup[0]] for tup in occurrences_of_all_words[word]]

    # 2. Get the embeddings for all the sentences that contain the target word
    sentence_embeddings = elmo.sents2elmo(sentences_with_word)  # list of numpy arrays, each with the shape = (seq_len, embedding_size)
    print("\nSentence_embeddings length of {}: ".format(corpus_id), len(sentence_embeddings))
    print("element 0 shapes: ", sentence_embeddings[0].shape, "\n")

    # 3. Get the individual word_embeddings (word_embeddings  is a list of 1D arrays: (occurrences of the word, 1024))
    word_embeddings = [sentence_embeddings[i][tup[1]] for i, tup in enumerate(occurrences_of_all_words[word])]
    print("word_embeddings length of {}: ".format(corpus_id), len(word_embeddings))
    print("element 0 shapes: ", word_embeddings[0].shape, "\n")

    # 4. Convert word_embeddings to a numpy array
    word_embeddings = np.asarray(word_embeddings)
    #embed_word_array = np.asarray(word_embeddings)
    print("Type of word_embeddings in {} after change to np array: ".format(corpus_id), type(word_embeddings))
    print("Shape: ", word_embeddings.shape, "\n")

    return sentence_embeddings, word_embeddings



def changed_sense(all_clusters, cor1_clusters, cor2_clusters, k, threshold):
    """
        Determines whether there has been a change in a word's sense(s). It does not matter whether the word
        has gained or lost sense. A word is classified as gaining a sense, if the sense is never attested in corpus1,
        but attested at least k times in corpus2 (and the other way around).

        :param all_clusters: a list of unique clusters
        :param cor1_clusters:  a dictionary of {cluster1_label: num_occurrences_in_corpus1}
        :param cor2_clusters:  a dictionary of {cluster2_label: num_occurrences_in_corpus1}
        :param k: the number of times a word has to occur in order to be classified as having changed
        :param threshold: the threshold which determines whether a cluster can be regarded as not having any datapoints
        :return True: if a word has changed senses, False otherwise
    """
    n_occur = 0
    if len(all_clusters) == 1:
        print("No change as combined corpora have only 1 cluster in total")
        return False


    for label in all_clusters:
        # if word appears in both corpora, ignore it
        if label in cor1_clusters and label in cor2_clusters:
            if cor2_clusters.get(label) > threshold and cor1_clusters.get(label) > threshold:
                print("Label ", label, " is above the threshold in both corpora: ", cor1_clusters.get(label))
                continue
            if cor1_clusters.get(label) == cor2_clusters.get(label): # no change if the number of data points in both is the same
                print("Label ", label, " has equal number of datapoints", cor1_clusters.get(label))
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
            return True
    return False

def get_start_index(word_indices, hist_corp_length):
    """
        Get the index at which the second corpus starts (in the combined corpus).

        :param word_indices: all the indices at which the target word occurs (in both corpora)
        :param hist_corp_length: the length of the historic corpus
        :return idx : the idx at which corpus2 starts, -1 if not found
    """
    for idx, t in enumerate(word_indices):  # ind_joined[word]):
        if t[0] > hist_corp_length - 1:
            print("Index:", t[0], "length of historic corpus:", hist_corp_length, " last index in historic corpus:",
                  hist_corp_length - 1)
            return idx
    return -1

def get_averaged_context_embeddings(context_embed):
    """
    Method that averages the context embeddings for EACH SENTENCE
    :param context_embed: output of get_sentence_embeddings (embeddings for context around the word)
    :return averaged_embeddings: the average of each context of each sentence [np.array([0.25]), np.array([0.9])]
    """
    averaged_embeddings = [] # [np.array([0.25]), np.array([0.9])] where each element is the context of a word
    for word_context in context_embed:
        mean = sum(context_embed)/len(word_context) # 0.56
        sentence_np_array = np.array([mean]) # np.array([0.56])
        averaged_embeddings.append(sentence_np_array)
    return averaged_embeddings


def print_analysis(historic_clusters, modern_clusters ,comb_clusters):
    """
        OPTIONAL
        A helper method to print the clustering layouts.
        :param historic_clusters:
        :param modern_clusters:
        :param comb_clusters:
    """
    # Printing the clusters in all three corpora
    print("\nNumber of clusters in corpus 1: ", len(historic_clusters.keys()))
    for c in historic_clusters.keys():
        print(c, historic_clusters.get(c))
    print("\nNumber of clusters in corpus 2: ", len(modern_clusters.keys()))
    for c in modern_clusters.keys():
        print(c, modern_clusters.get(c))
    print("")

    print("\nNumber of clusters in the combined corpus: ", len(comb_clusters.keys()))
    for c in comb_clusters.keys():
        print(c, comb_clusters.get(c))
    print(DIVIDER)

def print_optional(corpus_historic, corpus_modern, word, elmo, combined_clusters, embed_word = True):
    # OPTIONAL STEP:  DO 7 and 8 FOR ALL CORPORA (BECAUSE WE WANT TO SEE HOW CLUSTERING DIFFERS)
    indices_corpus1 = collect_all_occurrences(corpus_historic)
    indices_corpus2 = collect_all_occurrences(corpus_modern)

    # FOR REFERENCE:
    # get_sent_emb(occurrences_of_all_words, all_sentences, elmo, word, corpus_id):
    # get_word_embeddings(sentence_embed, word_indices, corpus_id):
    # get_context_embeddings(sentence_embed, word_indices, corpus_id, empty_element = False):

    # Get sentence embeddings for both corpora, they will be used to extract either context or word embeddings
    sent_embeddings_corpus1 = get_sentence_embeddings(indices_corpus1, corpus_historic, elmo, word,"corpus1")
    sent_embeddings_corpus2 = get_sentence_embeddings(indices_corpus2, corpus_modern, elmo, word, "corpus2")

    if embed_word:
        embeddings_corpus1 = get_word_embeddings(sent_embeddings_corpus1, indices_corpus1[word], "corpus1")
        embeddings_corpus2 = get_word_embeddings(sent_embeddings_corpus2, indices_corpus2[word], "corpus2")
    else:
        embeddings_corpus1 = get_context_embeddings(sent_embeddings_corpus1,indices_corpus1[word], "corpus1")
        embeddings_corpus2 = get_context_embeddings(sent_embeddings_corpus2, indices_corpus2[word], "corpus2")

    # TODO: need to add a method that averages the context embeddings

    print("_______________________________________________________________")
    labels_historic = cluster(lang, "corpus1", word, embeddings_corpus1, True)
    labels_modern = cluster(lang, "corpus2", word, embeddings_corpus2, True)

    # Use the Counter object to count clusters
    historic_corpus_clusters = Counter(labels_historic)
    modern_corpus_clusters = Counter(labels_modern)

    # OPTIONAL: Print analysis for ALL THREE:
    print(DIVIDER)
    print("Analyzing clustering results for {} AMONG THE THREE CORPORA:".format(word))
    print(DIVIDER)
    print_analysis(historic_corpus_clusters, modern_corpus_clusters, combined_clusters)

def print_labels(labels1, labels2, word):
    print("\n" + DIVIDER)
    print("\nAnalyzing clustering results for {} in the COMBINED CORPUS".format(word))
    print(DIVIDER)
    print("Labels corpus 1: \n")
    print(labels1)
    print("Labels corpus 2: \n")
    print(labels2)


if __name__ == '__main__':

    # TODO: USAGE for single word: python cluster_fixed.py latin k classify_single_target_word -w quis
    # TODO: USAGE for MULTIPLE words:  make a targets folder in train_elmo/SemEval2020/venv
    #  python cluster_fixed.py latin k classify_words -t file_with_targets.txt

    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, help="the language of the corpora")
    parser.add_argument("k", type=int, help="the k used for determining if a word changed (given by the SemEval people")
    parser.add_argument("-t", "--target_words", dest = "targets", help="the file to read target words from")
    parser.add_argument("-w", "--word", help=" The word to classify")
    parser.add_argument('command',
                      choices=('classify_single_target_word', 'classify_words'))
    args = parser.parse_args()

    #  Some variables to be used later
    lang = args.language
    k = args.k
    target_file = args.targets
    target_words = []

    # TODO: choose whether we want to cluster context or word embeddings. if contexts -> set cluster_words = False
    cluster_words = True
    results = dict()

    if args.command == 'classify_words':
        if target_file is None:
            parser.error('The file containing the target words is required if you want to classify words. Please '
                         'use the -t argument')
        else:
            target_words = load_targets('targets/{}'.format(target_file)) # targets/file.txt

    else:
        if args.word is None:
            parser.error('Please enter the word that should be classified.')
        else:
            target_words.append(args.word)

    # 2. Load the elmo model for this language
    model_path = '../models/{}-model'.format(lang)
    if lang == "english":
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    else:
        elmo = Embedder(model_path, batch_size=64)

    # 3. Load the corpora
    path_to_corpus = '../starting_kit/trial_data_public/corpora/{}/corpus{}/corpus{}.txt'
    corpus_historic = load_corpus(path_to_corpus.format(lang, 1, 1))
    corpus_modern = load_corpus(path_to_corpus.format(lang, 2, 2))

    if lang == "latin":
        corpus_historic = clean_corpus(corpus_historic)
        corpus_modern = clean_corpus(corpus_modern)

    # 4. Concatenate the two corpora
    joined_corpus = corpus_historic + corpus_modern


    # TODO: there's a problem when the 2nd corpus is read. For some reason, it shows that there are fewer sentences in the corpus
    print("The length of the first corpus is ", len(corpus_historic))
    print("The length of the second corpus is ", len(corpus_modern))
    print("This means that the index of the last sentence in the first corpus is ", len(corpus_historic) - 1)
    print("The length of the joined corpus is ", len(joined_corpus))


    # 5. Create a dictionary of indices of each word in the corpus
    indices_joined = collect_all_occurrences(joined_corpus)  # returns a dictionary with ALL WORDS and their indices
    # for x in ind_joined[word]:
    #    print(x)


    # 6. Iterate over the target words
    for word in target_words: #target_words = [""]
        print("Printing the indices at which ", word, " occurs...")
        for x in indices_joined[word]:
            print(x)

        print("\nSTARTING THE ANALYSIS FOR TARGET WORD: " + word + "\n" + DIVIDER)

        # 7. Get sentence embeddings for the combined corpus
        #sent_embeddings_both, word_embeddings_both = get_embeddings(indices_joined, joined_corpus, elmo, word,
                                                                   # "both_corpora")

        # TODO: use the official English model
        # Extract ELMo features
        #embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
        #embeddings.shape

        sent_embeddings_both = get_sentence_embeddings(indices_joined, joined_corpus,elmo,word, "both_corpora")

        # 8. Determine what to cluster : word embeddings or context embeddings. Get the embeddings
        if cluster_words:
            final_embeddings_both = get_word_embeddings(sent_embeddings_both, indices_joined[word], "both_corpora")
        else:
            final_embeddings_both = get_context_embeddings(sent_embeddings_both, indices_joined[word], "both_corpora")
            final_embeddings_both = get_averaged_context_embeddings(final_embeddings_both) # RETURNS THE AVERAGE!

        print(DIVIDER)

        # 9. Cluster the WORD/CONTEXT embeddings and get the labels for the combined corpus
        labels_both = cluster(lang, "both_corpora", word, final_embeddings_both, True)
        combined_clusters = Counter(labels_both)

        # TODO: this is the optional part (analysis of SEPARATE clustering for corpus 1, corpus2)
        # embed_word = True is if we want to cluster word embeddings, set to False if you want context embeddings
        print_optional(corpus_historic, corpus_modern, word, elmo, combined_clusters, embed_word=True)

        # 10. Determine where sentences from corpus2 start in sentences_with_target_word
        start_ind = get_start_index(indices_joined[word],len(corpus_historic)) # get start index of corpus2 (helps divide the labels)
        labels_corpus1 = labels_both[0:start_ind]  # [0, 0, 1, 1, 1, 1, 0, 3, 3]
        labels_corpus2 = labels_both[start_ind:]  # [0, 0, 0, 2, 2, 2, 0, 3, 3, 3]
        print_labels(labels_corpus1,labels_corpus2, word)

        # 11. Use the Counter object to count clusters in each corpus AFTER partitioning them
        cor1_clusters = Counter(labels_corpus1)
        cor2_clusters = Counter(labels_corpus2)

        print_analysis(cor1_clusters, cor2_clusters, combined_clusters)

        # 12. Based on the clustering results, decide whether a word has changed senses or not

        # TODO: Problem: what if C1: 3 and C2:2 and threshold = 2
        threshold = 1 # threshold for determining whether a cluster has enough data points
        if changed_sense(combined_clusters.keys(), cor1_clusters, cor2_clusters, k, threshold):
            results[word] = "Changed sense(s)"
        else:
            results[word] = "No change in senses"

    print("Results for {}: ".format(lang), results)
    print(DIVIDER)
