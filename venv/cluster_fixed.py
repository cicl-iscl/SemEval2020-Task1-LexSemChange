import sys
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import gzip
from sklearn.metrics import accuracy_score
import tensorflow as tf
import torch
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from elmoformanylangs import Embedder
from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.cluster import KMeans
from silhouette_score import get_silhouette_scores, get_silhouette_score_with_plot
np.set_printoptions(threshold=sys.maxsize)
import tensorflow_hub as hub
import tensorflow as tf
from metrics import calculate_jsd
from sklearn.cluster import DBSCAN
from plotting import plot_word_2dim,plot_word_3dim
DIVIDER = "---------------------------------------------"
tf.config.optimizer.set_jit(True)

def load_corpus(filename):
    """
        Loads a corpus into a list.

        :param filename: the name of the file containing the corpus
        :returns tokenized: a list of sentences, where each sentence is a list of words [["I", "eat", "apples"], ["she", "sleeps"]]
    """
    print(filename)
    tokenized = []
    """if '.gz' in filename:
        with gzip.open(filename, 'r') as f:
            for line in f:
                tokenized.append(line.strip().split())
    else:"""
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
    tokenized = [line.strip().split() for line in content]
    return tokenized

    """
def load_model(lang):
    model_path = '../models/{}-model'.format(lang)

    if lang == "english":
        model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    else:
        model = Embedder(model_path, batch_size=64)

    return model
    """


def load_model(lang, trained_on_own_corpora=True):
    if trained_on_own_corpora:  # IF THE MODEL SHOULD BE THE ONE WE TRAINED
        options_file = "../final_models_trained/{}_model/options.json".format(lang)
        weight_file = "../final_models_trained/{}_model/weights.hdf5".format(lang)  # sys.argv[1]
        model = Elmo(options_file, weight_file, 2, dropout=0)
    else:
        model_path = '../models/{}-model'.format(lang)
        if lang == "english":
            model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        else:
            model = Embedder(model_path, batch_size=64)

    return model


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
    print("_______________")
    #print(corpus[0])
    print("_______________")
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
    #print(indices)
    return indices


# CHANGED!!! created this function
# these eps and min_sample hyperparameters seem to be a good choice
# when there are many data points (word occurs often in corpus) it makes sense to set min_samples much higher
# when there are only very few, set it to a low number (maybe 2 or 3?)
# eps is difficult to set. Between 3 and 4.5 seems to be a reasonable choice,
# it looks like the more datapoints the lower eps has to be
def cluster_DBSCAN(language, corpus_id, word, embeddings, eps, min_samples, save_to_file=False):

    """
        Clustering using sklearn's DBSCAN
        -> first 3 arguments are only for naming the file that is saved if save_to_file=True
        :param language: german, latin, english or swedish
        :param corpus_id: "1" = historic corpus, "2" = modern corpus, "combinded" = combined corpus
        :param word: target word
        :param embeddings: target word embedding(size: 16, 1024)
        :param save_to_file: save return value to file (default is False)
        -> PARAMETERS TO TUNE:
        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                    This is not a maximum bound on the distances of points within a cluster.
                    This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as
                            a core point. This includes the point itself.
        :return: labels_corp_word: the resulting labels from the clustering
    """
    # Cluster and get the labels. Count the amount of each label
    # as later on we need to know how big is a cluster to correctly classify as lex. sem. change or not
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)

    # clustering.labels_ returns an array like: array([1, 1, 1, 0, 0, 0], dtype=int32) that's converted to a list
    labels_corp_word = list(clustering.labels_)

    # 3. Save the labels into a text file
    if save_to_file:
        with open("./out/DBSCAN-labels_" + word + "_" + language + "_" + corpus_id + ".txt", "w+", encoding="utf8") as f:
            f.write(str(labels_corp_word).replace("[", "").replace("]", ""))

    return labels_corp_word

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



def get_k(word, emb_single_word, range_upper_bound, kmk, cluster_words = False):
    """
        Determine the optimal number of k for a single word.
        Creates an elbow plot, and automatically determines the bend in the plot.
        The idea comes from Youtube Video of Bhavesh Bhatt "Finding K in K-means Clustering Automatically",
        see: https://www.youtube.com/watch?v=IEBsrUQ4eMc

        :param word: target word
        :param emb_single_word: embeddings of target word
        :param range_upper_bound: the maximum k that will be clustered for
                                  (it will always start with k=1, k=2, ..., k=range_upper_bound - 1)
        :return: best_k: the optimal value for k
    """
    dist_points_from_cluster_center = []

    # reshape the array into 2d array (which is needed to fit the model)
    if cluster_words:
        nsamples, _ = emb_single_word.shape
        print("n_samples if word embeddings, not context embeddings", nsamples)
    else:
        nsamples = len(emb_single_word)
        print("nsamples in get_k for context vectors:", len(emb_single_word))
    #d2_emb_single_word = emb_single_word.reshape((nsamples, nx * ny))

    # in case there are less occurrences of the word in the corpus, e.g. Gott appears only 3 times in GER2 corpus:
    # the range where we can cluster can be maximally 3 clusters then.
    # So check if range_upper_bound is bigger than nsamples. If yes: set range_upper_bound to nsamples
    if nsamples == 1:
        return 1

    if (range_upper_bound > nsamples):
        range_upper_bound = nsamples + 1

    # create the actual range
    K = range(1, range_upper_bound) # TODO : adjust range K (from 2)

    print("looking for the best k...")

    for no_of_clusters in K:
        k_model = KMeans(n_clusters=no_of_clusters)
        k_model.fit(emb_single_word)
        dist_points_from_cluster_center.append(k_model.inertia_)
    # TODO: OUTCOMMENT TO GET K = 2
    # it is possible to only have one cluster, so we need a value for 0 as well. Default is to double the value for k=1
    print(type(1.2))
    print(type(kmk))
    distance_at_zero = dist_points_from_cluster_center[0] * kmk # get k=1 distance and multiply by the value given
    dist_points_from_cluster_center.insert(0, distance_at_zero)  # add it at the beginning of the list created above
    K = range(0, range_upper_bound)  # TODO : adjust range K (from 2)

    # plot the elbow graph
    # plt.plot(K, dist_points_from_cluster_center)
    # plt.savefig("elbow-plot-for-" + word + ".png")
    # plt.clf()

    # draw a line so the elbow line will get a "hypotenuse" and calculate the distance from each elbow-point to the hypotenuse
    # --> where the longest distance is -> this is the optimal k
    # https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
    a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[range_upper_bound - 1]
    b = K[range_upper_bound - 1] - K[0]
    c1 = K[0] * dist_points_from_cluster_center[range_upper_bound - 1]
    c2 = K[range_upper_bound - 1] * dist_points_from_cluster_center[0]
    c = c1 - c2
    distance_of_points_from_line = []
    for k in range(0, range_upper_bound): # TODO : adjust range K (from 2)
        distance_of_points_from_line.append(calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))

    # plot the three lines: elbow, distance_of_points_from_line, and hypotenuse
    plt.plot(K, dist_points_from_cluster_center)
    plt.plot(K, distance_of_points_from_line)
    plt.plot([K[0], K[range_upper_bound - 1]], [dist_points_from_cluster_center[0],
                                                dist_points_from_cluster_center[range_upper_bound - 1]], 'ro-')
    plt.savefig("./out/max-dist-from-line-at-k-for-" + word + ".png")
    plt.clf()

    best_k = distance_of_points_from_line.index(max(distance_of_points_from_line))

    return best_k



"""
    Determines the optimal amount of clusters for each word and clusters accordingly.
    Saves the elbow plots and the labels for each word and each corpus in the directory ./out (automatically created)
    Args:
        e:          embeddings
        language:   one of those -> EN, GER, LAT, SWE (language abbreviation)
        path_corp: path of the historic corpus of that language
        word:       the individual word that is investigated
"""

def cluster(language, corpus_id, word, embeddings, kmk, save_to_file = False,  cluster_words = False):

    # 1. Cluster the data points of the single word of one corpus and get the optimal k
    # get the optimal k from the reduced embeddings, indicate the upper range bound of k (k = 11-1 --> k will be maximally 10)
    optimal_k_corp_word = get_k(word + "_" + language + corpus_id, embeddings, 11, kmk, cluster_words)

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

def get_sentence_embeddings(occurrences_of_all_words, all_sentences, elmo, word, corpus_id, language, trained_on_own_corpora = True):
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
    print(len(sentences_with_word))
    if trained_on_own_corpora:
        char_ids = batch_to_ids(sentences_with_word)
        embeddings = elmo(char_ids)
        sentence_embeddings = embeddings["elmo_representations"][0]
        print("\nOWN MODEL: Sentence_embeddings length of {}: ".format(corpus_id), len(sentence_embeddings))
        print("element 0 shapes: ", sentence_embeddings[0].shape, "\n")

    else:
        # 2. Get the embeddings for all the sentences that contain the target word
        if language == "english":
            sentence_embeddings = elmo(sentences_with_word, signature="default", as_dict=True)["elmo"]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sentence_embeddings = sess.run(sentence_embeddings)

            print("PRINTING SHAPE OF ENGLISH SENTENCE EMBEDDINGS:")
            print(sentence_embeddings.shape)
            print("Printing the type of sentence embeddings in english: ", type(sentence_embeddings))
        else:
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
    print("type of sent_embed: ", type(sentence_embed))
    #word_embeddings = [sentence_embed[i][tup[1]] for i, tup in enumerate(word_indices)]
    word_embeddings = [sentence_embed[i][tup[1]].detach().numpy() for i, tup in enumerate(word_indices)]
    print("word_embeddings length of {}: ".format(corpus_id), len(word_embeddings))
    print("element 0 shapes: ", word_embeddings[0].shape, "\n")
    print("element 0 type", type(word_embeddings[0]))

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
    print("type of sentence_embed:", type(sentence_embed))
    context_embeddings = [np.delete(sentence_embed[i],tup[1], axis=0)for i, tup in enumerate(word_indices)]
    print("context_embeddings length of {}: ".format(corpus_id), len(context_embeddings), type(context_embeddings))
    print("Shape of context embedding element 0 before deleting word:", sentence_embed[0].shape)
    print("element 0 shapes after deleting: ", context_embeddings[0].shape, "\n")

    # 4. Convert word_embeddings to a numpy array
    context_embeddings = np.asarray(context_embeddings)
    #embed_word_array = np.asarray(word_embeddings)
    print("Type of context_embeddings in {} after change to np array: ".format(corpus_id), type(context_embeddings))
    print("Shape: ", context_embeddings.shape, "\n")

    return context_embeddings


def changed_sense(all_clusters, cor1_clusters, cor2_clusters, k, n):
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
    for label in all_clusters:
        #print(all_clusters)
        print(cor1_clusters.items())
        print(cor2_clusters.items())

        if cor1_clusters.get(label) is None:
            n_occur_c1 = 0
        else:
            n_occur_c1 = cor1_clusters.get(label)
        if cor2_clusters.get(label) is None:
            n_occur_c2 = 0
        else:
            n_occur_c2 = cor2_clusters.get(label)

        if n_occur_c1 <= k and n_occur_c2 >= n:
            print("CHANGED!")
            print(label, "\t", "occ in c1: ", n_occur_c1, "occ in c2: ", n_occur_c2)
            return True
        if n_occur_c2 <= k and n_occur_c1 >= n:
            print("CHANGED!")
            print(label, "\t", "occ in c1: ", n_occur_c1, "occ in c2: ", n_occur_c2)
            return True
        print("NO CHANGE!")
        print(label, "\t", "occ in c1: ", n_occur_c1, "occ in c2: ", n_occur_c2)
        continue

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
    print("returning -1")
    return -1

def get_averaged_context_embeddings(context_embed):
    """
    Method that averages the context embeddings for EACH SENTENCE
    :param context_embed: output of get_sentence_embeddings (embeddings for context around the word)
    :return averaged_embeddings: the average of each context of each sentence [np.array([0.25]), np.array([0.9])]
    """
    averaged_embeddings = [] # [np.array([0.25]), np.array([0.9])] where each element is the context of a word
    for sent_context in context_embed:
        #print(sent_context)
        word_array = []
        for word in sent_context:
            #print(sum(word))
            word_array.append(sum(word))
        #print("word array:", word_array)
        #print("length of word array:", len(word_array))
        #print()
        mean = sum(word_array)/len(word_array) # 0.56
        sentence_np_array = np.array([mean]) # np.array([0.56])
        averaged_embeddings.append(sentence_np_array)
    return np.array(averaged_embeddings)


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

def print_optional(corpus_historic, corpus_modern, word, elmo, combined_clusters, language, embed_word = True):
    # OPTIONAL STEP:  DO 7 and 8 FOR ALL CORPORA (BECAUSE WE WANT TO SEE HOW CLUSTERING DIFFERS)
    indices_corpus1 = collect_all_occurrences(corpus_historic)
    indices_corpus2 = collect_all_occurrences(corpus_modern)

    # FOR REFERENCE:
    # get_sent_emb(occurrences_of_all_words, all_sentences, elmo, word, corpus_id):
    # get_word_embeddings(sentence_embed, word_indices, corpus_id):
    # get_context_embeddings(sentence_embed, word_indices, corpus_id, empty_element = False):

    # Get sentence embeddings for both corpora, they will be used to extract either context or word embeddings
    sent_embeddings_corpus1 = get_sentence_embeddings(indices_corpus1, corpus_historic, elmo, word,"corpus1",  language)
    sent_embeddings_corpus2 = get_sentence_embeddings(indices_corpus2, corpus_modern, elmo, word, "corpus2", language)

    if embed_word:
        embeddings_corpus1 = get_word_embeddings(sent_embeddings_corpus1, indices_corpus1[word], "corpus1")
        embeddings_corpus2 = get_word_embeddings(sent_embeddings_corpus2, indices_corpus2[word], "corpus2")
    else:
        embeddings_corpus1 = get_context_embeddings(sent_embeddings_corpus1,indices_corpus1[word], "corpus1")
        embeddings_corpus2 = get_context_embeddings(sent_embeddings_corpus2, indices_corpus2[word], "corpus2")
        embeddings_corpus1 = get_averaged_context_embeddings(embeddings_corpus1)
        embeddings_corpus2  = get_averaged_context_embeddings(embeddings_corpus2)

    # TODO: need to add a method that averages the context embeddings

    print("_______________________________________________________________")
    cluster_with_dbscan = True
    if cluster_with_dbscan:
        labels_historic = cluster_DBSCAN(lang, "corpus1", word, embeddings_corpus1, True)
        labels_modern = cluster_DBSCAN(lang, "corpus2", word, embeddings_corpus2, True)

    else:
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

def load_gold_labels(filename):
    dictionary = {}
    with open(filename, encoding='utf8') as f:
        content = f.readlines()
    for line in content:
        vals = line.split()
        print(line)
        print(vals)
        if len(vals) != 2:
            print("does not contain two elements")
            raise Exception("does not contain two elements")
        dictionary[vals[0]] = vals[1]
    return dictionary

def load_ranking_info(filename):
    """

    :param filename:
    :return: [(word, real ranking), ... , (wordn, real ranking]
    """
    rankings = []
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
        rankings = [tuple(line.strip().split()) for line in content]
    return rankings

def compare_ranks(gold_rankings_filename, jsd_scores):

    gold_ranking = load_ranking_info(gold_rankings_filename)
    gold_ranking = sorted(gold_ranking, key = lambda x: x[1], reverse = True)
    jsd_scores = {k: v for k, v in sorted(jsd_scores.items(), key=lambda item: item[1], reverse = True)}
    print(gold_ranking)
    print(jsd_scores)

if __name__ == '__main__':

    # TODO: USAGE for single word: python3 cluster_fixed.py latin k classify_single_target_word kmeans -w quis
    # TODO: USAGE for MULTIPLE words :  make a targets folder in train_elmo/SemEval2020/venv
    #  KMEANS:
    #  python3 cluster_fixed.py LANGUAGE K N classify_words kmeans -kmk VALUE_OF_KMK -t file_with_targets.txt
    #  DBSCAN:
    #  python3 cluster_fixed.py LANGUAGE K N classify_words dbscan -eps VALUE_OF_EPS - nsamp NUM_OF_SAMPLES -t file_with_targets.txt

    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, help="the language of the corpora")
    parser.add_argument("k", type=int, help="the k used for determining if a word changed. k=0 for Latin and k=2 for other languages.")
    parser.add_argument("n", type=float, help="The threshold for the changed_sense method (n). n=1 for Latin and n=5 for "
                                              "the other languages.")
    parser.add_argument('command',
                      choices=('classify_single_target_word', 'classify_words'))
    parser.add_argument('clustering_algorithm',
                        choices=('dbscan', 'kmeans'))
    parser.add_argument("-kmk","--kmk",  help="the k used in kmeans")
    parser.add_argument("-eps","--eps", help="Epsilon used in dbscan")
    parser.add_argument("-nsamp", "--nsamp", help="Number of samples used in dbscan")
    parser.add_argument("-t", "--target_words", dest = "targets", help="the file to read target words from")
    parser.add_argument("-w", "--word", help=" The word to classify")
    args = parser.parse_args()

    #  Some variables to be used later
    lang = args.language
    k = args.k
    target_file = args.targets
    target_words = []
    cluster_with_dbscan = False
    n = int(args.n)

    # TODO: choose whether we want to cluster context or word embeddings. if contexts -> set cluster_words = False
    cluster_words = True  # TODO: should be reset if do not need to cluster embeddings
    results = dict() # absolute semantic change (binary classificaiton)
    results_jsd = dict() # degree of semantic change: Jensen-Shannon distance

    if args.command == 'classify_words':
        if target_file is None:
            parser.error('The file containing the target words is required if you want to classify words. Please '
                         'use the -t argument')
        else:
            # target_words = load_targets('targets/{}'.format(target_file)) # targets/file.txt
            target_words = load_targets('targets/{}'.format(target_file))
            #target_words = load_targets('../starting_kit_2/test_data_public/{}/{}'.format(lang, target_file))

    else:
        if args.word is None:
            parser.error('Please enter the word that should be classified.')
        else:
            target_words.append(args.word)
    if args.clustering_algorithm == 'dbscan':
        cluster_with_dbscan = True
        if args.eps is None:
            parser.error('The epsilon is required when clustering with DBSCAN. Please use the -eps argument')
        if args.nsamp is None:
            parser.error('The number of samples is required when clustering with DBSCAN. Please use the -nsamp argument')
        eps = float(args.eps)
        n_samples = int(args.nsamp)
    else:
        if args.kmk is None:
            parser.error('The k used in k means is required. Please use the -kmk argument')
        kmk = float(args.kmk)

    # 2. Load the elmo model for this language
    elmo = load_model(lang) # TODO: ADD FALSE in case you want to load pretrained(Elmo for many langs and official EN) models!

    # 3. Load the corpora

    #path_to_corpus = '../starting_kit_2/test_data_public/{}/corpus{}/lemma/corpus{}.txt'
    # these were used for tuning:
    # TODO: uncomment path if you want to use a second file for the same language (also might have to change semcor)
    path_to_corpus = '../starting_kit_1/trial_data_public/corpora/{}2/corpus{}/semcor{}.txt' # English semcor
    #path_to_corpus = '../starting_kit_1/trial_data_public/corpora/{}/corpus{}/corpus{}.txt' # corpora provided by organizers

    corpus_historic = load_corpus(path_to_corpus.format(lang, 1, 1))
    corpus_modern = load_corpus(path_to_corpus.format(lang, 2, 2))

    if lang == "latin":
        corpus_historic = clean_corpus(corpus_historic)
        corpus_modern = clean_corpus(corpus_modern)

    # 4. Concatenate the two corpora
    joined_corpus = corpus_historic + corpus_modern
    print("Printing joined corpus at index 0 ...")
    print(len(joined_corpus[0]))


    # TODO: there's a problem when the 2nd corpus is read. For some reason, it shows that there are fewer sentences in the corpus
    print("The length of the first corpus is ", len(corpus_historic))
    print("The length of the second corpus is ", len(corpus_modern))
    print("This means that the index of the last sentence in the first corpus is ", len(corpus_historic) - 1)
    print("The length of the joined corpus is ", len(joined_corpus))


    # 5. Create a dictionary of indices of each word in the corpus
    indices_joined = collect_all_occurrences(joined_corpus)  # returns a dictionary with ALL WORDS and their indices
    # for x in ind_joined[word]:
    #    print(x)

    # TODO: have to reshape the sentence arrays
    if lang == 'english':
        joined_corpus = [" ".join(sent) for sent in joined_corpus] # flatten list because the English model takes list of sentences
        print("Length of joined corpus for English", len(joined_corpus))
        print(len(joined_corpus[0]), len(joined_corpus[1]))

    # 6. Iterate over the target words
    for word in target_words: #target_words = [""]
        print("Printing the indices at which ", word, " occurs...")
        #for x in indices_joined[word]:
            #print(x)

        print("\nSTARTING THE ANALYSIS FOR TARGET WORD: " + word + "\n" + DIVIDER)

        # 7. Get sentence embeddings for the combined corpus

        sent_embeddings_both = get_sentence_embeddings(indices_joined, joined_corpus, elmo, word, "both_corpora", lang)

        # 8. Determine what to cluster : word embeddings or context embeddings. Get the embeddings
        if cluster_words:
            final_embeddings_both = get_word_embeddings(sent_embeddings_both, indices_joined[word], "both_corpora")
        else:
            final_embeddings_both = get_context_embeddings(sent_embeddings_both, indices_joined[word], "both_corpora")
            final_embeddings_both = get_averaged_context_embeddings(final_embeddings_both) # RETURNS THE AVERAGE!
            print("Length of averaged embeddings:", len(final_embeddings_both),"shape of its first element:", final_embeddings_both[0].shape)

        print(DIVIDER)

        # 9. Cluster the WORD/CONTEXT embeddings and get the labels for the combined corpus
        # CLUSTER WITh DBSCAN OR KMEANS
        if cluster_with_dbscan:
            labels_both = cluster_DBSCAN(lang, "both_corpora", word, final_embeddings_both, eps, n_samples)
        else:
            labels_both = cluster(lang, "both_corpora", word, final_embeddings_both, kmk, True, cluster_words)
        # labels_both = cluster_DBSCAN(lang, "both_corpora", word, final_embeddings_both, False)
        combined_clusters = Counter(labels_both)

        # TODO: this is the optional part (analysis of SEPARATE clustering for corpus 1, corpus2)
        # embed_word = True is if we want to cluster word embeddings, set to False if you want context embeddings
        #print_optional(corpus_historic, corpus_modern, word, elmo, combined_clusters, embed_word=True, lang)
        print("printing combined_clusters")
        print(combined_clusters)
        # 10. Determine where sentences from corpus2 start in sentences_with_target_word
        start_ind = get_start_index(indices_joined[word],len(corpus_historic)) # get start index of corpus2 (helps divide the labels)
        print(start_ind)

        if start_ind != -1:
            labels_corpus1 = labels_both[0:start_ind]  # [0, 0, 1, 1, 1, 1, 0, 3, 3]
            labels_corpus2 = labels_both[start_ind:]  # [0, 0, 0, 2, 2, 2, 0, 3, 3, 3]
        else:
            labels_corpus1 = labels_both  # [0, 0, 1, 1, 1, 1, 0, 3, 3]
            labels_corpus2 = []
        print_labels(labels_corpus1,labels_corpus2, word)

        # 11. Use the Counter object to count clusters in each corpus AFTER partitioning them
        cor1_clusters = Counter(labels_corpus1)
        cor2_clusters = Counter(labels_corpus2)

        print_analysis(cor1_clusters, cor2_clusters, combined_clusters)

        # 12. Based on the clustering results, decide whether a word has changed senses or not

        # TODO: Problem: what if C1: 3 and C2:2 and threshold = 2
       # threshold = 1 # threshold for determining whether a cluster has enough data points
        if changed_sense(combined_clusters.keys(), cor1_clusters, cor2_clusters, k, n):
            results[word] = "1"
        else:
            results[word] = "0"
        print(cor1_clusters)
        print(cor2_clusters)
        jsd = calculate_jsd(cor1_clusters, cor2_clusters)
        results_jsd[word] = round(jsd, 3)

    # 13. Calculate the accuracy score. Need a file with word true_label
    # TODO: name your file with gold labels as show here: e.g. "english_with_change_info.txt"
    true_vals = load_gold_labels("targets/{}_with_change_info_1.txt".format(lang))
    incorrectly_predicted = []
    y_true = []
    y_pred = []
    for i in true_vals.keys():
        #print(results)
        print(i, "true: ", true_vals[i], " prediction: ", results[i])
        y_true.append(true_vals[i])
        y_pred.append(results[i])
        if true_vals[i] != results[i]:
            incorrectly_predicted.append(i)
    with open ("out/incorrect_pred_{}.txt".format(lang), 'w') as f:
        for i in incorrectly_predicted:
            f.write(i)

    print(DIVIDER)


    #print("Results for {}: ".format(lang))
    #for i in results.items():
        #print(i)

    #  python3 cluster_fixed.py LANGUAGE K N classify_words kmeans -kmk VALUE_OF_KMK -t file_with_targets.txt
    #  DBSCAN:
    #  python3 cluster_fixed.py LANGUAGE K N classify_words dbscan -eps VALUE_OF_EPS - nsamp NUM_OF_SAMPLES -t file_with_targets.txt
    out_file_path = "out/tuning_results_{}_word_embeddings_TRAINED_BY_US.txt".format(target_file[:-4])
    #out_file_path = '../starting_kit_2/test_data_public/predictions/predictions_{].txt'.format(target_file[:-4])
    to_file = {}
    to_file['model'] = "trained_by_us"
    to_file["Language"] = lang
    to_file["Target file"] = target_file
    to_file["k"] = k
    to_file["n"] = n
    if cluster_with_dbscan:
        to_file["algorithm"] = "DBSCAN"
        to_file["epsilon"] = eps
        to_file["Number of samples"] = n_samples
    else:
        to_file["algorithm"] = "KMEANS"
        to_file["k value for get_k"] = kmk

    to_file["accuracy"] = accuracy_score(y_true, y_pred)
    to_file["classification_score"] = results
    to_file["jsd_score"] = results_jsd


    print(DIVIDER)
    for k, v in to_file.items():
        print(k,v)
    with open (out_file_path, 'a+') as f:
        f.write("\n")
        for i in to_file.keys():
            line = i + "\t" + str(to_file[i]) + "\n"
            f.write(line)
            ''' alternative version: use to see if 
            if i == "jsd_score":
                jsd_scores = {k: v for k, v in sorted(to_file[i].items(), key=lambda item: item[1], reverse=True)}
                f.write("\nJSD SCORES:")
                for score in jsd_scores:
                    prediction = score + "\t" + str(to_file[i][score]) + "\n"
                    f.write(prediction)
            else:
                line = i + "\t"+ str(to_file[i]) + "\n"
                f.write(line)'''


