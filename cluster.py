from allennlp.modules.elmo import Elmo, batch_to_ids
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

"""
    SOME PREPROCESSING STEPS FOR THE LATIN CORPUS(JUST FOR NOW) 
    IN CASE WE DONT USE PRE-TRAINED EMBEDDINGS
"""
def load_corpus(filename):
    print(filename)
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
   #print(content[0])
    tokenized = [line.strip().split() for line in content]
    return tokenized

def load_targets(filename):
    with open(filename, encoding="utf8") as f:
        content = f.readlines()
    target_words = [line.strip()for line in content]
    return target_words

def get_all_occurrences_of_word(corpus, word):
    indices = []
    for i, sent in enumerate(corpus):
        if word in sent:
            idx = sent.index(word)
            indices.append((i,idx))
            #print("index of current sentence: ", i)
            #print("index of ", word, " in ", sent, " is ", idx)
    return indices
"""
    Removes # from words 
"""
def clean_corpus(orig_corpus):
    clean_corpus =[]
    for line in orig_corpus:
        sent = [token[:token.index("#")] if "#" in token else token for token in line]
        clean_corpus.append(sent)
        #print(sent)
    return clean_corpus

def get_all_occurrences_of_word(corpus, word):
    indices = []
    for i, sent in enumerate(corpus):
        if word in sent:
            idx = sent.index(word)
            indices.append((i,idx))
            #print("index of current sentence: ", i)
            #print("index of ", word, " in ", sent, " is ", idx)
    return indices

def collect_all_occurrences(corpus):
    indices = {}
    for i, sent in enumerate(corpus):
        for ind, word in enumerate(sent):
            idx = sent.index(word)
            # if word does not exist, create an entry and add a list of indices as value
            if word not in indices:
                indices[word] = [(i,idx)]
            # if word already exists in the dictionary, add the new indices to the list of indices
            else:
                indices[word].append((i,idx))
            #print("index of current sentence: ", i)
            #print("index of ", word, " in ", sent, " is ", idx)
    return indices

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


def cluster(unlab, labeled, labels, k=7):
    """ Exercise 1: Cluster given unlabeled data using k-means,
        and evaluate on the labeled date set.
    Arguments:
        unlab    An Nx2 array, columns are first (f1) and second (f1) formants.
        labeled An Mx2 array where columns are f1 and f2
        labels   A sequence with M labels (vowels)
    """
    model = KMeans(n_clusters=k).fit(unlab)                                         # train the model
    predictions = model.predict(labeled)                                            # predict clusters [0,3,4,5] etc

    return model

def changed_sense(all_clusters,cor1_clusters,cor2_clusters,k):
    n_occur = 0
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
            print("Label ", label, "occurs ",n_occur, " times.", " k is ", k)
            return True
    return False


if __name__ == "__main__":

    lang = sys.argv[0]
    word = sys.argv[1]
    path_to_model = '/home/anna/train_elmo/WORDEMBED/models_pretrained/{}'

    # GET THE PATH TO THE CORRECT MODEL
    if lang == "latin":
        path_to_model.format('162')
    elif lang == "en":
        pass # TODO: add code
    elif lang == "de":
        pass # TODO: add code
    elif lang == "swedish":
        path_to_model.format('173')

    elmo = Embedder(path_to_model)

    # 2. Load the target file
    path_to_targets = "targets/{}".format(lang)
    targets = load_targets(path_to_targets)

    # 3. Load the corpora
    # corpus1 = load_corpus("/WORDEMBED/{}/corpus1/corpus1.txt".format(lang))
    path_old = "{}/corpus1/corpus1.txt".format(lang)
    corpus1 = load_corpus(path_old)
    corpus_old = clean_corpus(corpus1)

    path_new = "{}/corpus2/corpus2.txt".format(lang)
    corpus2 = load_corpus(path_new)
    corpus_new = clean_corpus(corpus2)
    print("The length of the first corpus is ", len(corpus_old))
    print("The length of the second corpus is ", len(corpus_new))
    print("This means that the index of the last sentence in the first corpus is ", len(corpus_old)-1)

    # 4. Merge the two corpora into 1
    joined_corpus = corpus_old + corpus_new

    # 5.Create a dictionary of indices of each word in the corpus
    ind_joined = collect_all_occurrences(joined_corpus) # returns a dictionary with ALL WORDS and their indices
    print("Printing the indices at which ", word, " occurs...")
    for x in ind_joined[word]:
        print(x)

    # 6.Get the sentences that contain the word we want
    sentences_with_word = [joined_corpus[tup[0]] for tup in ind_joined[word]] # tup is (0,4) where 0 is the sent ind and 4 is the index of the word

    # 7. Get the embeddings for the sentences we want
    char_ids = batch_to_ids(sentences_with_word)        # TODO: maybe this is for models trained by us? use sent2elmo
    embeddings_1 = elmo(char_ids)
    embeddings_all_words = embeddings_1["elmo_representations"][0]

    # 8. Get the embeddings for the context (WITHOUT the target word)
    context_embed_no_target_word = []
    for idx in ind_joined[word]:
        start_idx_target_word = idx[1]
        context_embed_no_target_word.append(embeddings_all_words[0:start_idx_target_word]+ embeddings_all_words[start_idx_target_word+1:])
        print(sentences_with_word[idx[0]])  # prints the sentence
        print("sentence embedding with target word:")
        print(embeddings_all_words[idx[0]]) # prints the embedding for the whole sentence
        print("sentence embedding WITHOUT target word:")
        print(embeddings_all_words[0:start_idx_target_word]+ embeddings_all_words[start_idx_target_word+1:])

    # 9. Determine the best k for K-means
    # TODO: [add Pia's code]
    # 10. Cluster the CONTEXT embeddings WITHOUT the target word
    # TODO: use cluster() method to get the labels for sentences
    # labels = cluster()

    # 11.  Divide the labels into two lists

    # 11.1 Determine where sentences from corpus2 start in sentences_with_word
    tmp = [(1,2),(2,3),(3,3),(4,3)]
    joined = [["a","b"], ["a","b","f"],["r"],["bla"],["z","t","b","f"]]
    corpus_old = [["a","b"], ["a","b","f"],["r"]]
    start_ind = 0
    for idx, t in enumerate(tmp): #ind_joined[word]):
        if t[0] > len(corpus_old)-1:
            start_ind = t[0]
            print("Index:", t[0], "length of old corpus:",len(corpus_old)," last index in old corpus:",len(corpus_old)-1)
            break

    labels_corpus1 = [0,0,1,1,1,1,0,3,3] #labels[0:start_ind]
    labels_corpus2 = [0,0,0,2,2,2,0,3,3,3] #labels[start_ind:]
    # TODO: problem: the corpora where joined before collecting the indices of the words, so need to check the indices



    # 12. Use the Counter object to count clusters
    cor1_clusters = Counter(labels_corpus1)
    cor2_clusters = Counter(labels_corpus2)

    # get kmeans.labels_
    #all_clusters = kmeans.labels_
    all_clusters = [0,1,2,3]
    # TODO: CHANGE k
    k = 5
    if changed_sense(all_clusters,cor1_clusters,cor2_clusters,6):
        print("Changed sense")
    else:
        print("No change in senses detected")
