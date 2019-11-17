import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.manifold import TSNE
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.tensorboard.plugins import projector
import torch
from elmoformanylangs import Embedder
import sys
import numpy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#numpy.set_printoptions(threshold=sys.maxsize)


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


if __name__ == '__main__':

    # following this tutorial with the German language: https://github.com/HIT-SCIR/ELMoForManyLangs

    # arg 1: the absolute path from the repo top dir to you model dir
    # arg 2: default batch_size: 64
    e = Embedder('/home/pia/Python/SemEval2020/142-german-model')

    sentences = preprocess('/home/pia/Python/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1-small.txt')
    # the list of lists which store the sentences

    embs = e.sents2elmo(sentences)
    # will return a list of numpy arrays
    # each with the shape=(seq_len, embedding_size)
    print(len(embs))
    print(embs[0])
    print(embs[0].shape)
    print(sentences[0])
    print(len(sentences[0]))

    embs_array = numpy.concatenate(embs, axis=0)
    print("type of embs_array: ", type(embs_array))


    # use PCA and t-SNE to reduce the 1,024 dimensions which are output from ELMo
    # down to 2 so that we can review the outputs from the model
    pca = PCA(n_components=50)  # reduce down to 50 dim
    y = pca.fit_transform(embs_array)
    reduced = TSNE(n_components=2).fit_transform(y)  # further reduce to 2 dim using t-SNE
    print("finished reduction")

    print("reduced space dimensions: ", reduced.shape)
    print("reduced emb space: ", reduced)

    all_tokens = []
    with open('/home/pia/Python/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1-small.txt',
              'r', encoding="utf8") \
            as f:
        for line in f:
            for word in line.split():
                all_tokens.append(word)


    count = 0
    with open("emb_german_reduced.txt", "r+", encoding="utf8") as f:
        for i in reduced:
            string = str(i[0]) + "\t" + str(i[1]) + "\t" + all_tokens[count]
            f.write(string + '\n')
            count += 1

    """
    # plot the whole space
    x_dim = np.loadtxt('emb_german_reduced.txt', usecols=0)
    y_dim = np.loadtxt('emb_german_reduced.txt', usecols=1)
    labels = np.loadtxt('emb_german_reduced.txt', dtype=str, usecols=2)

    ax = plt.subplot()
    ax.scatter(x_dim, y_dim)

    for i, text in enumerate(labels):
        ax.annotate(text, (x_dim[i], y_dim[i]))

    plt.title("Context Embeddings German Hist. Corpus (small)")
    plt.xlabel("x_dimension")
    plt.ylabel("y_dimension")
    plt.show()
    """

    # plot only a single word, e.g. "liegen"
    indices = [i for i, x in enumerate(all_tokens) if x == "Zeit"]

    with open("emb_german_reduced_liegen.txt", "w", encoding="utf8") as f_liegen:
        for i in indices:
            string = str(reduced[i][0]) + "\t" + str(reduced[i][1]) + "\t" + all_tokens[i]
            f_liegen.write(string + '\n')

    # plot
    x_dim = np.loadtxt('emb_german_reduced_liegen.txt', usecols=0)
    y_dim = np.loadtxt('emb_german_reduced_liegen.txt', usecols=1)
    labels = np.loadtxt('emb_german_reduced_liegen.txt', dtype=str, usecols=2)
    ax = plt.subplot()
    ax.scatter(x_dim, y_dim)

    for i, text in enumerate(labels):
        ax.annotate(text, (x_dim[i], y_dim[i]))

    plt.title("Context Embeddings German Hist. Corpus (small)")
    plt.xlabel("x_dimension")
    plt.ylabel("y_dimension")
    plt.show()
