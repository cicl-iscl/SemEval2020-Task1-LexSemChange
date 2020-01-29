import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
np.set_printoptions(threshold=sys.maxsize)


def plot_word_2dim(word_to_plot, filename_plot, embeddings_array, labels):
    """
        Creates a 2 dimensional scatter plot of the data points of a single word labelled with the cluster labels.
        :param word_to_plot: string of the word that should be plotted
        :param filename_plot: name under which the plot will be saved
        :param embeddings_array: embedding vector of the specific word
        :param labels: the cluster labels of the embeddings_array
    """
    #print("unreduced array has dimensions: ", embeddings_array.shape)
    n_components = embeddings_array.shape[0]

    # reduce the dimensions of the embeddings to 2
    pca = PCA(n_components=n_components)
    y = pca.fit_transform(embeddings_array)
    reduced = TSNE(n_components=2).fit_transform(y)  # further reduce to 2 dim using t-SNE

    x_dim = reduced[:, 0]  # returns the first columm
    y_dim = reduced[:, 1]  # returns the first columm

    # defining an array of colors (if there are more than 10 clusters more colors need to be added)
    colors = ["cyan", "purple", "olive", "red", "blue", "green", "black", "grey", "orange", "yellow"]
    ax = plt.subplot()
    for i in range(reduced.shape[0]):
        col = colors[labels[i]]
        ax.scatter(x_dim[i], y_dim[i], color=col)

    print("Saving the 2d plot...")
    plt.title("2-dim Embeddings for \"" + word_to_plot + "\"")
    plt.xlabel("X")
    plt.ylabel("Y")
    # save the plot as it's not possible to show the plot when running on the server
    plt.savefig("./out/" + filename_plot)
    plt.clf()


def plot_word_3dim(word_to_plot, filename_plot, embeddings_array, labels):
    """
        Creates a 3 dimensional scatter plot of the data points of a single word labelled with the cluster labels.
        :param word_to_plot: string of the word that should be plotted
        :param filename_plot: name under which the plot will be saved
        :param embeddings_array: embedding vector of the specific word
        :param labels: the cluster labels of the embeddings_array


        DELETED :param path_corpus: path of the corpus that contains the word
        DELETED :param filename_datapoints_words: text file that will be created containing the data points of all the words
                                          from the corpus
                                            format: x1   y1   word1
                                                    x2   y2   word2
        DELETED :param filename_datapoints_specific_word: text file that will be created containing the data points of
                                                  only the word_to_plot (same format as before)

    """

    #print("unreduced array has dimensions: ", embeddings_array.shape)
    n_components = embeddings_array.shape[0]

    # reduce the dimensions of the embeddings to 3
    pca = PCA(n_components=n_components)
    y = pca.fit_transform(embeddings_array)
    reduced = TSNE(n_components=3).fit_transform(y)  # further reduce to 2 dim using t-SNE

    # defining an array of colors (if there are more than 10 clusters more colors need to be added)
    colors = ["cyan", "purple", "olive", "red", "blue", "green", "black", "grey", "orange", "yellow"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_data = reduced[:, 0]  # returns the first columm
    y_data = reduced[:, 1]  # returns the second columm
    z_data = reduced[:, 2]  # returns the third columm

    for i in range(reduced.shape[0]):
        col = colors[labels[i]]
        ax.scatter(x_data[i], y_data[i], z_data[i], c=col, depthshade=True, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # save the plot as it's not possible to show the plot when running on the server
    print("Saving the 3d plot...")
    plt.title("2-dim Embeddings for \"" + word_to_plot + "\"")
    plt.savefig("./out/" + filename_plot)
    plt.clf()
