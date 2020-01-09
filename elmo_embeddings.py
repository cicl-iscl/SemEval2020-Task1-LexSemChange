import tensorflow as tf
from allennlp.modules.elmo import Elmo, batch_to_ids
from preprocessing import load_corpus, collect_all_occurrences

#import matplotlib.pyplot as plt
#import numpy as np
#from sklearn.manifold import TSNE
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.tensorboard.plugins import projector
#import torch
#from elmoformanylangs import Embedder
#import sys
#import numpy
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#import plotly.plotly as py
#import chart_studio.plotly as py
#import plotly.graph_objects as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#numpy.set_printoptions(threshold=sys.maxsize)


def preprocess_corpus(filename):
    with open(filename, encoding='utf8') as f:
        content = f.readlines()
        tokenized = [line.strip().split() for line in content]

    len_longest_sent = 0  # get the longest sentence to later pad all other sentences
    tokens_lengths = []   # collect the different lengths of the sentences in a list
    for sent in tokenized:
        tokens_lengths.append(len(sent))
        if len(sent) > len_longest_sent:
            len_longest_sent = len(sent)  # set this sentence length to the new longest sentence length
    # add padding
    for sent in tokenized:
        while len(sent) < len_longest_sent:
            sent.append("")

    return tokenized, tokens_lengths



"""
The elmo model consists of two files:

options.json : These are the parameters/options using which the language model was trained on
weights.hdf5 : The weights file for the best model

The input to the pre trained model (elmo) above can be fed in two different ways: Tokens and Default
With the tokens signature, the module takes tokenized sentences as input. The input tensor is a string tensor with 
shape [batch_size, max_length] and an int32 tensor with shape [batch_size] corresponding to the sentence length. 
The length input is necessary to exclude padding in the case of sentences with varying length.


Example of tokens_input

[["Argentina", "played", "football", "very", "well", "", "", "", ""],
 ["Brazil", "is", "a", "strong", "team", "", "", "", ""],
 ["Artists", "all", "over", "the", "world", "are", "attending", "the", "play"],
 ["Child", "is", "playing", "the", "guitar", "", "", "", ""],
 ["There", "was", "absolute", "silence", "during", "the", "play", "", ""]]

Each element contains one layer of ELMo representations with shape
(5, 9, 1024).
 5    - the batch size
 9    - the sequence length of the batch (longest sentence has 9 words)
 1024 - the dimension of each ELMo vector
 
The output (embeddings) is a dictionary with following keys:
    - word_emb: the character-based word representations with shape [batch_size, max_length, 512].
    - lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
    - lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
    - elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has 
            shape [batch_size, max_length, 1024]
    - default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
--> The "elmo" value is selected.


USE ELMO PROGRAMMATICALLY:

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "path to options file"
weight_file = "path to weights file"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

2 is an integer which represents num_output_representations. 
Typically num_output_representations is 1 or 2. For example, in the case of the SRL model in the above paper, 
num_output_representations=1 where ELMo was included at the input token representation layer. 
In the case of the SQuAD model,num_output_representations=2as ELMo was also included at the GRU output layer.

use batch_to_ids to convert sentences to character ids:

sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

embeddings[elmo] is length two list of tensors. 
Each element contains one layer of ELMo representations with shape
(2, 3, 1024).


"""

if __name__ == "__main__":

    # process corpus so it can be used as token_input and tokens_length can be extracted
    #tokens_input = preprocess_corpus('/home/pia/Python/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1-small.txt')[0]
    #tokens_length = preprocess_corpus('/home/pia/Python/SemEval2020/starting_kit/trial_data_public/corpora/german/corpus1/corpus1-small.txt')[1]
    #embeddings = elmo(inputs={"tokens": tokens_input, "sequence_len": tokens_length}, signature="tokens", as_dict=True)["elmo"]

    # ENGLISH CORPUS 1
    with open('/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus1/corpus1.txt',
              encoding='utf8') as f:
        content = f.readlines()
        # tokenized looks like: [['First', 'sentence', '.'], ['Another', '.']]
        tokenized_EN1 = [line.strip().split() for line in content]

    # ENGLISH CORPUS 2
    with open('/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus2/corpus2.txt',
            encoding='utf8') as f:
        content = f.readlines()
        # tokenized looks like: [['First', 'sentence', '.'], ['Another', '.']]
        tokenized_EN2 = [line.strip().split() for line in content]

    options_file_EN = '/home/pia/train_elmo/SemEval2020/elmo-model-checkpoint-EN/options.json'
    weight_file_EN = '/home/pia/train_elmo/SemEval2020/elmo-model-checkpoint-EN/weights.hdf5'

    elmo_EN = Elmo(options_file_EN, weight_file_EN, 2, dropout=0)

    # use batch_to_ids to convert sentences to character ids
    character_ids_EN1 = batch_to_ids(tokenized_EN1)
    character_ids_EN2 = batch_to_ids(tokenized_EN2)

    embeddings_EN1 = elmo_EN(character_ids_EN1)
    embeddings_EN2 = elmo_EN(character_ids_EN2)
    # output: embeddings_EN1[elmo_representations] is a list of two tensors.
    # Each element contains one layer of ELMo representations with shape (2, max_length, 1024)
    #print("embeddings_EN1 keys: ", embeddings_EN1.keys()) -->  dict_keys(['elmo_representations', 'mask'])
    elmo_embeddings_EN1 = embeddings_EN1['elmo_representations'][0]
    elmo_embeddings_EN2 = embeddings_EN2['elmo_representations'][0]
    print(elmo_embeddings_EN2)
    print("DONE WITH GETTING THE EMBEDDINGS FOR CORPUS EN1 and EN2!!!!!!!!!!!!")

    # NEXT: iterate through dict word_positions and get the corresponding vectors for each word to cluster them later:
    # load ENGLISH CORPUS and get word_positions dict
    corpus_english_1 = load_corpus(
        "/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus1/corpus1.txt")
    corpus_english_2 = load_corpus(
        "/home/pia/train_elmo/SemEval2020/starting_kit/trial_data_public/corpora/english/corpus2/corpus2.txt")

    # Get the dicts of word postitions, they look like:
    # manual [(1653, 4), (3317, 53)]
    # distinct [(1653, 6), (1740, 8), (1800, 40), (2615, 40), (2666, 56), (4569, 19), (4773, 9)] ...
    word_positions_english_1 = collect_all_occurrences(corpus_english_1)
    word_positions_english_2 = collect_all_occurrences(corpus_english_2)

    # Get the vector(s) for a single word
    print('word_positions_english_2[manual]: ', word_positions_english_2['manual'])
    print('get the vectors at these positions:')
    print('emb size ', elmo_embeddings_EN2.shape)
    for index in word_positions_english_2.get('manual'):
        print('index: ', index)
        print(elmo_embeddings_EN2[index[0]][index[1]])

    # collect all vectors from the same occurrence
    # -> visualize it with code from embeddings.py (embs.shape = elmo_embeddings_EN1/2.shape)








