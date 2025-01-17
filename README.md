# SemEval2020 - System Description

Google form:
https://docs.google.com/forms/d/e/1FAIpQLSf87XLY1D7aMhGrSFyZ2FfxjrJmBbKx9byX6HMEPtGNj9MQ_w/viewform

- Short description of your system. *: clustering of contextual word embeddings
- Semantic Representation of your model (e.g. count vectors): Contextual word embeddings (Elmo)
- Alignment type (if applicable, e.g. column intersection): none
- Change measure (e.g., cosine distance): size and amount of clusters

DETAILED DESCRIPTION:

- Contextual embeddings: Elmo, we used pretrained word embeddings, word embeddings (english=official embs, other langs=elmoformanylangs)

- we combined the lemmatized corpora (historic + modern) → this way, no alignment is necessary as they are in the same embedding space

- collect indizes of each word in corpus

- get embeddings for sentences that contain target word 

- cluster these embeddings with dbscan (GER & LAT) or kmeans (ENG & SWE)

- kmeans: find optimal number of clusters:
	→ cluster with number of clusters=1-10 
	→ use elbow method to pick optimal cluster number (+ create artificial value for 0 clusters)
	→ use this optimal number to do final clustering> 

- dbscan: set epsilon + min_samples

- take cluster labels and determine which labels belong to historic and which one to modern corpus, count amount of data points in each cluster

- detect change using k and n (given by task organizers)

- get Jensen-Shannon-Distance of the probability distributions of the cluster labels in C1 and C2



# SemEval2020-Task1-LexSemChange
participation in the shared task of SemEval 2020

MEETINGS
- kickoff: individual meeting with Cagri -> Tuesday, November 12 at 12:30
- next class: Tuesday, January 14 -> summary of team work
- in between: individual meeting(s) with Cöltekin
    -> Tuesday, Nov. 19th, 12:30

TO DO until monday, November 11:
- find & read papers (file "literature")
- prepare/read the task
- check the data sets of the task

NEXT
- write basic code: next one/two weeks
- find similar approaches to our task and how they solved it

- solve the alignment of the embedding over 2 corpora
- need more memory - can we use server from SfS? -> YES, connected via SSH





TRAINING MODEL CONTEXTUAL EMBEDDINGS:

This is in the python file /bilm-tf-master/bin/train_elmo.py - n_train_tokens have to be adjusted for each corpus (we do not use held out files anymore)

number of tokens in training data (this for 1B Word Benchmark)

n_train_tokens = 768648884

these are the amount of words in all training files (1 sentence per file) ignoring the heldout files if we have them

n_train_tokens = 181530  //GERMAN C1 + C2

n_train_tokens = 176451  //ENGLISH C1 + C2 (train model on own corpus)




# Best Tuning Results for each language

Language	latin - using word embeddings (not context embeddings)

- Target file	latin.txt
- k	0
- n	1.0
- algorithm	KMEANS
- k value for get_k	1.6
- accuracy	0.8115942028985508


Language    english - using embeddings (not context embeddings)

- Target file	english.txt
- k	2
- n	5
- algorithm	KMEANS
- k value for get_k	1.3
- accuracy	0.5
