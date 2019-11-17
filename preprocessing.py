import csv
import collections
import random



"""
    SOME PREPROCESSING STEPS FOR THE LATIN CORPUS(JUST FOR NOW) 
    IN CASE WE DONT USE PRE-TRAINED EMBEDDINGS
"""
def load_corpus(filename):
    with open(filename) as f:
        content = f.readlines()
    print(content[0])
    tokenized = [line.strip().split() for line in content]
    return tokenized

"""
    Removes # from words 
"""
def clean_corpus(orig_corpus):
    clean_corpus =[]
    for line in orig_corpus:
        sent = [token[:token.index("#")] if "#" in token else token for token in line]
        clean_corpus.append(sent)
        print(sent)
    return clean_corpus


def create_vocab_file(corpus, filename):
    counter = collections.Counter(x for sent in corpus for x in sent)
    special_char = ['<S>', '</S>', '<UNK>']
    with open(filename, 'w', newline='\n') as f:
        for sc in special_char:
            f.write(sc)
            f.write("\n")
        sort = counter.most_common(len(dict(counter)))
        for token, count in sort:
            f.write(token)
            print(token, count)
            if count < len(sort):
                f.write("\n")

        """
        spamwriter = csv.writer(csvfile, delimiter=' ')
        for sc in special_char:
            spamwriter.writerow(sc)
        sort = counter.most_common(len(dict(counter)))
        for x,i in sort:
            spamwriter.writerow(x)
 """

def create_training_files(corpus):
    print(len(corpus))
    num_heldout = int(len(corpus)*.2)
    print("words in the corpus: ", len(corpus))
    print("held out sentences: ", num_heldout)
    # TODO: NEED TO ADD random shuffling?

    # make a heldout file for testing
    heldout = corpus[:num_heldout]
    create_file(heldout,"heldout.txt")

    # make separate training files
    training = corpus[num_heldout+1:]
    path = 'training_files/{}'
    for i,sent in enumerate(training):
        with open(path.format(i), 'w', newline='\n') as f:
                f.write(" ".join(sent))
                f.write("\n")
    flat_list = [item for sublist in training for item in sublist]
    print("number of tokens in training", len(flat_list))
    """
    shuffled_corpus = random.shuffle(corpus) 
    print(random.shuffle(corpus))
    
    for s in shuffled_corpus:
        print(s)
    """
def create_file(corpus, file):
    with open(file, 'w', newline='\n') as f:
        for line in corpus:
            f.write(" ".join(line))
            f.write("\n")


if __name__ == "__main__":
        corpus = load_corpus("german/corpus1/corpus1.txt")
        new_corpus = clean_corpus(corpus)
        print(new_corpus)
        create_training_files(new_corpus)
        #print(clean_corpus)
        create_vocab_file(new_corpus, "vocab.txt")
"""
        with open('clean_latin.csv', 'w', newline='\n') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            spamwriter.writerows(clean_corpus)
"""