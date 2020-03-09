import sys
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

def create_results_file(path, predictions, rank_by_jsd_score = False): #predictions is {"word1":1, "word2":1} or {"word1":0.419, "word2":0.231}
    if rank_by_jsd_score:
        predictions= {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse = True)}
    else:
        predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[0])}
    with open(path, 'w', encoding='utf8') as f:
        for k,v in predictions.items():
            line = str(k) + '\t'+ str(v) +'\n'
            f.write(line)

if __name__ == "__main__":

    if sys.argv[1]:
        language = sys.argv[1]
    else:
        print("Please provide a language.")
        sys.stderr.write("Please provide a language.")



    label_predictions = {'imperator': '1', 'fidelis': '0', 'sanctus': '0', 'civitas': '0', 'scriptura': '0', 'dux': '0', 'pontifex': '0', 'virtus': '0', 'cohors': '0', 'itero': '0', 'consul': '0', 'humanitas': '0', 'potestas': '0', 'dolus': '0', 'beatus': '0', 'poena': '0', 'credo': '0', 'honor': '0', 'consilium': '0', 'acerbus': '0', 'nobilitas': '0', 'templum': '0', 'ancilla': '0', 'sapientia': '0', 'senatus': '0', 'hostis': '0', 'titulus': '0', 'sensus': '0', 'adsumo': '0', 'oportet': '0', 'simplex': '0', 'regnum': '0', 'dubius': '0', 'licet': '0', 'voluntas': '0', 'nepos': '0', 'necessarius': '0', 'jus': '0', 'salus': '0', 'sacramentum': '0'}
    jsd_scores = {'imperator': 0.658, 'fidelis': 0.055, 'sanctus': 0.224, 'civitas': 0.289, 'scriptura': 0.079, 'dux': 0.076, 'pontifex': 0.095, 'virtus': 0.135, 'cohors': 0.207, 'itero': 0.006, 'consul': 0.098, 'humanitas': 0.03, 'potestas': 0.043, 'dolus': 0.06, 'beatus': 0.382, 'poena': 0.096, 'credo': 0.024, 'honor': 0.184, 'consilium': 0.069, 'acerbus': 0.054, 'nobilitas': 0.026, 'templum': 0.21, 'ancilla': 0.156, 'sapientia': 0.117, 'senatus': 0.027, 'hostis': 0.06, 'titulus': 0.119, 'sensus': 0.047, 'adsumo': 0.21, 'oportet': 0.123, 'simplex': 0.0, 'regnum': 0.201, 'dubius': 0.132, 'licet': 0.081, 'voluntas': 0.085, 'nepos': 0.066, 'necessarius': 0.033, 'jus': 0.074, 'salus': 0.107, 'sacramentum': 0.31}

    # TODO: only for the development phase: can be used to see how the predicted ranking compare to the real ones
    #gold_rankings_filename = "{}_rankings_gold.txt".format(language)
    #compare_ranks(gold_rankings_filename, jsd_scores)

    #path_task1 = "/home/anna/train_elmo/SemEval2020/starting_kit_2/test_data_public/answer/task{}/{}.txt".format(1, language)
    #path_task2 = "/home/anna/train_elmo/SemEval2020/starting_kit_2/test_data_public/answer/task{}/{}.txt".format(2, language)
    path_task1 = "answer/task{}/{}.txt".format(1, language)
    path_task2 = "answer/task{}/{}.txt".format(2, language)

    # CREATES FILES WITH ANSWERS. NEED TO HAVE created an answer folder with two subfolders task1 and task2
    create_results_file(path_task1,label_predictions)
    # TODO: set to true if lines should be in order of the jsd_value, not alphab. order of words
    create_results_file(path_task2, jsd_scores, False)