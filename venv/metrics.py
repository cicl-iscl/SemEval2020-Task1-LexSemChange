import numpy as np
from collections import Counter
from scipy.spatial import distance

"""
    Calculates the Jensen Shannon Distance between two 1-dim. probability arrays, examples:
    distance.jensenshannon(prob_array_1, prob_array_2, base_logarithm) -> probability arrays need to have same dimensions
    distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    
    we have the Counter objects of the clusters for each corpus and we know that their labels correspond between corpora:
    word, corpus1: {cluster_0: amount_datapoints, cluster_1: amount_datapoints, cluster_2: amount_datapoints}
          corpus2: {cluster_0: amount_datapoints, cluster_1: amount_datapoints}
          
    we need: 
    word, corpus1: [prob_cluster0, prob_cluster1, prob_cluster2]
          corpus2: [prob_cluster0, prob_cluster1, prob_cluster2]
          
    example:
    "walk", corpus1: {0: 18, 1: 53, 2: 9}  ---> total of 80 datapoints
            corpus2: {0: 51, 1: 2}         ---> total of 53 datapoints
            
    "walk", corpus1: [18/80, 53/80, 9/80]
            corpus2: [51/53, 2/53, 0]
"""

"""
    Converts two Counter or Dictionary object to a probability array and puts the into the same length
    {0: 18, 1: 53, 2: 9} ---> [18/80, 53/80, 9/80]
    Args:
        label_dict_corp1: dict or Counter of labels of corpus 1
        label_dict_corp2: dict or Counter of labels of corpus 2
                log_base: base of logarithm that will be used for the Jensen-Shannon-Divergence, default is 2 
    
    returns: the Jensen Shannon Distance
    
    EXAMPLE: Swedish word "första":
    labels corpus 1: [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0] --> dict object = {0: 11, 1: 7}
    labels corpus 2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]                         --> dict object = {0: 10}
    
    
    labels1:  {-1: 3409, 0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
    labels2:  {-1: 1839, 5: 3, 6: 7, 7: 2, 8: 3, 9: 2, 10: 2}
    
    
"""
def calculate_jsd(label_dict_corp1, label_dict_corp2, log_base=2.0):
    # convert argument to dict (in case we are given a Counter objects)
    labels1 = dict(label_dict_corp1)
    print("labels1: ", labels1)
    labels2 = dict(label_dict_corp2)
    print("labels2: ", labels2)

    # find out the distinct values of labels1 and labels2
    distinct_labels = set()
    for key, value in labels1.items():
        print("key: ", key)
        distinct_labels.add(key)
    for key, value in labels2.items():
        print("key: ", key)
        distinct_labels.add(key)

    print("distinct labels: ", distinct_labels)

    # replace cluster label -1 with a positive number: take the highest cluster label int and add 1
    max_int_label = max(distinct_labels)
    labels1[max_int_label+1] = labels1[-1]
    del labels1[-1]
    labels2[max_int_label+1] = labels2[-1]
    del labels2[-1]

    print("labels1 after renaming key: ", labels1)
    print("labels2 after renaming key: ", labels2)


    # initialize two np arrays to the size of distinct_labels
    prob_array1 = np.zeros(len(distinct_labels))
    prob_array2 = np.zeros(len(distinct_labels))

    print("proba1", prob_array1)
    print("test", prob_array1[-1])
    print("proba2", prob_array2)


    # get the total amount of datapoints per word per corpus
    total_datapoints1, total_datapoints2 = 0, 0
    for key, value in labels1.items():
        total_datapoints1 += value
    for key, value in labels2.items():
        total_datapoints2 += value

    print("total_datapoints1", total_datapoints1)
    print("total_datapoints2", total_datapoints2)

    for key, value in labels1.items():
        prob_array1[key] = value/total_datapoints1
        print("prob_array1[",key,"] = ", value, "/", total_datapoints1)

    for k, v in labels2.items():
        prob_array2[k] = v/total_datapoints2
        print("prob_array2[", k, "] = ", v, "/", total_datapoints2)

    print("prob array 1: ", prob_array1)
    print("prob array 2: ", prob_array2)

    jsd = distance.jensenshannon(prob_array1, prob_array2, log_base)

    return jsd



if __name__ == '__main__':

    # possible to pass dict objects OR Counter objects
    #a1 = {0: 11, 1: 7}
    #a2 = {0: 10}
    a1 = Counter({-1: 3409, 0: 2, 1: 2, 2: 2, 3: 2, 4: 2})
    a2 = Counter({-1: 1839, 5: 3, 6: 7, 7: 2, 8: 3, 9: 2, 10: 2})

    jsd = calculate_jsd(a1, a2, log_base=2.0)
    print("JSD: ", jsd)

