#!/bin/bash

# Script to tune the english and latin model using different hyperparameters
#
# latin and english only as we have dev sets only for these languages
# K=0, N=2 for latin
# K=2, N=5 for en, ger, swe
#  KMEANS:
#  python3 cluster_fixed.py LANGUAGE K N classify_words kmeans -kmk VALUE_OF_KMK -t file_with_targets.txt
#  DBSCAN:
#  python3 cluster_fixed.py LANGUAGE K N classify_words dbscan -eps VALUE_OF_EPS -nsamp NUM_OF_SAMPLES -t file_with_targets.txt
#
# add this before python3 ... when running English: TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" (makes model faster)

echo "TUNING FOR LATIN:"
echo "start processing k-means kmk values (1.1 - 2.0)..."
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.1 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.3 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.4 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.5 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.6 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.7 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.8 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 1.9 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words kmeans -kmk 2.0 -t file_with_targets.txt

echo "done with k-means for Latin"

echo "start processing dbscan values for epsilon (2.5 - 5.0) and n-samples (2 and 3)..."

python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 2.5 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 3.0 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 3.5 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 4.0 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 4.5 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 5.0 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 2.5 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 3.0 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 3.5 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 4.0 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 4.5 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py latin 0 2 classify_words dbscan -eps 5.0 -nsamp 3 -t file_with_targets.txt

echo "done with dbscan for Latin"


echo "TUNING FOR ENGLISH:"

echo "start processing k-means kmk values (1.1 - 2.0)..."
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.1 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.3 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.4 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.5 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.6 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.7 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.8 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 1.9 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words kmeans -kmk 2.0 -t file_with_targets.txt

echo "done with k-means for English"

echo "start processing dbscan values for epsilon (2.5 - 5.0) and n-samples (2 and 3)..."

python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 2.5 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 3.0 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 3.5 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 4.0 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 4.5 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 5.0 -nsamp 2 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 2.5 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 3.0 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 3.5 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 4.0 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 4.5 -nsamp 3 -t file_with_targets.txt
python3 cluster_fixed.py english 2 5 classify_words dbscan -eps 5.0 -nsamp 3 -t file_with_targets.txt

echo "done with dbscan for English"


