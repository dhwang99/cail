#/bin/bash

basedir=`pwd`

ngrams="3 1"
ngrams="1 3"
dims1="5000 10000 20000 30000 50000"
dims2="100000 150000 200000 300000 4000000"
seg_methods='jieba thulac'
min_df=5
train_fname='data_train.json'

loop_train() 
{
    dims=$1
    seg_method=$2
    ngram=$3
    min_df=$4
    train_fname=$5

    for dim in $dims2
    do
        echo "Process $dim"
        python3 svm.py  $dim $seg_method $ngram  $min_df $train_fname &
    done 

    wait
}

for ngram in $ngrams
do
    for seg_method in $seg_methods
    do
        loop_train "$dims1" $seg_method $ngram $min_df $train_fname
        loop_train "$dims2" $seg_method $ngram $min_df $train_fname
    done
done
