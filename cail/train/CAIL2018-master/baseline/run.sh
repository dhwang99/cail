#/bin/bash

basedir=`pwd`

ngrams="3 1"
ngrams="1 3"
dims1="5000 10000 20000 30000 50000"
dims2="100000 150000 200000 300000 4000000"

dims1="50000 100000"
dims2="200000 300000"
seg_methods='jieba thulac'
seg_methods='jieba'
min_df=20

train_fname="../../segdocs/big.seg_by_jieba.clear.old"
train_fname="../../segdocs/small.seg_by_jieba"
test_fname="../../segdocs/test.seg_by_jieba"

loop_train() 
{
    dims=$1
    seg_method=$2
    ngram=$3
    min_df=$4
    train_fname=$5
    test_fname=$6

    for dim in $dims
    do
        echo "Process $dim"
        python3 svm_full.py  $dim $seg_method $ngram  $min_df $train_fname $test_fname &
    done 

    wait
}

for ngram in $ngrams
do
    for seg_method in $seg_methods
    do
        loop_train "$dims1" $seg_method $ngram $min_df $train_fname $test_fname
        loop_train "$dims2" $seg_method $ngram $min_df $train_fname $test_fname
    done
done
