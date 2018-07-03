#/bin/bash

basedir=`pwd`

ngrams="1 3"
ngrams="3 1"
dims1="5000 10000 20000 30000 50000"

dims1="100000"
dims2="200000"
dims="100000 150000 200000 300000 400000 500000 600000"
seg_methods='jieba'
min_df=20
balance='balanced'
balances='none balanced'

train_fname="../../segdocs/small.seg_by_jieba"
train_fname="../../segdocs/big.seg_by_jieba.clear"
test_fname="../../segdocs/test.seg_by_jieba"

loop_train() 
{
    dims=$1
    seg_method=$2
    ngram=$3
    min_df=$4
    train_fname=$5
    test_fname=$6
    class_weight=$7

    for dim in $dims
    do
        echo "Process $dim"
        python svm_full.py  $dim $seg_method $ngram  $min_df $train_fname $test_fname $class_weight &
    done 

    #wait
}

for ngram in $ngrams
do
    for seg_method in $seg_methods
    do
	for balance in $balances
	do
	    loop_train "$dims" $seg_method $ngram $min_df $train_fname $test_fname $balance
	done
    done
done
