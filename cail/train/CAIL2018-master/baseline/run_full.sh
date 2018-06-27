#/bin/bash

train_fname="../../segdocs/big.seg_by_jieba.clear.old"
train_fname="../../segdocs/small.seg_by_jieba"
test_fname="../../segdocs/test.seg_by_jieba"
t1_fname="../../segdocs/test1"

python3 svm_full.py 20000 jieba 3 5 $train_fname $test_fname
#python3 svm_full.py 20000 jieba 3 5 $t1_fname $t1_fname
