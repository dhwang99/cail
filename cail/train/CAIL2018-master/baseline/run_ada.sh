#/bin/bash

train_fname="../../segdocs/small.seg_by_jieba"
train_fname="../../segdocs/big.seg_by_jieba.clear"
test_fname="../../segdocs/test.seg_by_jieba"
t1_fname="../../segdocs/test1"

#python adaboost_svm.py 200001 jieba 3 20 $train_fname $test_fname
python adaboost_svm.py 20000 jieba 3 5 $t1_fname $t1_fname
#python svm_full.py 20000 jieba 3 5 $t1_fname $t1_fname
