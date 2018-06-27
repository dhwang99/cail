#/bin/bash

train_fname="../../segdocs/big.seg_by_jieba.clear.old"
train_fname="../../segdocs/small.seg_by_jieba"
test_fname="../../segdocs/test.seg_by_jieba"

python3 svm_full.py 20000 jieba 3 5 $train_fname $test_fname
