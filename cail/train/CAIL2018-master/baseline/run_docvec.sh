#/bin/bash

clf_name=$1
seg_name="jieba"

#small train data
train_fname="../../segdocs/small/train.seg_by_jieba"
test_fname="../../segdocs/small/test.seg_by_jieba"
validation_fname="../../segdocs/small/validation.seg_by_jieba"

nohup python gen_docvec.py $train_fname $validation_fname $test_fname $seg_name $clf_name 1>>gendoc.std 2>>gendoc.err &

#check data
test_fname="../../segdocs/check/test.seg_by_jieba"
train_fname="../../segdocs/check/train.seg_by_jieba"
validation_fname="../../segdocs/check/validation.seg_by_jieba"

nohup python gen_docvec.py $train_fname $validation_fname $test_fname $seg_name $clf_name 1>>gendoc.std 2>>gendoc.err &

#full train data
train_fname="../../segdocs/big/train.seg_by_jieba"
test_fname="../../segdocs/big/test.seg_by_jieba"
validation_fname="../../segdocs/big/validation.seg_by_jieba"

nohup python gen_docvec.py $train_fname $validation_fname $test_fname $seg_name $clf_name 1>>gendoc.std 2>>gendoc.err &

