#/bin/bash


#full train data
train_fname="../../segdocs/big/train.seg_by_jieba"
test_fname="../../segdocs/big/test.seg_by_jieba"
validation_fname="../../segdocs/big/validation.seg_by_jieba"

#check data
test_fname="../../segdocs/check/test.seg_by_jieba"
train_fname="../../segdocs/check/train.seg_by_jieba"
validation_fname="../../segdocs/check/validation.seg_by_jieba"

#small train data
train_fname="../../segdocs/small/train.seg_by_jieba"
test_fname="../../segdocs/small/test.seg_by_jieba"
validation_fname="../../segdocs/small/validation.seg_by_jieba"

#LR(LogisticRegression) SVC(linearSVC) SGD(SGDClassifier)
clf_name=$1
seg_name="jieba"

touch params.lst 
rm param.lineid
total=24
id=0

#删除过去的log
rm gs.std gs.err

while ((id<total))
do
    echo "id:$id"
    ((id=id+1))
    nohup python train_task.py $train_fname $validation_fname $test_fname $seg_name $clf_name 1>>gs.std 2>>gs.err &
done

#python3 grid_search_test.py 200000 jieba 3 20 $train_fname $test_fname $validation_fname
