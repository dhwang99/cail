
do_seg_docs=0
gen_word_dist=0
do_feature_select=1

rawtrainfile='segdocs/big.seged'
trainfile='segdocs/big.seged.clear'
testfile='segdocs/test.seged'

if [ $do_seg_docs -eq 1 ]
then
    python seg_docs.py ../data/cail2018_big/id.cail2018_big.json $rawtrainfile
    python seg_docs.py ../data/cail_0518/id.data_test.json $testfile

    python dedup_doc.py $testfile $rawtrainfile $trainfile
    exit
    
fi

methods='DF'
ngram=1

methods='CE MI DF'
ngram=2


wd_f="conf/word_${ngram}_gram.dist"
fea_prefix="features/${ngram}gram"
if [ $gen_word_dist -eq 1 ]
then
    python calc_word_dist.py $trainfile segdocs/.dist $ngram
    sort -nr -k 2 -t $'\t' segdocs/.dist > $wd_f
fi
#exit

if [ $do_feature_select -eq 1 ]
then
    for method in $methods
    do
        feaname=$fea_prefix"_by_"$method
        python feature_select.py $wd_f $feaname $method
    done
fi

for method in $methods
do
    fea_filename="${fea_prefix}_by_${method}.json"
    python doc2svm.py $trainfile svm_format_samples/train.${method}_${ngram} $fea_filename  300000
    python doc2svm.py $testfile svm_format_samples/test.${method}_${ngram} $fea_filename 300000 
done

