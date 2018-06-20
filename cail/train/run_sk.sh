dd=`date +%Y%m%d_%H`
methods="MI CE DF"
fea_nums="1500 5000 10000 15000 30000 50000 100000"

methods="CE"
fea_nums="100000"
ngram=1

function do_train_for_num()
{
    fea_num=$1
    for method in $methods
    do
        rst_dir="train_rst"
        [[ ! -d $rst_dir ]] && ( mkdir $rst_dir )
    
        train_file="svm_format_samples/train.${method}_${ngram}"
        test_file="svm_format_samples/test.${method}_${ngram}"
        nohup python train_by_sk.py $train_file $test_file $fea_num $method > ${rst_dir}/${fea_num}_gram${ngram}_${method}.out.$dd &
    done
}

for fn in $fea_nums
do
    do_train_for_num $fn
done

