#encoding: utf8
import numpy as np
import sys
import json
import pdb

'''
MI(X) = sum_Y sum_X (p(yi,xi) log(p(yi,xi)/(p(xi)p(yi)), xi: 0,1: x
      = sum_Y p(y_i) sum_X(p(xi|yi) log(p(xi|yi)/p(xi)), for p(xi,yi) = p(xi|yi) * p(yi)
x_doc_nums: 
 x_doc_nums[0] (yi, xi_docnums, si_wordfreq) 
y_doc_nums:  numpy.array
y_dist: y numpy.array  
'''
def MI(x_doc_nums, y_doc_nums, y_dist):
    #
    total_doc_num = y_doc_nums[0]
    x_num = x_doc_nums[0][1]
    p_x1 = x_num * 1./ total_doc_num 
    p_x = [1. - p_x1, p_x1]
    p1 = np.zeros(len(y_dist))   #p(x1|yi) = 0, p(x1|yi) * log(p(x1|yi)/p(x1)) = 0
    p0 = -np.ones(len(y_dist)) / np.log(p_x[0])  #p(x0|yi) = 1, p(x0|yi) * log(p(x0|yi)/p(x0)) = -1/log(p(x0))
    p0[0] = 0.
    for yi,docnum,wfreq in x_doc_nums[1:]:
        if docnum != 0:
            p_x1_given_yi = docnum*1./y_doc_nums[yi]
            p_x0_given_yi = 1-p_x1_given_yi
            
            p1[yi] = p_x1_given_yi * np.log(p_x1_given_yi/p_x[1])
            if docnum < y_doc_nums[yi]:
                p0[yi] = p_x0_given_yi * np.log(p_x0_given_yi/p_x[0])
            else:
                p0[yi] = 0.
    return np.sum(y_dist * (p0 + p1))

'''
document frequency 
idf. idf
'''
def DF(x_doc_nums, y_doc_nums=None, y_dist=None):
    return x_doc_nums[0][1]

'''
cross entropy 
CE(w) = sum{p(ci|w) * log[p(ci|w)/p(ci)]}
'''
def CE(x_doc_nums, y_doc_nums, y_dist):
    x_nums = [df for yi,df,wf in x_doc_nums[1:] ]
    y_ids = [yi for yi,df,wf in x_doc_nums[1:] ]
    
    p_cw = np.array(x_nums) *1. / np.take(y_doc_nums, y_ids)
    return np.sum(p_cw * np.log(p_cw/np.take(y_dist,y_ids)))

'''
expect cross entroy:
    ECE(w) = p(w) * CE(w)
'''
def ECE(x_doc_nums, y_doc_nums, y_dist):
    doc_num = y_doc_nums[0]
    ce = CE(x_doc_nums, y_doc_nums, y_dist)
    p_x = x_doc_nums[0][1] * 1./doc_num
    return p_x * ce

fea_select_dict = {"MI":MI, "CE":CE, "ECE":ECE, "DF": DF}

def select_feature(general_dist, detail_dist_list, s_name="CE", low_bound=-1):
    y_nums, y_dist = general_dist
    sfun = fea_select_dict.get(s_name)
    rlist = []
    for w,w_c_nums in detail_dist_list:
        if w_c_nums[0][1] <= low_bound:
            continue
        rank = sfun(w_c_nums, y_nums, y_dist) 
        rlist.append((w, rank, w_c_nums))
    rlist.sort(cmp=lambda x,y:cmp(x[1], y[1]), reverse=True)
    return rlist

def parse_dist_line(line):
    arr = line.strip().split('\t')
    word = arr[0]
    x_c_nums = [[int(x) for x in st.split(',')] for st in arr[1:]]
    x_c_nums.sort(cmp=lambda x,y:cmp(x[0],y[0]))
    return word, x_c_nums

'''
: cid-->freq, cid==0,all docs_count
: word--> [(cid,df, wf)], cid==0: document frequence 
cid ==0, 
'''
def load_word_dist(fname='conf/word.dist'):
    general_dist = None
    detail_dist_list = [] 
    with open(fname) as fp:
        line =fp.readline()
        w,c_nums = parse_dist_line(line)
        #
        y_nums = map(lambda x:x[1], c_nums)
        #
        y_nums = np.array(y_nums)
        y_dist = y_nums / float(y_nums[0])
        general_dist = (y_nums, y_dist)
        
        for line in fp:
            w, w_c_nums = parse_dist_line(line) 
            detail_dist_list.append((w, w_c_nums))
    
    return general_dist, detail_dist_list

def save_features(features, outfn, total_docs_num):
    # w,rank, w_c_nums: 
    def  idf(df):
        return np.log(total_docs_num*1./df)
    features = map(lambda x,y:(y, x[0],x[1], x[2][0][1], idf(x[2][0][1])), features, range(1, len(features)+1))
    with open(outfn + ".json", 'w') as outf:
        json.dump(features, outf, indent=2)
    tmpfn = outfn + ".txt"
    with open(tmpfn, 'w') as outf:
        for wid,w,score,df,idf in features:
            line = '%s\t%s\t%.4f\t%s\t%.4f\n' % (wid, w, score, df, idf)
            outf.write(line)


method_lst = ["MI", "CE", "ECE", "DF"]

word_dist_fname=sys.argv[1]
out_fname=sys.argv[2]
method=sys.argv[3]

general_dist,detail_dist_list = load_word_dist(word_dist_fname)
docs_num = general_dist[0][0]
rlst = select_feature(general_dist, detail_dist_list, method, low_bound=20)
fname = '%s_by_%s' % (out_fname, method)
save_features(rlst, fname, docs_num)
