#encoding: utf8
import numpy as np
import sys
import pdb
import json

ngram=3
def doc_to_svm(filename, svmfn, feature_fn, feature_num):
    max_sent_len = 25
    word_dict = None
    with open(feature_fn) as fp: 
        features = json.load(fp)
        features = features[:feature_num]
        word_dict = dict(map(lambda x:(x[1],x), features))
    svm_out = open(svmfn, 'w')

    def smooth_si(si):
        return min(max_sent_len,si[0]), si[1]
    
    with open(filename) as fin:
        id = 0
        for line in fin:
            bow = dict()
            line = line.decode('utf8').strip()
            aa = line.strip().split('\t')
            if len(aa) != 4:
                pdb.set_trace()
                continue
            
            ws = aa[3].split(',')
            word_poses = [x.split(' ') for x in ws ]
            la = 0
            for wi in xrange(len(word_poses)):
                w,p = word_poses[wi]
                if p[0] == 'u' or p[0] == 'x':
                    la = 0
                    continue
                bow.setdefault(w, 0)
                bow[w] += 1
                la += 1
                if la >= 2 and ngram >= 2:
                    w2 = "%s_%s" % (word_poses[wi-1][0], w)
                    bow.setdefault(w2, 0)
                    bow[w2] += 1
                    if la >= 3 and ngram >= 3:
                        w3 = "%s_%s" % (word_poses[wi-2][0], w2)
                        bow.setdefault(w3, 0)
                        bow[w3] += 1
            
            features = []
            for w,freq in bow.iteritems():
                fea = word_dict.get(w)
                if fea:
                    tfidf = fea[4] * freq
                    features.append((fea[0], tfidf))
            
            if len(features) == 0:
                continue
            #
            features.sort(cmp=lambda x,y:cmp(x[0],y[0]))
            svm_str = ' '.join(map(lambda x:'%d:%.4f'%x, features))
            #
            accids = aa[1].split(',')
            for accid in accids:
                s = '%s %s\n' % (accid, svm_str)
                svm_out.write(s)
                #pdb.set_trace()
                svm_out.flush()
    svm_out.close() 
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "%s source_file dst_file feature_file feature_num" % sys.argv[0]
        sys.exit(1)
    fn = sys.argv[1]
    sf=sys.argv[2]
    feature_file = sys.argv[3]
    feature_num = int(sys.argv[4])
    doc_to_svm(fn, sf, feature_file, feature_num)
