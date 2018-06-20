#encoding: utf8
import jieba
import jieba.posseg as pseg
import json
import sys
import pdb
infile = open(sys.argv[1])
outfile = open(sys.argv[2], 'w')
ngram = int(sys.argv[3])
def shrink_words(word_dist, lb=3): 
    law_freq_words = set()
    for w,ds in word_dist.iteritems():
        freqs = reduce(lambda x,y:x+y, map(lambda x:x[1], ds.itervalues()))
        if freqs <= lb:
            law_freq_words.add(w)
    for w in law_freq_words:
        word_dist.pop(w)
# word --> [accid:(docfreq, wordfreq)]
word_dist = {}
alldoc_dist = {}
docs_num = 0
lineid=0
for line in infile:
    line = line.decode('utf8')
    aa = line.strip().split('\t')
    if len(aa) != 4:
        pdb.set_trace()
        continue
    
    lineid += 1
    if lineid%10000 == 0:
        print "Process lineid: ", lineid
    
    #
    if lineid%500000 == 0:
        shrink_words(word_dist)
    
    #
    accids = aa[1].split(',')
    #
    for accid in accids:
        accid = int(accid)
        alldoc_dist.setdefault(accid, 0)
        alldoc_dist[accid] += 1
        docs_num += 1
    
    words = {}
    ws = aa[3].split(',')
    word_poses = [x.split(' ') for x in ws ]
    la = 0
    for wi in xrange(len(word_poses)):
        w,p = word_poses[wi]
        try:
            # pos: x: 
            if p[0] == u'u' or p[0] == u'x':
                la = 0
                continue
            words.setdefault(w, 0)
            words[w] += 1
            la += 1
            if la >= 2 and ngram >= 2:
                w2 = "%s_%s" % (word_poses[wi-1][0], w)
                words.setdefault(w2, 0)
                words[w2] += 1
                if la >= 3 and ngram >= 3:
                    w3 = "%s_%s" % (word_poses[wi-2][0], w2)
                    words.setdefault(w3, 0)
                    words[w3] += 1
        except:
            pdb.set_trace()
    
    for word,freq in words.iteritems():
        dd = word_dist.setdefault(word, {})
        for accid in accids:
            freqs = dd.setdefault(int(accid), [0,0])
            freqs[0] += 1
            freqs[1] += freq
            
print "Process END. total docs: %s; total lines: %s" % (docs_num, lineid)
shrink_words(word_dist, lb=10)
alldoc_lst = alldoc_dist.items()
alldoc_lst.sort(cmp=lambda x,y:cmp(x[1], y[1]), reverse=True)
dist_str = "\t".join(["%s,%s,%s" % (x,y,y) for x,y in alldoc_lst])
alldoc_str = "ALL\t0,%d,%d\t%s\n" % (docs_num, docs_num, dist_str)
outfile.write(alldoc_str)
for word,dist in word_dist.iteritems():
    wlst = dist.items()
    wlst.sort(cmp=lambda x,y:cmp(x[1][0],y[1][0]), reverse=True)
    df = reduce(lambda x,y:x+y, map(lambda x:x[1][0], wlst))
    wf = reduce(lambda x,y:x+y, map(lambda x:x[1][1], wlst))
    dist_str = "\t".join(["%s,%s,%s" %(x,y[0],y[1]) for x,y in wlst])
    allword_str = "%s\t0,%d,%d\t%s\n" % (word, df, wf, dist_str) 
    outfile.write(allword_str.encode('utf8'))

