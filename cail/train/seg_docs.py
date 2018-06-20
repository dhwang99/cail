#encoding: utf8

import jieba
import jieba.posseg as pseg
import json
import sys
import re
import pdb

def load_acc_names(acc_fname='conf/accu.txt'):
    n2k = {}
    acc_id=1
    with open(acc_fname) as f:
        for line in f:
            line = line.strip().decode('utf8')
            n2k[line] = str(acc_id)
            acc_id += 1
    return n2k

user_dict_fname = 'conf/name_dict.lst'

def gen_criminals_names(infilename):
    #criminals
    names = {} 
    infile = open(infilename)
    for line in infile:
        aa = line.strip().split('\t')
        if len(aa) != 2:
            continue
        jsobj = json.loads(aa[1])
        meta = jsobj[u'meta']
        criminals = meta['criminals']
        for cr in criminals:
            names.setdefault(cr, 0)
            names[cr] += 1
    
    ofile = open(user_dict_fname, 'w')
    for name,freq in names.iteritems():
        ss = "%s %s nr\n" % (name, freq)
        ofile.write(ss.encode('utf8'))
    ofile.close()

if len(sys.argv) == 2:
    gen_criminals_names(sys.argv[1])
    sys.exit(0)

accname2id = load_acc_names()
jieba.enable_parallel(24)

with open(user_dict_fname) as fp:
    for line in fp:
        w,freq,pos = line.split(' ')
        if int(freq) >= 5:
            jieba.add_word(w, freq, pos)

#jieba.load_userdict(user_dict_fname)
infile = open(sys.argv[1])
outfile = open(sys.argv[2], 'w')
null_re = re.compile(u'[\r\n\t]')
for line in infile:
    aa = line.strip().split('\t')
    if len(aa) != 2:
        continue
    lineid = aa[0]
    jsobj = json.loads(aa[1])
    #fact = jsobj[u'fact'].strip().replace('\r', '').replace('\n', ' ').replace('\t', ' ')
    fact = null_re.subn(' ', jsobj[u'fact'])[0]
    meta = jsobj[u'meta']
    
    #
    accs = meta['accusation']
    accids = [accname2id[acc] for acc in accs]
    
    #
    arcs = meta['relevant_articles']
    #
    word_pos = pseg.cut(fact, hmm=False)
    ss = "%s\t%s\t%s\t%s\n" % (lineid, ','.join(accids), ','.join([str(x) for x in arcs]), ','.join([x + " " + y for x,y in word_pos if x != ' ' and x != ',']))

    ss = ss.encode('utf8')
    outfile.write(ss)

outfile.close()
