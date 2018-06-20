#encoding: utf8

import json
import sys
import pdb

infname=sys.argv[1]
outprefix=sys.argv[2]

accusations = {}
relevant_articles = {}
all_accus = {}


with open(infname) as fp:
    for line in fp:
        aa = line.split('\t')
        if len(aa) != 2:
            continue

        lineid = int(aa[0])
        jsobj = json.loads(aa[1])
        meta = jsobj[u'meta']

        accs = meta['accusation']
        for acc in accs:
            accusations.setdefault(acc, 0)
            accusations[acc]  += 1

        sa = ' '.join(accs)
        all_accus.setdefault(sa, 0)
        all_accus[sa] += 1
        
        arcs = meta['relevant_articles']
        for arc in arcs:
            relevant_articles.setdefault(arc, 0)
            relevant_articles[arc] += 1


accs = accusations.items()
accs.sort(cmp=lambda x,y:cmp(x[1],y[1]), reverse=True)
#pdb.set_trace()
total = reduce(lambda x,y:x+y, map(lambda x:x[1],  accs))

of = open(outprefix + "_accurstions.lst", 'w')
base=0.
for acc in accs:
    cu = acc[1] * 100./total
    base += cu
    ss = "%s\t%s\t%.05f%%\t%.05f%%\n" % (acc[0], acc[1], cu, base)
    of.write(ss.encode('utf8'))
of.close()

of = open(outprefix + "_all_accurstions.lst", 'w')
all_accs = all_accus.items()
all_accs.sort(cmp=lambda x,y:cmp(x[1],y[1]), reverse=True)
total = reduce(lambda x,y:x+y, map(lambda x:x[1],  all_accs))
base=0.

for acc in all_accs:
    cu = acc[1] * 100./total
    base += cu
    ss = "%s\t%s\t%.05f%%\t%.05f%%\n" % (acc[0], acc[1], cu, base)
    of.write(ss.encode('utf8'))
of.close()

of = open(outprefix + "_relevant_articles.lst", 'w')
arcs = relevant_articles.items()
arcs.sort(cmp=lambda x,y:cmp(x[1],y[1]), reverse=True)
total = reduce(lambda x,y:x+y, map(lambda x:x[1],  arcs))
base=0.

for arc in arcs:
    cu = acc[1] * 100./total
    base += cu
    ss = "%s\t%s\t%.05f%%\t%.05f%%\n" % (acc[0], acc[1], cu, base)
    of.write(ss.encode('utf8'))
of.close()
