#!/usr/bin/env python
#encoding=utf8

import time
import sys
import pdb
import math

import fcntl
import signal
from multiprocessing import Pool

import esm

def search_keywords(outf, filename):
    outfile = open(outf, 'a+')
    infile = open(filename, 'r')

    for line in infile:
        arr = line.split('\t')
        global index
        rst = index.query(arr[2])
        if len(rst) > 0:
            #print arr[0], rst
            r = ','.join([i[1] for i in rst])
            s = '%s\t%s' % (arr[0], r)
            fcntl.flock(outfile,fcntl.LOCK_EX)
            outfile.flush()
            outfile.write("%s\n" % s)
            outfile.flush()
            fcntl.flock(outfile,fcntl.LOCK_UN)

def sigint_handler(signum, frame):
    print 'Signal handler called with signal', signum
    raise IOError("Couldn't open device!")
    g_pool.terminate()

global g_pool
index = esm.Index()
filename='dirty.txt'
with open(filename, 'r') as f:
    for i in f.readlines():
        index.enter(i.strip())
    index.fix()

if __name__ == '__main__':
    g_pool = Pool(processes=32)
    signal.signal(signal.SIGINT, sigint_handler)
    R = []

    outfile = sys.argv[1]
    files = sys.argv[2:]
    
    '''
    #for test
    search_keywords(outfile, files[0])
    sys.exit()
    '''

    for  fname in files:
        r = g_pool.apply_async(search_keywords, (outfile, fname,))
        R.append(r)

    for r in R:
        r.get()
