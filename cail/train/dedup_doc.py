#encoding: utf8

import hashlib
import sys
import pdb

testfile=open(sys.argv[1])
trainfile=open(sys.argv[2])
dstfile = open(sys.argv[3], 'w')

dup_set = set()

for line in testfile:
    arr = line.strip().split('\t')
    if len(arr) != 4:
        continue

    digest = hashlib.md5(arr[3]).hexdigest()
    #pdb.set_trace()
    dup_set.add(digest)

for line in trainfile:
    arr = line.strip().split('\t')
    if len(arr) != 4:
        continue

    digest = hashlib.md5(arr[3]).hexdigest()
    if digest not in dup_set:
        dstfile.write(line)

testfile.close()
trainfile.close()

dstfile.close()
