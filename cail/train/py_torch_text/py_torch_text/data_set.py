#encoding:utf-8
from torch.utils import data
import torch as t
import numpy as np
import random
from glob import glob
import json
import pdb
    
class CailDataSet(data.Dataset):
    def __init__(self,filenames, opt, augument=True):
        self.augument=augument
        self.opt = opt
        _d = np.load(filenames[0])
        self.data = _d['content_word']
        self.accu_labels = _d['accu_labels']

        for filename in filenames[1:]:
            _d = np.load(filename)
            self.data = np.concatenate((self.data, _d['content_word']), axis=0)
            self.accu_labels = np.concatenate((self.accu_labels, _d['accu_labels']), axis=0)
        
        self.len_ = len(self.data)
   
    def shuffle(self,d):
        return np.random.permutation(d.tolist())

    def labels(self):
        return self.accu_labels

    def dropout(self,d,p=0.5):
        dlen_ = len(d)
        index = np.random.choice(dlen_,int(dlen_*p))
        d[index]=0
        return d

    '''
    获取 dropout/shuffle后的训练条目和label
    '''
    def __getitem__(self, index):
        items = self.data[index]
    
        '''
        if self.augument:
            newitems = []

            for item in items:
                augument=random.random()
                if augument>0.5:
                    item = self.dropout(item,p=0.3)
                else:
                    item = self.shuffle(item)
                newitems.append(item)

            items = newitems
        '''

        labels = self.accu_labels[index]
        labels_npy = np.zeros((labels.shape[0], self.opt.num_classes), dtype='int')
        #pdb.set_trace()
        for i in range(len(labels)):
            label = labels[i]
            labels_npy[i, label] = 1

        '''
        data = t.from_numpy(item).long()
        label_tensor = t.zeros(self.label_size).scatter_(0,t.LongTensor(labels),1).long()
        return data,label_tensor
        '''
        #pdb.set_trace()
        return items, labels_npy, labels

    def __len__(self):
        return self.len_


class BatchDataLoader:
    def __init__(self, data_set, opt, catdist_fname=None):
        self.data_set = data_set
        self.sample_count = len(data_set)
        self.opt = opt

        #把数据的ID丢到各分类下。如果一个文档有多个id, 那它会被丢进多个类下
        dl = [[] for i in range(self.opt.num_classes)]
        labels = self.data_set.labels() 
        for ii in range(self.sample_count):
            label = labels[ii]
            for l in label:
                dl[l].append(ii)
        
        docid_in_cats = []
        cat_doc_dist = []
        for docs in dl:
            docid_in_cats.append(np.array(docs))
            cat_doc_dist.append(len(docs))

        self.docid_in_cats = docid_in_cats

        if catdist_fname != None:
            json.dump(cat_doc_dist, fp=open(catdist_fname, 'w'), indent=2)
        
        #加载分类样本分配表
        cat_dist = []
        with open(self.opt.cate_dist_path) as f:
            for line in f:
                arr = line.split('\t')
                cat_dist.append((int(arr[0]) - 1, float(arr[1])))

        cat_dist.sort(cmp=lambda x,y:cmp(x[1], y[1]))
        self.cat_dist = cat_dist
        self.bulk_batch_count = 3000   #每个桶的batch数
        self.bulk_sample_size =  self.bulk_batch_count * self.opt.batch_size
        if self.bulk_sample_size > self.sample_count:
            self.bulk_sample_size = self.sample_count

        #每一个epoch，桶的个数
        self.big_bulk_count =  int(np.ceil(np.float(self.sample_count)/self.bulk_sample_size))

    '''
    用于测试、评测数据的生成
    '''
    def batch_iter(self):
        cur_id = 0
        while cur_id < self.sample_count:
            yield self.data_set[cur_id:(cur_id+self.opt.batch_size)]
            cur_id += self.opt.batch_size
    
    '''
    用于训练数据的采样
    基于分类分布的比例进行数据采样。对小类上采样。对大类，则下采样了
    '''
    def batch_iter_by_catdist(self):
        #shuffle cat docs
        [np.random.shuffle(ids) for ids in self.docid_in_cats]

        for cur_bulk_id in range(self.big_bulk_count):
            docid_in_bulks = []
            cat_pos = np.zeros(self.opt.num_classes, dtype='int')

            for catid, ratio in self.cat_dist:
                all_cat_docs = self.docid_in_cats[catid]
                num = int(np.ceil(ratio * self.bulk_sample_size))

                ids = np.array([], dtype='int')
                remain_num = num  #对小类进行过采样
                while True:
                    cur_p = cat_pos[catid]
                    ids = self.docid_in_cats[catid][cur_p:(cur_p+remain_num)]
                    docid_in_bulks.extend(ids.tolist())
                    if ids.shape[0] == remain_num:
                        cat_pos[catid] = cur_p + remain_num
                        break
                    cat_pos[catid] = 0
                    remain_num -= ids.shape[0]

            np.random.shuffle(docid_in_bulks)
            rec_count = len(docid_in_bulks)
            cur_id = 0
            while cur_id < rec_count:
                ids=docid_in_bulks[cur_id:cur_id+self.opt.batch_size]
                yield self.data_set[ids]
                cur_id += self.opt.batch_size
