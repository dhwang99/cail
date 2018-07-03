#encoding: utf8

'''
支持py3语法
'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from pprint import pprint
import six 
from time import time

import multiprocessing as mp
mp.set_start_method('forkserver')

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import PredefinedSplit
from sklearn.metrics import f1_score, make_scorer

import json
from predictor import data
import sys
from judger import Judger
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle
import pdb



'''
    logfilename="../log/train.log
    logger = create_logger(logfilename)

    logger = logging.getLogger()
    infostr = "tagid: %s or method:%s is not digit." % (tagid, method)
    logger.info(infostr)
    logger.debug(infostr)
'''
def create_logger(logfilename=None, logName=None) :
    import logging,logging.handlers
    logger = logging.getLogger(logName)
    infohdlr = None
    if logfilename ==None:
        infohdlr = logging.StreamHandler(sys.stdout)
    else:
        infohdlr = logging.FileHandler(logfilename)
    infohdlr.setLevel(logging.INFO)
    #detail

    formatter = logging.Formatter('%(asctime)s %(levelname)6s  %(threadName)-12s %(filename)-10s  %(lineno)4d:%(funcName)16s|| %(message)s')
    formatter = logging.Formatter('%(asctime)s %(levelname)6s %(message)s')

    infohdlr.setFormatter(formatter)

    logger.addHandler(infohdlr)

    logger.setLevel(logging.DEBUG)
    return logger


def line2sample(line):
    ans = {}
    arr = line.strip().split('\t')
    if len(arr) != 5:
        return None

    ans['accusation'] = [int(x) for x in arr[1].split(',')]
    ans['articles'] = [int(x) for x in arr[2].split(',')]
    ans['imprisonment'] = int(arr[3])

    ws = arr[4].split(',')
    #words = [x.split(' ')[0] for x in ws ]
    word_poses = [x.split(' ') for x in ws ]
    words = [x for x,p in word_poses if p[0] != u'u' and p[0] != u'x']
    text = ' '.join(words)

    return (ans, text)

class PredictorLocal(object):
    def __init__(self, tfidf_model, accu_model, law_model, time_model):
        self.tfidf = tfidf_model
        self.accu = accu_model
        self.law = law_model
        self.time = time_model

    def predict_law(self, vec):
        y = [-1]
        if self.law != None:
            y = self.law.predict(vec)
        return [y[0]]
    
    def predict_accu(self, vec):
        y = [-1]
        if self.accu != None:
            y = self.accu.predict(vec)
        return [y[0]]
    
    def predict_time(self, vec):
        if self.time == None:
            return -2

        y = self.time.predict(vec)[0]
        
        #返回每一个罪名区间的中位数
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6
        
    def predict(self, content, seg_method='jieba'):
        fact = ''
        
        vec = self.tfidf.transform([fact])
        ans = {}

        ans['accusation'] = self.predict_accu(vec)
        ans['articles'] = self.predict_law(vec)
        ans['imprisonment'] = self.predict_time(vec)
        return ans

    def predict_file(self, test_filename):
        all_test_predicts = []
        all_test_labels = []
        test_f = open(test_filename, 'rb')
        for line in test_f:
            line = line.decode('utf8')
            sample = line2sample(line)
            if None == sample:
                continue

            label, text = sample
            vec = self.tfidf.transform([text])
            ans = {}
            ans['accusation'] = self.predict_accu(vec)
            ans['articles'] = self.predict_law(vec)
            ans['imprisonment'] = self.predict_time(vec)

            all_test_predicts.append(ans)
            all_test_labels.append(label)
        
        return all_test_labels, all_test_predicts


def f1_func(ground_truth, predictions):
    micro_f1 = f1_score(ground_truth, predictions, average='micro')
    macro_f1 = f1_score(ground_truth, predictions, average='macro')

    return (micro_f1 + macro_f1) / 2.

def train_tfidf(train_data, dim=5000, ngram=3, min_df=5):
    ngram_range = (1,3)
    if ngram == 1:
        ngram_range = (1,1)
    elif ngram == 2:
        ngram_range = (1,2)

    tfidf = TfidfVectorizer(
            min_df = min_df,
            max_features = dim,
            ngram_range = ngram_range,
            use_idf = 1,
            smooth_idf = 1
            )
    tfidf.fit(train_data)
    
    return tfidf

def gettime(time):
    #将刑期用分类模型来做
    v = time
    if v == -2:
        return 0
    if v == -1:
        return 1
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    else:
        return 8

'''
只反回了第1个label
'''
def read_trainData(path, all_text, accu_label, law_label, time_label):
    fin = open(path, 'rb')

    linid = 0
    for line in fin:
        linid += 1
        if linid%5000 == 0:
            print("Process train file at: %d" % linid)
        
        line = line.decode('utf8')
        sample = line2sample(line)
        if None == sample:
            continue

        label, text = sample
        all_text.append(text)
        accu_label.append(label['accusation'][0])
        law_label.append(label['articles'][0])
        time_label.append(gettime(label['imprisonment']))


    fin.close()

    return linid


'''
返回的label是一个列表, 同时初始化预测结果列表
'''
def read_testData(path):
    all_text = []
    all_test_predicts = []
    all_test_labels = []

    test_f = open(test_filename, 'rb')
    for line in test_f:
        line = line.decode('utf8')
        sample = line2sample(line)
        if None == sample:
            continue

        label, text = sample
        ans = {}
        ans['accusation'] = [-1]
        ans['articles'] = [-1]
        ans['imprisonment'] = -3

        all_text.append(text)
        all_test_predicts.append(ans)
        all_test_labels.append(label)

    test_f.close()

    return all_text, all_test_labels, all_test_predicts 


def train_SVC(vec, label):
    SVC = LinearSVC(class_weight='balanced')

    #SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC

def train_Adaboost_SVC(vec, label):
    SVC = LinearSVC()
    bdt_real = AdaBoostClassifier(
            base_estimator=SVC,
            n_estimators=10,
            algorithm='SAMME',
            learning_rate=1)

    #SVC = LinearSVC()
    bdt_real.fit(vec, label)
    return bdt_real 


if __name__ == '__main__':
    import logging
    logfilename="train.log"
    root_logger = create_logger()
    logger = create_logger(logfilename)

    dim = int(sys.argv[1])
    seg_method = sys.argv[2]
    ngram = int(sys.argv[3])
    min_df = int(sys.argv[4])
    train_fname = sys.argv[5]
    test_filename = sys.argv[6]
    val_filename = sys.argv[7]

    #train
    print('reading train data...')
    sys.stdout.flush()
    
    #所有训练文本
    train_data = []

    accu_label = []
    law_label = []
    time_label = []
    print('reading train data.')
    train_docs_num = read_trainData(train_fname, train_data, accu_label, law_label, time_label)
    
    print('reading val data.')
    val_docs_num = read_trainData(val_filename, train_data, accu_label, law_label, time_label)


    print('reading test data...')
    sys.stdout.flush()
    test_data, test_labels, test_predicts = read_testData(test_filename)
    
    law_model = None
    accu_model = None
    time_model = None

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'tfidf__max_df': [0.75, 1.],   #过滤了几十个词
        'tfidf__min_df': (5, 10, 20, 50),
        'tfidf__max_features': (200000, 400000, 600000),
        'tfidf__ngram_range': [(1, 3)],  # unigrams or trigrams,  use trigrams
        'tfidf__use_idf': [1],
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000005, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),

    }

    test_fold = np.zeros((train_docs_num + val_docs_num), dtype='int')
    test_fold[:train_docs_num] = -1
    ps = PredefinedSplit(test_fold = test_fold)
    
    t0 = time()

    my_score = make_scorer(f1_func, greater_is_better=True)
    grid_search = GridSearchCV(pipeline, parameters, cv=ps, n_jobs=-1, verbose=1, scoring=my_score)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    grid_search.fit(train_data, accu_label)

    print("done in %0.3fs" % (time() - t0))

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


    #test
    _predicts = grid_search.best_estimator_.predict(test_data)
    for i in xrange(len(_predicts)):
        lab = _predicts[i]
        test_predicts[i]['accusation'] = [lab]

    judge = Judger("../baseline/accu.txt", "../baseline/law.txt")
    result = judge.test2(test_labels, test_predicts)
    print(result)
    rst = judge.get_score(result)

    print(rst)
    rstr = "ACCU:(%.4f, %.4f, %.4f); LAW:(%.4f, %.4f, %.4f) TIME: %.4f"% \
            (rst[0][0], rst[0][1], rst[0][2], rst[1][0], rst[1][1], rst[1][2], rst[2]) 

    sinfo = 'Prog:%s TrainFile:%s Seg:%s DIM:%s NGRAM:%d RESULT: %s' % (sys.argv[0], train_fname, seg_method, dim, ngram, rstr)
    logger.info(sinfo)

