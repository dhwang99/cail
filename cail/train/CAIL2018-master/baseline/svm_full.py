from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from predictor import data
import sys
sys.path.append("..")
from judger.judger import Judger
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac
import jieba
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
        y = self.law.predict(vec)
        return [y[0]]
    
    def predict_accu(self, vec):
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
        if seg_method == 'jieba':
            a = jieba.cut(content, cut_all=False, HMM=False)
            fact = ' '.join(a)
        else:
            fact = self.cut.fast_cut(content, text = True)
        
        vec = self.tfidf.transform([fact])
        ans = {}

        ans['accusation'] = self.predict_accu(vec)
        ans['articles'] = self.predict_law(vec)
        ans['imprisonment'] = self.predict_time(vec)
        return ans

    def predict_file(self, test_filename):
        all_test_predicts = []
        all_test_labels = []
        test_f = open(test_filename, encoding='utf8')
        for line in test_f:
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


def train_tfidf(train_data, dim=5000, ngram=3, min_df=5):
    ngram_range = (1,3)
    if ngram == 1:
        ngram_range = (1,1)
    elif ngram == 2:
        ngram_range = (1,2)

    tfidf = TFIDF(
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

def read_trainData(path):
    fin = open(path, 'r', encoding = 'utf8')
    
    alltext = []
    
    accu_label = []
    law_label = []
    time_label = []

    linid = 0
    for line in fin:
        linid += 1
        if linid%5000 == 0:
            print("Process train file at: %d" % linid)

        sample = line2sample(line)
        if None == sample:
            continue

        label, text = sample
        alltext.append(text)
        accu_label.append(label['accusation'][0])
        law_label.append(label['articles'][0])
        time_label.append(gettime(label['imprisonment']))


    fin.close()

    return alltext, accu_label, law_label, time_label


def train_SVC(vec, label):
    #SVC = LinearSVC(class_weight='balanced')

    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


if __name__ == '__main__':
    import logging
    jieba.setLogLevel(logging.CRITICAL)
    logfilename="train.log"
    root_logger = create_logger()
    logger = create_logger(logfilename)

    dim = int(sys.argv[1])
    seg_method = sys.argv[2]
    ngram = int(sys.argv[3])
    min_df = int(sys.argv[4])
    train_fname = sys.argv[5]
    test_filename = sys.argv[6]

    #train
    print('reading train data...')
    train_data, accu_label, law_label, time_label = read_trainData(train_fname)
    print('train tfidf...')
    tfidf = train_tfidf(train_data, dim, ngram, min_df)
    
    vec = tfidf.transform(train_data)
    
    print('accu SVC')
    accu = train_SVC(vec, accu_label)
    print('law SVC')
    law = train_SVC(vec, law_label)
    print('time SVC')
    time = train_SVC(vec, time_label)
   
    #test
    print('predict')
    predictor = PredictorLocal(tfidf, accu, law, time)
    test_label, test_predict = predictor.predict_file(test_filename)
    
    #metrics
    judge = Judger("../baseline/accu.txt", "../baseline/law.txt")
    result = judge.test2(test_label, test_predict)
    print(result)
    rst = judge.get_score(result)

    print(rst)
    rstr = "ACCU:(%.4f, %.4f, %.4f); LAW:(%.4f, %.4f, %.4f) TIME: %.4f"% \
            (rst[0][0], rst[0][1], rst[0][2], rst[1][0], rst[1][1], rst[1][2], rst[2]) 

    sinfo = 'Seg:%s DIM:%s NGRAM:%d RESULT: %s' % (seg_method, dim, ngram, rstr)
    logger.info(sinfo)

    print('begin test model:')
    print('saving model')
    joblib.dump(tfidf, 'predictor/model/tfidf.model')
    joblib.dump(accu, 'predictor/model/accu.model')
    joblib.dump(law, 'predictor/model/law.model')
    joblib.dump(time, 'predictor/model/time.model')
