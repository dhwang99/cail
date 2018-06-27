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

class PredictorLocal(object):
    def __init__(self, tfidf_model, law_model, accu_model, time_model):
        self.tfidf = tfidf_model
        self.law = law_model
        self.accu = accu_model
        self.time = time_model

    def predict_law(self, vec):
        y = self.law.predict(vec)
        return [y[0]+1]
    
    def predict_accu(self, vec):
        y = self.accu.predict(vec)
        return [y[0]+1]
    
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


def cut_text(alltext, seg_method='jieba'):
    count = 0    
    cut = thulac.thulac(seg_only = True)
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        if seg_method == 'jieba':
            arr = jieba.cut(text, cut_all=False, HMM=False)
            train_text.append(' '.join(arr))
        else:
            train_text.append(cut.fast_cut(text, text = True))
    
    return train_text


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


def read_trainData(path):
    fin = open(path, 'r', encoding = 'utf8')
    
    alltext = []
    
    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    while line:
        aa = line.strip().split('\t')
        if len(aa) != 4:
            continue

        accids = aa[1].split(',')
        law = aa[2].split(',')
        time = 0
        
        ws = aa[3].split(',')
        word_poses = [x.split(' ') for x in ws ]
        words = ' '.join(map(lambda x:x[0], word_poses))

        alltext.append(words)
        accu_label.append(int(accids[0]))
        law_label.append(int(law[0]))
        time_label.append(time)

        '''
        d = json.loads(line)
        alltext.append(d['fact'])
        accu_label.append(data.getlabel(d, 'accu'))
        law_label.append(data.getlabel(d, 'law'))
        time_label.append(data.getlabel(d, 'time'))
        '''
        line = fin.readline()

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

    print('reading...')
    dim = int(sys.argv[1])
    seg_method = sys.argv[2]
    ngram = int(sys.argv[3])
    min_df = int(sys.argv[4])
    train_fname = sys.argv[5]

    #train
    train_data, accu_label, law_label, time_label = read_trainData(train_fname)
    print('cut text...')
    #train_data = cut_text(alltext, seg_method)
    print('train tfidf...')
    tfidf = train_tfidf(train_data, dim, ngram, min_df)
    
    vec = tfidf.transform(train_data)
    
    print('accu SVC')
    accu = train_SVC(vec, accu_label)
    print('law SVC')
    law = train_SVC(vec, law_label)
    print('time SVC')
    time = None
    #time = train_SVC(vec, time_label)
   
    #test
    test_filename='data_test.json'
    pred = PredictorLocal(tfidf, law, accu, time)
    all_test_predicts = []
    all_test_metas = []
    test_f = open(test_filename)
    for line in test_f:
        js = json.loads(line)
        text = js["fact"]
        meta = js["meta"]

        ans = pred.predict(text)
        all_test_predicts.append(ans)
        all_test_metas.append(meta)
    
    #metrics
    judge = Judger("../baseline/accu.txt", "../baseline/law.txt")
    result = judge.test2(all_test_metas, all_test_predicts)
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
