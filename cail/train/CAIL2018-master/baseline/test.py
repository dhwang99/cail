from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from predictor import data
from svm import read_trainData
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac
from sklearn import metrics

import pdb
import json

def predict_file(filename, fon):
    alltext, accu_label, law_label, time_label = read_trainData(filename)
    
    accu_pred = []
    law_pred = []
    time_pred = []
    fo = open(fon, 'w')

    for i in range(len(alltext)):
        text = alltext[i]
        ans = pred.predict([text])[0]
        pa,pl, pt = ans['accusation'][0],ans['articles'][0],ans['imprisonment']
        accu_pred.append(pa)
        law_pred.append(pl)
        time_pred.append(pt)

        ta,tl,tt = accu_label[i], law_label[i], time_label[i]

        print("real: %s, %s, %s; predict: %s, %s, %s" % (ta,tl,tt, pa,pl,pt))
        
        ans['accusation'] = [int(pa)]
        ans['articles'] = [int(pl)]

        print(json.dumps(ans), file=fo)
    
    fo.close()
    accu_f1_micro = metrics.f1_score(accu_label, accu_pred, average='micro')
    accu_f1_macro = metrics.f1_score(accu_label, accu_pred, average='macro')

    law_f1_micro = metrics.f1_score(law_label, law_pred, average='micro')
    law_f1_macro = metrics.f1_score(law_label, law_pred, average='macro')

    print("accu: %.4f, %.4f; law: %.4f, %.4f" % (accu_f1_micro, accu_f1_macro, law_f1_micro, law_f1_macro))

if __name__ == '__main__':
    print('reading...')

    filenames = ['src/data_test.json']
    outfilenames =['rst/data_test.json']

    for i in range(0, len(filenames)):
        fn = filenames[i]
        fon = outfilenames[i]

        print("Predict File %s:" % fn)
        pr = predict_file(fn, fon)
        print("\n")
