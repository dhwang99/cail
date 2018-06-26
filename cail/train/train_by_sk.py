# encoding: utf8
import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np  
import cPickle as pickle  
import pdb
import logging

'''
    logfilename="../log/train.log
    logger = create_logger(logfilename)

    logger = logging.getLogger()
    infostr = "tagid: %s or method:%s is not digit." % (tagid, method)
    logger.info(infostr)
    logger.debug(infostr)
'''
def create_logger(logfilename , logName=None) :
    import logging,logging.handlers
    logger = logging.getLogger(logName)
    infohdlr = logging.StreamHandler(sys.stdout)
    infohdlr.setLevel(logging.INFO)
    #detail
    debughdlr = logging.StreamHandler(sys.stdout)
    debughdlr.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)6s  %(threadName)-12s %(filename)-10s  %(lineno)4d:%(funcName)16s|| %(message)s')

    infohdlr.setFormatter(formatter)
    debughdlr.setFormatter(formatter)

    logger.addHandler(infohdlr)
    #logger.addHandler(debughdlr)

    logger.setLevel(logging.DEBUG)
    return logger
  
reload(sys) 
sys.setdefaultencoding('utf8')  

# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2', multi_class='ovr', class_weight='balanced', n_jobs=1, solver='linear')
    #model = LogisticRegression(penalty='l2', multi_class='ovr', class_weight='balanced', n_jobs=16, solver='lbfgs')
    model.fit(train_x, train_y)  
    return model 

 
# SGDClassifier 
def sgd_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier(alpha=0.0001, average=False, class_weight='balanced', epsilon=0.1,
                          eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                          learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
                          n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
                          shuffle=True, tol=None, verbose=0, warm_start=False)

    model.fit(train_x, train_y)
    return model
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=16, n_jobs=16, class_weight='balanced')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=500)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC 
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  

# SVM Classifier using cross validation  
def svm_cross_validation2(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    # kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    model = SVC(kernel='rbf', probability=True)  
    #param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'gamma': np.logspace(-3, 0)}  
    param_grid = {'C': np.logspace(-3, 3, 20), 'gamma': np.logspace(-3, 0, 20)}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 24, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print para, val  

    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    #param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'gamma': np.logspace(-3, 0)}  
    param_grid = {'C': np.logspace(-3, 3, 20), 'gamma': np.logspace(-3, 0, 20)}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 24, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val

    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def read_data(train_fname, test_fname, fea_num=10000):
    from sklearn.datasets import load_svmlight_file
    #fea_num = 33800 * 50
    train_x, train_y, = load_svmlight_file(train_fname)
    test_x, test_y  = load_svmlight_file(test_fname, n_features=train_x.shape[1])

    print 'read train data, X_shape:', train_x.shape
    print 'read test data, X_shape:', test_x.shape

    x_fea_size = train_x.shape[1]
    if x_fea_size > fea_num:
        train_x = train_x[:, :fea_num]
        test_x = test_x[:, :fea_num]

    print 'real train data, X_shape:', train_x.shape
    print 'real test data, X_shape:', test_x.shape
    #pdb.set_trace()

    return train_x, train_y, test_x, test_y
     

if __name__ == '__main__':  
    thresh = 0.5  
    model_save_file = None  
    model_save = {}  

    logfilename="train.log"
    logger = create_logger(logfilename)
      
    test_classifiers = ['NB', 'LR', 'RF', 'DT', 'GBDT','SVM', 'SVMCV', 'KNN']
    test_classifiers = ['NB', 'RF', 'DT', 'GBDT', 'LR', 'KNN']
    test_classifiers = ['NB', 'SGDLR', 'DT', 'RF', 'GBDT']
    classifiers = {'NB':naive_bayes_classifier,
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,
                'SVMCV':svm_cross_validation,  
                'SVMCV2':svm_cross_validation2,  
                'SGDLR': sgd_regression_classifier,
                 'GBDT':gradient_boosting_classifier  
    }

    train_fname, test_fname = sys.argv[1:3]
    fea_num = int(sys.argv[3])
    fs_method = sys.argv[4]
      
    sinfo = 'reading training and testing data...'  
    logger.info(sinfo)

    train_x, train_y, test_x, test_y = read_data(train_fname, test_fname, fea_num)  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    num_test_true = np.where(test_y == 1)[0][1]
    is_binary_class = (len(np.unique(train_y)) == 2)  
    sinfo = 'training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)  
    logger.info(sinfo)

    sys.stdout.flush()

    for classifier in test_classifiers:  
        sinfo =  '******************* %s ********************' % classifier  
        logger.info(sinfo)

        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        use_time = time.time() - start_time

        stat_info = 'classifier: %s training took time: %fs' % (classifier, use_time)  
        logger.info(stat_info)

        predict = model.predict(test_x)  

        if model_save_file != None:  
            model_save[classifier] = model  
        precision = 0.0
        recall = 0.0
        if is_binary_class:  
            precision = metrics.precision_score(test_y, predict)  
            recall = metrics.recall_score(test_y, predict)  
            print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)  
        accuracy = metrics.accuracy_score(test_y, predict)
        f1_micro = metrics.f1_score(test_y, predict, average='micro')  
        f1_macro = metrics.f1_score(test_y, predict, average='macro')  

        stat_info = "TRAIN RESULT: %s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.4f\t%.4f\t%.4f" % \
                (train_fname, classifier, fs_method, num_train, num_test, num_feat, use_time, accuracy, f1_micro, f1_macro) 

        logger.info(stat_info)
  
    if model_save_file != None:  
        pickle.dump(model_save, open(model_save_file, 'wb'))  
