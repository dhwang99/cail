#encoding: utf8

'''
支持py3语法
'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from pprint import pprint
#import six 

import json


'''
    参考 grid_search 模式
    parameters = {
        'tfidf__max_df': [0.75],   #过滤了几十个词
        'tfidf__min_df': [5],
        'tfidf__max_features': [200000],
        #'tfidf__max_features': (50000, 100000, 200000, 400000),
        'tfidf__ngram_range': [(1, 3)],  # unigrams or trigrams,  use trigrams
        'tfidf__use_idf': [1],
        'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': [1000],
        'clf__C': (0.1, 0.5, 1.0, 2.0),
        'clf__class_weight':('balanced', None),
        'clf__solver': ('sag', 'liblinear'),
        #'clf__n_iter': (10, 50, 80),
    }
'''

'''
把参数列表 转为任务表的形式。
转为如下任务参数列表：
[任务名, {参数名，参数值}]
'''
def parse_params(parameters):
    all_plist = [[]]
    for key, values in parameters.items():
        cur_plist = list(all_plist)
        all_plist = []

        for val in values:
            for p in cur_plist:
                p_new = list(p)
                p_new.append((key,val))
                all_plist.append(p_new)
    
    params = []
    for plist in all_plist:
        cdict = {}
        for p,v in plist:
            step_name, param_name = p.split('__')
            param_dict = cdict.setdefault(step_name, {})
            param_dict[param_name] = v 

        params.append(cdict)

    return params


if __name__ == '__main__':
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'tfidf__max_df': [0.75],   #过滤了几十个词
        'tfidf__min_df': [5],
        'tfidf__max_features': [200000],
        #'tfidf__max_features': (50000, 100000, 200000, 400000),
        'tfidf__ngram_range': [(1, 3)],  # unigrams or trigrams,  use trigrams
        'tfidf__use_idf': [1],
        'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': [1000],
        'clf__C': (0.1, 0.5, 1.0, 2.0),
        'clf__class_weight':('balanced', None),
        'clf__solver': ('sag', 'liblinear'),
        #'clf__cv':ps,
        #'clf__n_iter': (10, 50, 80),
    }

    
    param_list = parse_params(parameters)
    dumpstr = json.dumps(param_list)
    fp = open('params.lst', 'w')
    json.dump(param_list, fp, indent=2)

