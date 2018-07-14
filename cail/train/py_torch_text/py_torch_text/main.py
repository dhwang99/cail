#coding:utf8
from config import opt
import models
import os
import tqdm
from data_set import CailDataSet, BatchDataLoader 
import torch as t
import time
import fire
import torchnet as tnt
from torch.utils import data
from torch.autograd import Variable
from utils.visualize import Visualizer
import numpy as np
import json

import pdb

vis = Visualizer(opt.env, port=5900)


def hook():pass

def get_score(cm):
    def f1_fun(ma):
        tp, fp, fn, tn = ma[0:4]
        if tp == 0:
            if fp == 0 and fn == 0:
                return 0, 0, 1.
            return 0, 0, 0.

        p = tp * 1./(tp + fp)
        r = tp * 1./(tp + fn)

        return p, r, 2.*p*r/(p+r)

    micro_array = np.zeros(4, dtype='int')
    details = []
    for ci in cm:
        ri = f1_fun(ci)
        details.append((ci.tolist(), ri))
        micro_array += ci

    macro_f1 = 0.

    micro_rst = f1_fun(micro_array)
    macro_f1 = np.sum(map(lambda x:x[1][2], details)) / len(details) 
    micro_f1 = micro_rst[2]

    return (micro_f1, macro_f1, (micro_f1 + macro_f1)/2.), (micro_array.tolist(), micro_rst), details

def add_confuse_matrix(cm, predict_scores, true_labels):
    threshold = opt.logit_threshold if opt.logit_threshold else 0.6 #高于这个值的，保留前3
    remain_threshold = 0.8  #只要高于这个就保留
    
    scores, ids = predict_scores.data.topk(5, dim=1)
    scores = scores.cpu()
    ids = ids.cpu()
    for i in range(ids.shape[0]):
        #top 1 always selected
        p_set = set()
        c_ids = ids[i]
        c_scores = scores[i]
        p_set.add(c_ids[0].item())
        if c_scores[1].item() > threshold:
            #perhaps get more
            p_set.add(c_ids[1].item())
            if c_scores[2].item() > remain_threshold:
                #very confidence
                p_set.add(c_ids[2].item())
                if c_scores[3].item() > remain_threshold:
                    p_set.add(c_ids[3].item())

        t_set = set(true_labels[i])
        for t in t_set:
            if t in p_set:
                cm[t][0] += 1   #tp
            else:
                cm[t][2] += 1   #nf

        for p in p_set:
            if p not in t_set:
                cm[p][1] += 1   #np

    return cm


def val(model, val_batch_iter):
    '''
    计算模型在验证集上的分数
    '''
    #confuse matrix
    cm = np.zeros((opt.num_classes , 4), dtype='int')

    model.eval()
    ii = 0
    for data, labels_npy, labels in val_batch_iter:
        ii += 1
        #data, label = t.stack([t.from_numpy(b) for b in batch], 0)
        data_t = t.from_numpy(data).long()
        label_t = t.from_numpy(labels_npy).long()
        with t.no_grad():  
            data_t,label_t = Variable(data_t.cuda()),\
                         Variable(label_t.cuda())
        score = model(data_t)
        # !TODO: 优化此处代码
        #       1. append
        #       2. for循环
        #       3. topk 代替sort

        add_confuse_matrix(cm, score, labels)
        del score

    model.train()
    
    return cm 

def main(**kwargs):
    '''
    训练入口
    '''

    opt.parse(kwargs,print_=False)

    model = getattr(models,opt.model)(opt).cuda()
    if opt.model_path:
        model.load(opt.model_path)
    print(model)

    opt.parse(kwargs,print_=True)
    path = ''

    vis.reinit(opt.env, port=5900)
    pre_loss=1.0
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  

    train_dataset = CailDataSet([opt.train_data_path, opt.validation_data_path], opt, augument=opt.augument)

    test_dataset = CailDataSet([opt.test_data_path], opt, augument=False)

    train_dataloader = BatchDataLoader(train_dataset, opt, catdist_fname='train_catedist.lst')
    test_dataloader = BatchDataLoader(test_dataset, opt, catdist_fname='test_catedist.lst')

    optimizer = model.get_optimizer(lr,lr2,opt.weight_decay)
    loss_meter = tnt.meter.AverageValueMeter()
    best_score = 0

    ii = 0
    for epoch in range(opt.max_epoch):
        batch_iters = train_dataloader.batch_iter_by_catdist()
        loss_meter.reset()
        #for ii, batch in tqdm.tqdm(enumerate(batch_iters)):
        cm = np.zeros((opt.num_classes, 4), dtype='int')
        for data, label_npy, labels in batch_iters:
            ii += 1
            #data, label = t.stack([t.from_numpy(b) for b in batch], 0)
            data = t.from_numpy(data).long()
            label = t.from_numpy(label_npy).long()
            # 训练 更新参数
            data_cuda,label_cuda = Variable(data.cuda()), Variable(label.cuda())
            optimizer.zero_grad()
            score = model(data_cuda)
            #loss = loss_function(score, label_cuda)
            loss = loss_function(score, label_cuda.float())
            loss_meter.add(loss.data.item())
            loss.backward()
            optimizer.step()

            add_confuse_matrix(cm, score, labels)

            if ii%opt.plot_every == 0:
                ### 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                
                f1, mi_detail, ma_details = get_score(cm)
                vstr = 'mi_f1:%.4f, ma_f1:%.4f, avg_f1:%.4f' %(f1[0], f1[1], f1[2])
                print "Train Result:  epoch: %d; ii: %d; rst: %s" %(epoch, ii, vstr)
                #vis.vis.text(vstr,win='tmp')
                vis.plot('avg_f1', f1)
                
                #eval()
                vis.plot('loss', loss_meter.value()[0])
                k = t.randperm(label.size(0))[0]
                cm = np.zeros((opt.num_classes, 4), dtype='int')

            if ii%opt.decay_every == 0:   
                # 计算在验证集上的分数,并相对应的调整学习率
                del loss
                val_batch_iters = test_dataloader.batch_iter()
                test_cm = val(model, val_batch_iters)
                f1, mi_detail, ma_details = get_score(test_cm)
                vstr = 'TestResult: mi_f1:%.4f, ma_f1:%.4f, avg_f1:%.4f' %(f1[0], f1[1], f1[2])
                print vstr

                rst_json = {'f1':f1, 'mi_detail': mi_detail, 'ma_details': ma_details}
                
                rfilename = 'result/%s_%s' % (epoch, ii)
                json.dump(rst_json, fp=open(rfilename, 'w'), indent=2)

                vis.log({' epoch:':epoch,' lr: ':lr, 'lr2': lr2, 'mi_f1': f1[0], 'ma_f1': f1[1], 'avg_f1':f1[2], \
                         'p':mi_detail[1][0], 'r':mi_detail[1][1], 'loss':loss_meter.value()[0]})

                vis.plot('test_avg_f1', f1)
                
                if f1[2]>best_score:
                    best_score =f1[2]
                    best_path = model.save(name = str(f1[2]),new=True)
                
                '''
                if f1[2] < best_score:
                    #model.load(best_path,change_opt=False)
                    lr = lr * opt.lr_decay
                    lr2= 2e-4 if lr2==0 else  lr2*0.8
                    optimizer = model.get_optimizer(lr,lr2,0)                        
                '''
                
                pre_loss = loss_meter.value()[0]
                loss_meter.reset()
                cm = np.zeros((opt.num_classes, 4), dtype='int')

if __name__=="__main__":
    fire.Fire()
