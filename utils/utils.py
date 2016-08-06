#!/usr/bin/env python
# coding=utf-8
__author__='dengbinbin'
import sys,os
import codecs,re
import jieba
import cPickle,pickle
import numpy as np
from pypinyin import *
#import keras
from keras.preprocessing import sequence
from keras.models import  Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Lambda,Merge,Flatten
from keras.layers.embeddings import Embedding
from keras.layers import  merge,Input
from keras.layers.convolutional import Convolution1D,Convolution2D,MaxPooling2D,MaxPooling1D
from keras.constraints import nonneg,maxnorm
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
from path import *
from plot import *
reload(sys)
sys.setdefaultencoding('utf-8')


def max_1d(X):
    return K.max(X,axis=1)
def load_userdict():
    jieba.load_userdict("./data/dic/skills.txt")
    #jieba.load_userdict("./data/dic/jobnames.txt")
    jieba.load_userdict("./data/dic/benefits.txt")
    jieba.load_userdict("./data/dic/citydic.txt")
    jieba.load_userdict("./data/dic/degrees.txt")
    jieba.load_userdict("./data/dic/firm.txt")
    jieba.load_userdict("./data/dic/jobposition.txt")
    jieba.load_userdict("./data/dic/majordic.txt")

def find_best_weight(save_dir):
    best_weight = ""
    best_val = 0.0
    for weight in os.listdir(save_dir):
        if weight.find("hdf5")!=-1:
            if float(weight.split("-")[1].split(".hdf5")[0])>best_val:
                best_val = float(weight.split("-")[1].split(".hdf5")[0])
                best_weight = weight
    print "the best weight is :{0}".format(best_weight)
    return  best_weight

def dump_labelCounts():
    #读入数据
    import cPickle
    x = cPickle.load(open(data_path+"51job_pinyin.p","rb"))
    revs, W, W2, word_idx_map, vocab,idx_word_map = x[0], x[1], x[2], x[3], x[4],x[5]
    #将label作为key，label的个数作为value加入到label_counts字典中去
    label_counts = {}
    for i in range(len(revs)):
        if revs[i]['label'] in label_counts.keys():
            label_counts[revs[i]['label']]+=1
        else:
            label_counts[revs[i]['label']] =1
    file_label = file(data_path+'label_counts.pkl','w')
    cPickle.dump(label_counts,file_label,protocol=cPickle.HIGHEST_PROTOCOL)    
dump_labelCounts()



def num2label(num):
    labelList = ["jdFrom","pubTime","incName","incScale","incType","incIndustry","incWorkLoc","incUrl",\
"incStage","incAliasName","investIns","incContactInfo","incCity","zipCode","incContactName","incIntro",\
"jobType","jobPosition","jobCate","jobSalary","jobWorkAge","jobDiploma","jobNum","jobWorkCity","jobWorkLoc",\
  "jobWelfare","age","jobEndTime","email","gender","jobMajorList","jobDesc",\
"keyWords","isFullTime","jdRemedy","posType","urgent","holidayWelfare","livingWelfare","salaryCombine",\
"socialWelfare","trafficWelfare","jobDepartment","jobReport","jobReportDetail","jobSubSize","language","overSea"]
    
    return str(labelList[num])



def get_idx_from_sent(sent, word_idx_map, max_l=51, k=200,pin = True):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent

    if pin == True:
        for word in words:
            if pinyin(word)[0][0] in word_idx_map:
                x.append(word_idx_map[pinyin(word)[0][0]])
            else:
            #print (u"{0} not find".format(word[0]))
                x.append(word_idx_map[u'unknown'])
    else:
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
            else:
                # print (u"{0} not find".format(word[0]))
                x.append(word_idx_map[u'unknown'])
    if len(x)>=max_l:
        x = x[0:max_l]
    return x

def pad_sent(sent, word_idx_map, max_l=51, k=200, filter_h=5):
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    x.extend(sent)
    if len(x)>=max_l+2*pad:
        x = x[0:max_l+2*pad]
        return x
    else:
        while len(x) < max_l+2*pad:
            x.append(0)
        return x

def make_idx_data_cv(revs, word_idx_map, max_l=51, k=200, filter_h=5,pin = True,config = None):
    """
    Transforms sentences into a 2-d matrix.
    """
    data = []
    sents = []
    concatenate_length = (config.total_sents-1)/2

    # get all sents without padding
    for index in range(len(revs)):
        sents.append(get_idx_from_sent(revs[index]['text'], word_idx_map,max_l, k,pin))
    #pad the sents
    for index in range(len(revs)):

        new_sent = []
        left_pad = abs(concatenate_length-index)
        right_pad = abs(concatenate_length - (len(revs)-1-index))

        if left_pad>concatenate_length:
            left_pad = 0
        if right_pad>concatenate_length:
            right_pad = 0
        if left_pad!=0:
            for i in range(left_pad):
                new_sent+=[0 for i in range(config.max_len)]
        if left_pad!=concatenate_length:
            for i in range(concatenate_length-left_pad,0,-1):
                new_sent+=sents[index-i]
        new_sent+=sents[index]
        if right_pad!=concatenate_length:
            for i in range(1,concatenate_length-right_pad+1):
                new_sent+=sents[index+i]
        if right_pad!=0:
            for i in range(right_pad):
                new_sent+=[0 for i in range(config.max_len)]

        sent = pad_sent(new_sent, word_idx_map,max_l*config.total_sents, k,filter_h)
        sent.append(int(revs[index]['label']))
        data.append(sent)
    # data = np.array(data,dtype="float32")
    split = int(len(data)*0.8)
    train_data = data[:split]
    test_data = data[split:]
    train_data = np.array(train_data,dtype="int32")
    test_data = np.array(test_data, dtype="int32")
    return [train_data, test_data]





def dump_pkl(obj, f):
    if isinstance(f, str) or isinstance(f, unicode):
        return pickle.dump(obj, open(f, "wb"))
    else:
        return pickle.dump(obj, f)


def count_sentence(fname):
    file_count = codecs.open(fname, 'rb', 'utf-8')
    sentence_len = {}
    sentence_label = {}
    for line in file_count.readlines():
        line = line.strip().split(u' ')
        length = len(line) - 1
        label = line[-1]
        if label == u'' or length == 0:
            pass
        else:
            if label in sentence_label.keys():
                sentence_label[label] += 1
            else:
                sentence_label[label] = 1
            if length in sentence_len.keys():
                sentence_len[length] += 1
            else:
                sentence_len[length] = 1

    return sentence_len, sentence_label







# if __name__=="__main__":
    
