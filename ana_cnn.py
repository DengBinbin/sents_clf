#coding=utf-8
'''
找不同filter选中的字
'''
import sys
import os
import pickle
import keras
import pandas as pd
import time
import theano
from collections import  Counter
import pdb
import numpy as np
np.random.seed(1337)  # for reproducibility
from cnn_architecture.flatten5 import *
from keras import backend as K
from time import time
from datetime import datetime
from utils.path import *
from utils.loader import *
from utils .utils import *
from utils.evaluate import *
from config.config import *
sys.path.append("/home/dengbinbin/text_clf")

from utils.loader import *


file_path = [data_path+'51job_pinyin.pkl',data_path+'51job_hanzi.pkl',data_path+'51job_word.pkl',data_path+'51job_raw_word.pkl']
file_path = file_path[int(sys.argv[1])]
save_dir = model_path + str(datetime.now()).split('.')[0].split()[0] + '/'+file_path.split(".")[0].split("_")[-1]+'/'
config = config()
#没有打包文件时
def load_data(config):
    # 读入数据
    x = cPickle.load(open(data_path+"51job_hanzi.p", "rb"))
    revs, W, W2, word_idx_map, vocab,idx_word_map = x[0], x[1], x[2], x[3], x[4],x[5]
    print ("data loaded!")
    datasets = make_idx_data_cv(revs, word_idx_map, max_l=config.max_len, k=200, filter_h=5,pin = False, config =config)

    return datasets,len(vocab),word_idx_map,idx_word_map
if not os.path.exists(file_path):
    f = file(file_path,'w')
    datasets,vocab_size,word_idx_map,idx_word_map = load_data(config)
    cPickle.dump(datasets,f,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(vocab_size,f,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(word_idx_map,f,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(idx_word_map,f,protocol=cPickle.HIGHEST_PROTOCOL)
    print "dump complete"
#有pkl文件时候
else:
    f = file(file_path,'r')
    datasets = cPickle.load(f)
    vocab_size = cPickle.load(f)
    word_idx_map = cPickle.load(f)
    idx_word_map = cPickle.load(f)
    print "load complete"

model_name = "flatten_10"
model = load_model(model_name)
model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
##找到最佳权重然后载入
def find_best_weight():
    best_weight = ""
    best_val = 0.0
    for weight in os.listdir(save_dir):
        if weight.find("hdf5")!=-1:
            if float(weight.split("-")[1].split(".hdf5")[0])>best_val:
                best_val = float(weight.split("-")[1].split(".hdf5")[0])
                best_weight = weight
    return  best_weight
model.load_weights(save_dir+find_best_weight())
# model.load_weights('model/{0}.hdf5'.format(model_name))



def count_ngram(data,fn,ngram_values,ngram_cnts):

    print (data.shape)
    out = fn([data,1])
    idxs = map(lambda x:np.argmax(x,axis=1),out)
    nb = len(out)
    # print out[0]

    for n,_idx in zip(xrange(1,nb+1),idxs):
        '''different length of conv'''
        value = out[n-1]
        ngram_value = ngram_values[n-1]
        ngram_cnt = ngram_cnts[n-1]

        for i,idx in enumerate(_idx):
            '''
            enumerate sample
            '''
            d = data[i]
            x = value[i]
            for row,start in enumerate(idx):
                '''
                different kernel
                '''
                tp = tuple(d[start:start+n])
                ngram_cnt[row][tp] +=1
                ngram_value[row][tp] = x[start,row]

# def id2k(di,init=0):
#     return dict((w_id+init,w) for w_id,w in enumerate(sorted(di.iterkeys())))



desc_conv_names = filter(lambda x:x.find("conv")!=-1,
             map(lambda x:x.name,model.layers) )

ngram = 5
convs = map(lambda x:model.get_layer(x).get_output_at(0),desc_conv_names[:ngram])
desc_conv = K.Function([model.get_input_at(0),K.learning_phase()],convs)



# w_cnts =  load_pkl(os.path.join(data_path,"all_input_char.pkl"))


# desc_id2w = id2k(w_cnts,init=1)
# desc_id2w[0] = "_"


ngram_cnts = [[Counter() for _ in xrange(200)] for _ in xrange(ngram)]

ngram_values = [[Counter() for _ in xrange(200)] for _ in xrange(ngram)]


num = 0

print("\n Start Predicting data and count it ......")






#导入配置
desc_id2w = idx_word_map
desc_id2w[0] = '_'
config.vocab_size = vocab_size + 2
train_data,test_data = datasets[0],datasets[1]
test_label = test_data[:50000,-1]
test_data = test_data[:50000,:-1]
print test_data[0]
print (test_data.shape)
print (type(test_data[0]))
print (type(test_data[0][0]))
print(model.get_input_at(0))
print convs
print (desc_conv_names)

count_ngram(test_data,desc_conv,ngram_values,ngram_cnts)

print("\n Predict data and count it finished!!!")

limits = 10000


min_freq = 6
min_value = 0.001

#print ngram_values

print("\n delete low frequent and low activation values n-gram")

for values,cnts in zip(ngram_values,ngram_cnts):
    for value,cnt in zip(values,cnts):
        deletes = filter(lambda x:cnt[x]<min_freq,cnt.iterkeys())
        deletes.extend(filter(lambda x:value[x]<min_value,value.iterkeys()))
        for k in deletes:
            # print("\r delete {}".format(k),end="")
            del value[k]
            del cnt[k]



def most_common(cnts,limits):
    res = []
    for cnt in cnts:
        gram = map(lambda x:x.most_common(limits),cnt)
        res.append(res)
    return res

if not os.path.exists(pkl_path):
    os.makedirs(pkl_path)
dump_pkl((most_common(ngram_cnts,limits),most_common(ngram_values,limits)),os.path.join(pkl_path,"conv_n_gram_count.pkl"))

topN = 300

print("\n Start Writing top {} n-grams to file......".format(topN))


for i,cnts,values in zip(xrange(ngram),ngram_cnts,ngram_values):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fo = open(os.path.join(result_path,"{}_gram".format(i+1)),"wb")
    for j,cnt,value in zip(xrange(len(cnts)),cnts,values):
        fo.write("Filter_{} : ".format(j))
        grams = []
        for tp,n in cnt.most_common(topN):
            # print(tp)
            # pdb.set_trace()
            gram = "-".join(map(lambda x:desc_id2w[x],tp))
            gram += "({},{:.4f})".format(n,value[tp])
            grams.append(gram)
        fo.write(u"\t".join(grams).encode("utf8"))
        fo.write("\n")

        grams = []
        for tp,n in value.most_common(topN):
            gram = "-".join(map(lambda x:desc_id2w[x],tp))
            gram += "({:.4f},{})".format(n,cnt[tp])
            grams.append(gram)
        fo.write(u"\t".join(grams).encode("utf8"))
        fo.write("\n")

print("\n Writing top {} n-grams to file finished!!!".format(topN))



