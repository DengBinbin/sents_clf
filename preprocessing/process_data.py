# coding=utf-8
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
sys.path.append("../")
import codecs,gensim
import matplotlib.pyplot as plt
import pandas as pd
from pypinyin import pinyin
from utils.path import *
def build_data_cv(data_folder, cv=10, clean_string=False,level = None):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    jd_file = data_folder
    vocab = defaultdict(float)
    with codecs.open(jd_file, "rb",'utf-8') as f:
        for line in f:
            rev = []
            rev.append(line.strip())

            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            label = orig_rev.split(' ')[-1]

            if len(label)==1:
                orig_rev = orig_rev[:-2]
            elif len(label)==2:
                orig_rev = orig_rev[:-3]
            num_words = len(orig_rev)
            new_rev = u''
            # for word in orig_rev:
            #     new_rev+=pinyin(word)[0][0]
            # if num_words> 10:
            #     flag = False
            # else:
            #     flag = True
            if level=="pinyin":
                words = set([pinyin(i)[0][0] for i in orig_rev])
            elif level=="hanzi" or level=="word" or level=="raw_word":
                words = set([i for i in orig_rev])
            else:
                raise("input is wrong")
            for word in words:
                vocab[word] += 1
                datum  = {"label":label,
                      "text": orig_rev,
                      "num_words": num_words,
                      "split": np.random.randint(0,cv)}
            # print datum
            if datum["num_words"]!=0:
                revs.append(datum)
    
    return revs, vocab

def get_W(word_vecs, k=400):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """

    vocab_size = len(word_vecs)
    word_idx_map = dict()
    idx_word_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        
        word_idx_map[word] = i
        idx_word_map[i] = word
        i += 1
    W[-1] = np.random.uniform(-0.25,0.25,k)
    word_idx_map[u'unknown'] = i
    idx_word_map[i] = u'unknown'

    return W, word_idx_map,idx_word_map

def load_bin_vec(vocab):

    word_vecs = {}
    model_word = gensim.models.Word2Vec.load(w2v_path+'wiki_word_200.model')
    model_hanzi = gensim.models.Word2Vec.load(w2v_path+'wiki_hanzi_200.model')

    #binary_len = np.dtype('float32').itemsize * 200
    for word in vocab:
        if word == u'':
            pass
        elif word in model_word.index2word:
            word_vecs[word] = np.fromstring(model_word[word], dtype='float32')
        elif word in model_hanzi.index2word:
            word_vecs[word] = np.fromstring(model_hanzi[word], dtype='float32')
        else:
            pass
    return word_vecs
def add_unknown_words(word_vecs, vocab, min_df=1, k=400):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """

    for word in vocab:

        if word not in word_vecs and vocab[word] >= min_df:

            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def plotpie(data):
    hw = pd.DataFrame(data)
    hw.columns = ['freq']
    count = hw['freq'].value_counts()
    save = []
    sum = 0
    for i in range(10):
        for j in range(1, 11):
            index = i * 10 + j
            if index in count.index:
                sum += count[index]

        save.append(sum)
        sum = 0
    for i in range(1, 10):
        for j in range(1, 101):
            index = i * 100 + j
            if index in count.index:
                sum += count[index]

        save.append(sum)
        sum = 0
    huitu = pd.DataFrame(save)
    huitu.index = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100',
                   '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000']
    huitu.plot.pie(autopct='%.2f', fontsize=10, subplots=True, figsize=(16, 8))
    plt.show()
    import time
    time.sleep(100)

if __name__=="__main__":
    '''
    51job.p存了5个东西，revs是存放了句子信息的列表，W[i]是第i个词的词向量，word_idx_map[word]是word的序号,
    W2和W类似，只不过是随机的向量，vocab是单词表
    '''
    #w2v_file = './model/wiki_word'
    file_name = sys.argv[1]
    data_folder = data_path+file_name
    print "loading data...",
    level = str(sys.argv[2])
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=False,level = level)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    #plotpie(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(vocab)

    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    k = 200
    add_unknown_words(w2v, vocab,min_df=1,k=k)
    print "after add unknown words,the words are" +str(len(w2v))
    W, word_idx_map,idx_word_map = get_W(w2v,k)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab,min_df=1,k=k)
    print "after add unknown words,the words are" +str(len(w2v))
    W2, _,_1_ = get_W(rand_vecs,k)
    cPickle.dump([revs, W, W2, word_idx_map, vocab,idx_word_map], open(data_path+file_name.split('_')[0]+"_{0}.p".format(level), "wb"))
    print "dataset created!"
 

