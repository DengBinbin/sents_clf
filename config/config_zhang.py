#coding=utf-8
import sys
sys.path.append('../')
from utils.path import *
import cPickle
import numpy as np
reload(sys)

class config_zhang():
    def __init__(self):
        config_zhang.model_name = "cnn_desc_zhang"
        
        config_zhang.word_dim = 400
        config_zhang.label_size = 48
        #config.label_dim = 100
        #config.sent_len = 300
        config_zhang.batch_size = 128
        config_zhang.nb_epoch = 10
        config_zhang.max_len = 123
        config_zhang.word_dims = 400
        config_zhang.nb_filters = [256,256,256,256,256,256]
        config_zhang.filter_lengths = [7,7,3,3,3,3]
        config_zhang.hidden_dims = 1024