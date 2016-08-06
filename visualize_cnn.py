from __future__ import print_function
import sys
import os
import pickle
import keras
from keras import backend as K
import pandas as pd
import time
import numpy as np
from numpy.linalg import norm
import theano
from collections import  Counter
import pdb
sys.path.append("/home/bookchan/data/job_clf/")

from utils import load_pkl,name2path,dump_pkl
from config.path import *
from utils.loader import Reader
from utils import  load_model,load_model_weights
from evaluate import evaluate_result



model_name = "pos_flat_cnn"
model = load_model( model_name)
model.compile(loss="mse",
                  optimizer="adam",
                  metrics=['accuracy'])
load_model_weights(model,"best")



layers = model.layers


conv1 =  layers[3]
cnn_input = conv1.get_input_at(0)
pred = model.get_output_at(0)
target = K.placeholder(ndim=len(model.get_output_shape_at(0)),name="target")
loss = K.mean(model.loss_functions[0](pred,target))


w_grad = K.gradients(loss,cnn_input)
grad = K.function([cnn_input,target,K.learning_phase()],w_grad)




emb_layer = model.get_layer("pos_w2v")
vs = emb_layer.get_output_at(0)
emb = K.function([model.get_input_at(0),K.learning_phase()],vs)

x = emb([[[1,2,3]],0])

print("Test emb f([[1]): {}".format(x.shape))

g = grad([x,[[1]*88],0])

print("Test grad norm(f([[1]])): {}".format(norm(g,ord=2,axis=-1)))





