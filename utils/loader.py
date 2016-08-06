#coding=utf8
from path import *

import os
import sys
import pickle
from keras.preprocessing import sequence
from keras.models import model_from_yaml
import json
from collections import Counter
import numpy as np
import pdb
import h5py


def best_weights(name):
    path = name2path(name,"model","_best_weight.h5")
    return path
    
def save_model(model):
    yaml_string = model.to_yaml()
    with  open(os.path.join(model_path,model.name+".yaml"),"wb") as fo:
        fo.write(yaml_string)


def load_model_weights(model,mode):
    if mode == "best":
        w_name = best_weights(model.name)
    elif isinstance(mode,int):
        w_name = iter_weights(model.name,model)
    else:
        w_name = mode
    if os.path.isfile(w_name):
        model.load_weights(w_name)
    else:
        print("warning! {} doesn't exist".format(w_name))
    return model
    
def load_model(model_name,init_w=None):
    with open(os.path.join(model_path,model_name+".yaml"),"rb") as fi:
        yaml_string = fi.read()
        model = model_from_yaml(yaml_string)
        return model

    
def  load_pkl(f):
    if isinstance(f,str) or isinstance(f,unicode):
        return pickle.load(open(f))
    else:
        return pickle.load(f)

def  dump_pkl(obj,f):
    if isinstance(f,str) or isinstance(f,unicode):
        return pickle.dump(obj,open(f,"wb"))
    else:
        return pickle.dump(obj,f)