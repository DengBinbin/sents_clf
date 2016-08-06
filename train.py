# coding=utf-8
import numpy as np
np.random.seed(1337)  # for reproducibility
from cnn_architecture.flatten5 import *
import time
from time import time
from datetime import datetime
from utils.path import *
from utils.loader import *
from utils .utils import *
from utils.evaluate import *
from config.config import *



def load_data(config,pin,file_path):
    # 读入数据
    file_path = file_path.split('.')[0]+".p"
    print file_path
    x = cPickle.load(open(file_path, "rb"))
    revs, W, W2, word_idx_map, vocab,idx_word_map = x[0], x[1], x[2], x[3], x[4],x[5]
    # print ("data loaded!")
    datasets = make_idx_data_cv(revs, word_idx_map, max_l=config.max_len, k=200, filter_h=5,pin = pin, config =config)

    return datasets,len(vocab),word_idx_map,idx_word_map

def train_model(save_dir,train_data,train_size):
    print (datasets[0].shape,  datasets[1].shape)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
    start_time = time.time()
    checkpointer = ModelCheckpoint(filepath=save_path, monitor='val_acc', verbose=1, save_best_only=True)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
    history = model.fit(train_data[:train_size, :-1], train_data[:train_size, -1], batch_size=128, nb_epoch=15,
                        callbacks=[checkpointer, earlyStopping], shuffle=True, verbose=0,class_weight=config.class_weight,
                        validation_split=0.2)
    '''test_loss,test_score = train_model(model,config,datasets[0][i*train_num:(i+1)*train_num],datasets[1][i*test_num:(i+1)*test_num],1)'''
    end_time = time.time()
    plot_loss_figure(history, os.path.join(save_dir,'_'.join(str(datetime.now()).split('.')[0].split())))
    print ("total time:{0}".format(end_time - start_time))


def test_model(save_dir,test_data,test_number,word_idx_map):
    ##找到最佳权重然后载入

    model.load_weights(save_dir+find_best_weight(save_dir))
    evaluate_model(model,test_data,test_number = test_number,bs=256,verbose=2)
    # test_num = 100000
    # test_data, test_label = test_data[:, :-1], test_data[:, -1]
    # print ("test data shape:{0},{1}".format(test_data.shape, test_label.shape))
    # # result是原始预测结果，result_max是最大的概率值的概率，result_index是最大概率取值时候的index值
    # result = model.predict(test_data[:test_num], batch_size=10, verbose=0)
    # result_argsort = [np.argsort(i) for i in result]
    # result_sort = [np.sort(i) for i in result]
    # score = evalueate_result(test_label[:test_num],result,verbose=2)
    
if __name__=='__main__':
    #file_path有三种选择，分别是拼音、汉字、词
    file_path = [data_path+'51job_pinyin.pkl',data_path+'51job_hanzi.pkl',data_path+'51job_word.pkl',data_path+'51job_raw_word.pkl',data_path+'51job_short.pkl']
    file_path = file_path[int(sys.argv[1])]
    config = config()
    save_dir = model_path + str(datetime.now()).split('.')[0].split()[0] + '/'+file_path.split(".")[0].split("_")[-1]+'/' # 模型保存在当天对应的目录中
    #打包数据方便以后调用
    if not os.path.exists(file_path):
        f = file(file_path,'w')
        if file_path.find("pin")!=-1:   
            pin=True
        else:
            pin=False
        print "pinyin:{0}".format(pin)
        datasets,vocab_size,word_idx_map,idx_word_map = load_data(config,pin,file_path)
        cPickle.dump(datasets,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(vocab_size,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(word_idx_map,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(idx_word_map,f,protocol=cPickle.HIGHEST_PROTOCOL)
        print "dump complete"
    #l载入已经打包好的数据
    else:
        f = file(file_path,'r')
        datasets = cPickle.load(f)
        vocab_size = cPickle.load(f)
        word_idx_map = cPickle.load(f)
        idx_word_map = cPickle.load(f)
        print "load complete"
    #导入配置
    config.vocab_size = vocab_size + 2
    
    print config.vocab_size
    # #构建CNN模型
    model = build_model(config)
    # save_model(model)
    model_name = "flatten_10"
    save_model(model)
    
    # #训练模型
    train_number =1000000 
    test_number = 100000
    train_data,test_data = datasets[0],datasets[1]
    train_model(save_dir,train_data,train_number)
    # #测试模型
    test_model(save_dir,test_data,test_number,word_idx_map)
