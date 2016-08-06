import os


base_path =  os.path.abspath(os.path.join(__file__, os.pardir,os.pardir))

data_path = os.path.join(base_path,"data/")

model_path = os.path.join(base_path,"model/")

#model_path = os.path.join(base_path,"cnn_architecture/")


pkl_path = os.path.join(base_path,"pkl/")

result_path = os.path.join(base_path,"result/")

png_path = os.path.join(base_path,"png/")

w2v_path = os.path.join(base_path,"w2v_model/")

