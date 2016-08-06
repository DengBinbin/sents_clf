
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter
import time
import os
import sys
from utils import *
from plot import *
 

def sparse_categorical_accuracy(y_true,y_pred,eps=10**-8):

    return -np.mean(np.log(y_pred[np.arange(y_pred.shape[0]), y_true]+eps))
 
# def evalueate_result_0(y_true,y_pred,verbose=0):
#     if not isinstance(y_true,list):
#         y_true = y_true.flatten().tolist()
#     label_size = len(set(y_true))
#     label_set = set(y_true)  
#     y_pred = np.array(y_pred)   

#     labels = np.argmax(y_pred,axis=1)
#     sample_size = len(y_true)
#     micro_acc = np.mean(labels==y_true)
#     cm = confusion_matrix(y_true=y_true,y_pred=labels,labels=np.arange(48))
#     cm_norm1 = cm.astype("float")/np.maximum(cm.sum(0),1)
#     accs = map(lambda x:cm_norm1[x,x],label_set)
#     cm_norm2 = cm.astype("float")/np.maximum(cm.sum(1),1)
#     recalls = map(lambda x:cm_norm2[x,x],label_set)
#     print cm.shape
#     macro_acc = sum(accs)*1.0/len(accs)
#     macro_recall = sum(recalls)*1.0/len(recalls)

#     if verbose>0:
#         print("\n***Test on {:,} samples***".format(sample_size))
#         cnt = Counter(y_true)
#         label,max_size = cnt.most_common(1)[0]
#         print("most predict: {:.4f}".format(max_size*1.0/sample_size))
#         print("micro_acc is: {:.4f} ;  macro_acc is {:.4f} ".format(micro_acc,macro_acc))
#         print("macro_recall is {:.4f} ".format(macro_recall))
#         print("")
#         if verbose>1:
#             label_index = []
#             for i in label_set:
#                 label_index.append(num2label(int(i)))
#             df1= pd.DataFrame(accs,columns =["acc"],index =label_index)

#             df2= pd.DataFrame(recalls,columns =["recall"],index =label_index)
            
#             print("\n***Acc on every label***")
#             print df1
            
#             for label in label_set:
#                 if (cm_norm1[:,label]).max()<1:
                     
#                      print("the true label is {0},the top4 probs are:{1}\nthe top4 labels are:{2}".format(num2label(int(label)),\
#                        map( lambda x:"{:.2f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]),[num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]]))

#             # print (df1.describe())
#             print("\n***Recall on every label***")
#             print(df2)
#     res = {"accs":accs,"recalls":recalls,"cm":cm,"acc":micro_acc,"macro_acc":macro_acc,"macro_recall":macro_recall}
#     return res




def evaluate_result(y_true,y_pred,verbose=0):
    if isinstance(y_true,list):
        y_true = np.array(y_true)
    y_true = y_true.reshape((-1,))

    assert len(y_true) == len(y_pred)

    labels = sorted(set(y_true.flatten().tolist()))
    label_size = len(set(labels))
    label_set = set(y_true) 
    pred_labels = np.argmax(y_pred,axis=1)
    
    loss = sparse_categorical_accuracy(y_true,y_pred)

    sample_size = len(y_true)

    micro_acc = np.mean(pred_labels==y_true)
    
    cm = confusion_matrix(y_true=y_true,y_pred=pred_labels,labels=np.arange(48))
    
    cm_norm1 = cm.astype("float")/np.maximum(cm.sum(0),1)
    accs = map(lambda x:cm_norm1[x,x],label_set)
    cm_norm2 = cm.astype("float")/np.maximum(cm.sum(1),1)
    recalls = map(lambda x:cm_norm2[x,x],label_set)
    macro_acc = sum(accs)*1.0/len(accs)
    macro_recall = sum(recalls)*1.0/len(recalls)
    if verbose>0:
        print("\n***Test on {:,} samples***".format(sample_size))
        cnt = Counter(y_true)
        label,max_size = cnt.most_common(1)[0]
        print("loss {:.4f}".format(loss))
        print("most predict: {:.4f}".format(max_size*1.0/sample_size))
        print("micro_acc is: {:.4f} ;  macro_acc is {:.4f} ".format(micro_acc,macro_acc))
        print("macro_recall is {:.4f} ".format(macro_recall))
        print("")
        error_label = 0
        probs  = [[] for i in range(9)]
        labels=[[] for i in range(9)]
        if verbose>1:
            label_index = []
            for i in label_set:
                label_index.append(num2label(int(i)))
            df1= pd.DataFrame(accs,columns =["acc"],index =label_index)
            df2= pd.DataFrame(recalls,columns =["recall"],index =label_index)
            print("\n***Acc on every label***")
            for label in label_set:
                tol = 0.98
                if (cm_norm1[:,label]).max()<tol:       
                    error_label+=1
                    print("the true label is {0},the top4 probs are:{1}\nthe top4 labels are:{2}".format(num2label(int(label)),\
                       map( lambda x:"{:.2f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]),[num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]]))
                    if num2label(label).find("Name")!=-1:
                        labels[0].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[0].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("Loc")!=-1:
                        labels[1].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[1].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("City")!=-1:
                        labels[2].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[2].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("Desc")!=-1 or num2label(label).find("Intro")!=-1 :
                        labels[3].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[3].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("Position")!=-1 or num2label(label).find("Cate")!=-1 :
                        labels[4].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[4].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("jobType")!=-1 or num2label(label).find("keyWords")!=-1 :
                        labels[5].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[5].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("gender")!=-1 or num2label(label).find("incContact")!=-1 :
                        labels[6].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[6].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    elif  num2label(label).find("incUrl")!=-1 or num2label(label).find("email")!=-1 :
                        labels[7].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[7].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
                    else:
                        labels[8].append([num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]])
                        probs[8].append(map( lambda x:"{:.3f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]))
            save_path = "/home/dengbinbin/text_clf/png/ana_wrongAns.png"
            plot_probs(probs, labels,save_path)
            print df1
            print(df1.describe())
            print("\n***Recall on every label***")
            print df2
            print(df2.describe())
        print "the number of label which acc are lower than {0}:{1}".format(tol,error_label)
    res = {"accs":accs,"recalls":recalls,"cm":cm,"micro_acc":micro_acc,
            "macro_acc":macro_acc,"macro_recall":macro_recall,
            "loss":loss}
    return res


def evaluate_model(model,test_data=None,test_number =None,bs=256,verbose=1):
 
    t0 = time.time()
    if verbose>0:
        print("loading test samples...")
    test_data, test_label = test_data[:test_number, :-1], test_data[:test_number, -1]
    print ("test data shape:{0},{1}".format(test_data.shape, test_label.shape))
    if verbose>0:
        print("loading test samples finish. Cost {:.2f} seconds".format(time.time()-t0))
    
    t0 = time.time()
    if verbose>0:
        print("predicting test samples...")
    y_pred = model.predict(test_data,batch_size=bs,verbose=1)
    if verbose>0:
        print("predicting test samples finished. Cost  {:.2f} seconds".format(time.time()-t0))
    
    if verbose>0:
        print("\n Evaluate test result...")

    res = evaluate_result(test_label,y_pred,verbose=verbose)

    return y_pred,res


if __name__=="__main__":
    _ = evalueate_result_0([0,1],[[40,10],[10,30]],verbose=2)
