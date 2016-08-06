import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter
from utils import *
def evalueate_result(y_true,y_pred,verbose=0):
    if not isinstance(y_true,list):
        y_true = y_true.flatten().tolist()
    label_size = len(set(y_true))
    label_set = set(y_true)  
    y_pred = np.array(y_pred)   

    labels = np.argmax(y_pred,axis=1)
    sample_size = len(y_true)
    micro_acc = np.mean(labels==y_true)
    cm = confusion_matrix(y_true=y_true,y_pred=labels,labels=np.arange(48))
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
        print("most predict: {:.4f}".format(max_size*1.0/sample_size))
        print("micro_acc is: {:.4f} ;  macro_acc is {:.4f} ".format(micro_acc,macro_acc))
        print("macro_recall is {:.4f} ".format(macro_recall))
        print("")
        if verbose>1:
            label_index = []
            for i in label_set:
                label_index.append(num2label(int(i)))

            df1= pd.DataFrame(accs,columns =["acc"],index =label_index)

            df2= pd.DataFrame(recalls,columns =["recall"],index =label_index)
            
            print("\n***Acc on every label***")
            print df1
            
            for label in label_set:
                if (cm_norm1[:,label]).max()<1:
                     
                     print("the true label is {0},the top4 probs are:{1}\nthe top4 labels are:{2}".format(num2label(int(label)),\
                       map( lambda x:"{:.2f}".format(x),[item for item in np.sort(cm_norm1[:,label])[-1:-5:-1]]),[num2label(item) for item in np.argsort(cm_norm1[:,label])[-1:-5:-1]]))

            # print (df1.describe())
            print("\n***Recall on every label***")
            print(df2)
    res = {"accs":accs,"recalls":recalls,"cm":cm,"acc":micro_acc,"macro_acc":macro_acc,"macro_recall":macro_recall}
    return res


    
if __name__ == '__main__':

    evalueate_result([1,0],[[10,20],[30,40]],verbose=2)