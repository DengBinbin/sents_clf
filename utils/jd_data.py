# coding: utf-8
#读入数据
import cPickle
from path import *
x = cPickle.load(open("./data/51job.p","rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]


# In[30]:

#将label作为key，label的个数作为value加入到label_counts字典中去
label_counts = {}
for i in range(len(revs)):
    if revs[i]['label'] in label_counts.keys():
        label_counts[revs[i]['label']]+=1
    else:
        label_counts[revs[i]['label']] =1
import cPickle
#label_counts = [label_counts]
file_label = file('./label_counts.pkl','w')
cPickle.dump(label_counts,file_label,protocol=cPickle.HIGHEST_PROTOCOL)    


# In[31]:

file_label = file('./label_counts.pkl','r')
labels = cPickle.load(file_label)

for cid,label in enumerate(labels):   
    print cid,label,labels[label]


# In[32]:

import pandas as pd
get_ipython().magic(u'matplotlib inline')


# In[33]:

sort_labels = sorted(labels,key=lambda x:labels[x])
values = map(lambda x:labels[x],sort_labels)
df = pd.DataFrame(values,index=sort_labels)
import numpy
print numpy.sort([int (i) for i in  sort_labels])
df.plot(kind="bar")


# In[ ]:

# long_sents  = []
# short_sents = []
# for i in range(len(revs)):
#     if revs[i]['label'] ==u'15' or revs[i]['label'] ==u'31':
#         long_sents.append(revs[i]['num_words']) 
#     else:
#         short_sents.append(revs[i]['num_words'])


# In[34]:

#给定一个列表，输出按照index排序好的值的频数
import pandas as pd
def count_words(sents):
    sents = pd.DataFrame(sents)
    sents.columns = ['words']
    counts = sents['words'].value_counts()
    return counts.sort_index()


# In[35]:

#将label作为key，句子的词数作为value加入到label_counts字典中去
label_counts = {}
for i in range(len(revs)):
    if revs[i]['label'] in label_counts.keys():
        label_counts[revs[i]['label']] .extend( [revs[i]['num_words']])
    else:
        label_counts[revs[i]['label']]  = []
        label_counts[revs[i]['label']] .extend( [revs[i]['num_words']])


# In[36]:

#统计不同label的字长分布情况，每一个label输出其字长的分布情况
import numpy as np
keys = [int(i)for i in label_counts.keys()]
keys =  np.sort(keys)
keys = [unicode(i) for i in keys]
for label in keys:
    print "label",label
    print count_words(label_counts[label])


# In[48]:

#查看不同label的文本具体是些什么，用来检测解析出来的结果是否准确
count = 0 
length = []
for i in range(len(revs)):
    if revs[i]['label']==u'12' and revs[i]['num_words']>=1:
        count+=1
        length.append(revs[i]['num_words'])
        print revs[i]['text']
print "total:",count
length = np.array(length)
np.mean(length)


# In[46]:




# In[187]:

#从51job.p中读取数据，载入模型，然后将数据处理称为我们想要的格式
from utils import *
import cPickle
x = cPickle.load(open("./data/51job.p", "rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
print (len(word_idx_map), len(vocab))
print ("data loaded!")

config = config()
config.total_sents = 3
config.max_desc_len = 40
config.vocab_size = len(vocab) + 2
print (config.vocab_size)

# 读取之前best_model.hdf5保存的训练参数
model = build_model(config)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
model.load_weights('model/best_clf_model.hdf5')
datasets = make_idx_data_cv(revs, word_idx_map, 0, max_l=config.max_desc_len,k=400, filter_h=5)


# In[188]:

#读入test_data，然后运行载入好的模型对其进行预测，保存下来三个值，result_softmax是softmax之后的48维向量，result_index是预测的类别，result_prob
#是以多大的概率做出的预测
test_data = datasets[1]
#test_num = len(test_data)
test_num = 1000
test_data = test_data[:test_num]
test_data,test_label = test_data[:,:-1],test_data[:,-1]
print ("test data shape:{0},{1}".format(test_data.shape,test_label.shape))
result_softmax = model.predict(test_data[:test_num],batch_size = 10,verbose = 0)
result_index = [np.argmax(i) for i in result_softmax]
result_index = np.array(result_index)
result_prob = [np.max(i) for i in result_softmax]


# In[100]:

test_data[1].shape


# In[189]:

#对正确预测情况和错误预测情况分别绘制箱型图
def plot_box(result,title):
    import pandas as pd
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set(alpha=0.2)
    plt.plot()
    plt.title(title)
    df = pd.DataFrame(result)
    df.columns = ['prob']
    df.prob.plot(kind = 'box')
    plt.show()


# In[190]:

#统计对的样本和错的样本分别是有多大的概率做出的预测
length = len(result_prob)
error_count = 0
result_right = []
result_wrong = []
for i in range(length):
    if result_index[i]!=test_label[i]:
        error_count+=1
        result_wrong.append(result_prob[i])
        print("the true label is:{0} the predict label is :{1}".format(num2label(int(test_label[i])), num2label(result_index[i])))
        print("the result is {0}".format(result_softmax[i]))
    else:
        result_right.append(result_prob[i])
error_rate = 1.0*error_count/test_num 
plot_box(result_right,'the right predict')
plot_box(result_wrong,'the wrong predict')


# In[103]:

#统计大概率做出的预测和小概率做出的预测的准确率
length = len(result_prob)
big_prob = 0
small_prob = 0
error_big = 0
error_small = 0
for i in range(length):
    if result_prob[i] >0.9:
        big_prob+=1
        if result_index[i]!=test_label[i]:
            error_big+=1
    else:
        small_prob +=1
        if result_index[i]!=test_label[i]:
            error_small +=1
big = 1.0*error_big/big_prob
small = 1.0*error_small/small_prob


# In[184]:

#sklearn官方绘制confusion matrix的例子
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix



y_pred = result_index
y_test = test_label
label = np.array(["jdFrom","pubTime","incName","incScale","incType","incIndustry","incLocation","incUrl",    "incStage","incAliasName","investIns","incContactInfo","incCity","incZipCode","incContactName","incIntro",    "jobType","jobPosition","jobCate","jobSalary","jobWorkAge","jobDiploma","jobNum","jobWorkCity","jobWorkLoc",    "jobWelfare","age","jobEndTime","email","gender","jobMajorList","jobDesc",    "keyWords","isFullTime","jdRemedy","posType","urgent","holidayWelfare","livingWelfare","salaryCombine",    "socialWelfare","trafficWelfare","jobDepartment","jobReport","jobReportDetail","jobSubSize","language","overSea"])

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
   
    tick_marks = np.arange(len(label))
    plt.xticks(tick_marks, label, rotation=270)
    plt.yticks(tick_marks, label)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred,labels=np.array(range(48)))
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


# In[194]:

#在sklearn 官方例子的基础上在图片上面添加了数值，这样可以精确的数据来显示分类精度
'''''compute confusion matrix 
labels.txt: contain label name. 
predict.txt: predict_label true_label 
'''  
from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plt  
import numpy as np  
#load labels.  
y_pred = result_index
y_test = test_label
labels = np.array(["jdFrom","pubTime","incName","incScale","incType","incIndustry","incLocation","incUrl",    "incStage","incAliasName","investIns","incContactInfo","incCity","incZipCode","incContactName","incIntro",    "jobType","jobPosition","jobCate","jobSalary","jobWorkAge","jobDiploma","jobNum","jobWorkCity","jobWorkLoc",    "jobWelfare","age","jobEndTime","email","gender","jobMajorList","jobDesc",    "keyWords","isFullTime","jdRemedy","posType","urgent","holidayWelfare","livingWelfare","salaryCombine",    "socialWelfare","trafficWelfare","jobDepartment","jobReport","jobReportDetail","jobSubSize","language","overSea"])


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
   
    tick_marks = np.arange(len(label))
    plt.xticks(tick_marks, label, rotation=270)
    plt.yticks(tick_marks, label)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):  
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)  
#     plt.title(title)  
#     plt.colorbar()  
#     tick_marks = np.array(range(len(labels))) + 0.5  
#     xlocations = np.array(range(len(labels)))  
#     plt.xticks(xlocations, labels, rotation=90)  
#     plt.yticks(xlocations, labels)  
#     plt.ylabel('True label')  
#     plt.xlabel('Predicted label')  
cm = confusion_matrix(y_test, y_pred,labels=np.array(range(48)))  
np.set_printoptions(precision=2)  
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]  

plt.figure(figsize=(12,8), dpi=120)  

ind_array = np.arange(len(labels))  
x, y = np.meshgrid(ind_array, ind_array)  
for x_val, y_val in zip(x.flatten(), y.flatten()):  
    c = cm_normalized[y_val][x_val]  
    if (c > 0.01):  
        plt.text(x_val, y_val, "%0.3f" %(c,), color='red', fontsize=7, va='center', ha='center')  
#offset the tick  
plt.gca().set_xticks(tick_marks, minor=True)  
plt.gca().set_yticks(tick_marks, minor=True)  
plt.gca().xaxis.set_ticks_position('none')  
plt.gca().yaxis.set_ticks_position('none')  
plt.grid(True, which='minor', linestyle='-')  
plt.gcf().subplots_adjust(bottom=0.15)  
  
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')  
#show confusion matrix  
plt.show() 


# In[177]:

from IPython.display   import   SVG
import theano
from keras.utils.visualize_util import  model_to_dot
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
#SVG(model_to_dot(model,show_shapes=True).create(prog='dot',format='svg'))


# In[176]:

model.summary()


# In[156]:



