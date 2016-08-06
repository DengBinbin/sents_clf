import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_figure(history, save_path):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, 'b', val_loss, 'r')
    plt.xlabel('train_loss: blue   val_loss: red      epoch')
    plt.ylabel('loss')
    plt.title('loss figure')
    plt.savefig(save_path)

def plot_probs(pros,labels,save_path):
    x,y = len(pros),len(pros[0])
    f, axarr = plt.subplots(x, y,sharey=True)
    for i in range(x):
        for j in range(y):
            if i==x-1 and j==y-1:
                continue
            df = pd.DataFrame({"pro":map(float,pros[i][j])},index=labels[i][j])
            ax = axarr[i][j]
            title = "{0}".format(labels[i][j][0])
            df.plot(kind="bar",ax=ax,ylim=[0,1],legend="",color=["r","y","y","y"],rot=0,fontsize=20,figsize=(20,14))#,figsize=(10,7)
            _x = ax.get_xlim()[-1]/2.0
            _y = ax.get_ylim()[-1]*0.6
            ax.text(_x,_y,title,ha='center', va='bottom',fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)