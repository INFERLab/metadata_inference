import pandas as pd
import numpy as np
from scipy import stats
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns

from IPython.display import set_matplotlib_formats, display, HTML
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300


def plot_box_over_sites(Val,savename = 'temp.pdf',
                        title = 'box plot',
                        xlabel='weighted F1 score',
                        ylabel='classifier',
                        size=(4,3),
                        vert=False
                        ):
    ax = Val.plot(kind='box',vert=vert,figsize=size,title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(savename,format='pdf', bbox_inches='tight', pad_inches=0)
    
def plot_df_line_dot(df, savename='temp.pdf',
                     title = 'box plot',
                     xlabel='weighted F1 score',
                     ylabel='classifier',
                     tols=None,
                     size=(4,3)):
    ax = df.plot(figsize=size,title=title,marker='*')
    if tols is not None:
        ax.set_xticks(tols)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(savename,format='pdf', bbox_inches='tight', pad_inches=0)

    
def plot_clf_feature_heatmap(Val,savename,
                            title='weighted F1 score matrix (median) using S1',
                            xlabel='classifier',
                            ylabel='feature',
                            size=(4,3)):
    plt.figure(figsize=size)
    Val = Val.reindex_axis(sorted(Val.columns,key=lambda s: s.lower()), axis=1)

    ax = sns.heatmap(Val, annot=True, cmap="YlGnBu",
                     annot_kws={"size": 9}, fmt='.2f')
    plt.yticks(rotation=0) 
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(savename,format='pdf', bbox_inches='tight', pad_inches=0) 

def plot_heatmap2(y_prob, y_true, tag, savename, size=(10,8), title='',xlabel='',ylabel=''):
    # y_prob : n_instance * n_classes
    # y_true : n_instance * 1
    plt.figure(figsize=size)
#     _,y_prob,y_true = raw_pred
    # [1] represents the random forest
    prob_matrix =  y_prob.copy()
    n_classes = len(tag)    
    cfm_prob = np.zeros([n_classes,n_classes])

    for i in range(len(tag)):
        ix = np.where(y_true==i)[0]
        cfm_prob[i,:] = np.mean(prob_matrix[ix,],0)
    labels = [t+'('+ str(len(np.where(y_true==count)[0])) + 
              ')'+'-'+str(count) for count,t in enumerate(tag)]
    Val = pd.DataFrame(cfm_prob, index=labels)
    ax = sns.heatmap(Val,cmap="YlGnBu", annot=False,fmt='.2f', annot_kws={"size": 9},
                     linewidths=.5, linecolor='black')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(savename,format='pdf', bbox_inches='tight', pad_inches=0)

def plot_prob_illustration(y_prob, y_true, tag, ix=range(10),savename='temp.pdf',
                           prob_threshold=0.5, size=(7,4)):
    sns.set()
    plt.figure(figsize=size)
    
    prob_matrix = y_prob.copy()
    prob_matrix[prob_matrix<prob_threshold] = 0

    prob_df = pd.DataFrame(prob_matrix[ix,:], 
                           index=[tag[i] for i in y_true[ix]],
                           columns=tag)

    g = sns.heatmap(prob_df)

    from matplotlib.patches import Rectangle

    ax = g.axes
    ax.set_title('probabilistic prediction for %d samples with threshold %.1f'%(len(ix),prob_threshold) )
    temp = np.argmax(y_prob[ix],axis=1)
    for i,j in enumerate(ix):
        ax.add_patch(Rectangle((temp[i],len(ix)-i-1), 1, 1, fill=False, edgecolor='blue', lw=2))
    plt.savefig(savename,format='pdf', bbox_inches='tight', pad_inches=0)