import pandas as pd
import numpy as np
from scipy import stats
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns

from IPython.display import set_matplotlib_formats, display, HTML
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300

def plot_2D_visualize(TSNEF,PCAF,type_label,filename,legends, size=(14,7), s=2):
    n = len(np.unique(type_label))
    
    colors = cm.rainbow(np.linspace(0, 1, n))
    # markers='ov^<>sph+Dx'
    fig = plt.figure(figsize=size)
    
    plt.subplot(1,2,1)
    for i, c in zip(range(n), colors):
        ix = np.where(type_label==i)[0]
        plt.scatter(TSNEF[ix,0], TSNEF[ix,1], color=c,label=legends[i],s=s)
    plt.title('2D Visulization of TSNE')
    
    plt.subplot(1,2,2)
    for i, c in zip(range(n), colors):
        ix = np.where(type_label==i)[0]
        plt.scatter(PCAF[ix,0], PCAF[ix,1], color=c,label=legends[i],s=s)
    plt.title('2D Visulization of PCA')

    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.savefig(filename,format='pdf', bbox_inches='tight', pad_inches=0)
    
def plot_point_frequency_count(y,tag,yy,selected,fig_path,cnt_threshold=10):
    # we show the types of points distribution with count more than 10
    
    s=pd.Series(np.array([tag[i] for i in y]))

    ss = pd.Series(s.value_counts())
    ss = ss.iloc[::-1]
    ss1 = ss[ss>=cnt_threshold]

    words=np.array([selected[i] for i in yy])

    many_words = words
    s = pd.Series(many_words)

    ss = pd.Series(s.value_counts())
    ss = ss.iloc[::-1]
    ss2 = ss[ss>cnt_threshold]

    # ss.plot(kind='barh', rot=0,figsize=(4,5), colors=['red'])
    color_array = ['green' if i in ss2.index else 'blue' for i in ss1.index]

    ax = ss1.plot(kind='barh', rot=0,figsize=(4,8), colors=color_array)
    ax.set_xlabel('frequency count')
    # ax.set_ylabel('point label')
    ax.set_ylabel('tags')
    plt.savefig(fig_path +'freq1.pdf',format='pdf', bbox_inches='tight', pad_inches=0)
    
def plot_point_count_per_site(select_df,selected,fig_path,size=(12,4)):
    grped = select_df.groupby(['customer','point_name'])

    cnt = grped.count()

    num_type = len(selected)
    sites_unq = select_df['customer'].unique()
    num_sites = len(sites_unq)

    ct_mtx = np.zeros([num_sites,num_type])

    for i in range(num_sites):
        for j in range(num_type):
            if  selected[j] in cnt.ix[sites_unq[i]].index:
                ct_mtx[i,j] = cnt.ix[sites_unq[i]].ix[selected[j]].values[0]


    plotdf = pd.DataFrame(ct_mtx,columns=selected,index=sites_unq)

    plotdf['total count'] = plotdf.sum(axis=1).astype(int)
    plotdf = plotdf.sort(columns='total count',ascending=False)
    
    newdf = plotdf.drop('total count',1)
    plt.figure(figsize=size)
    ax = sns.heatmap(newdf.T,cmap="Set3",annot=True,annot_kws={"size": 8},fmt='.0f')
    ax.set_xlabel('site number and the number of points from this site inside the bracket')
    ax.set_ylabel('point label')
    temp_xtick = [plotdf.index[i]+' - ['+str(plotdf['total count'][i])+']' for i in range(len(plotdf.index))]
    ax.set_xticklabels(temp_xtick,rotation=300,ha='left')
    plt.savefig(fig_path+'freq2-selected.pdf',format='pdf', bbox_inches='tight', pad_inches=0)