from __future__ import print_function
import os
import sys
from time import time
from collections import Counter,defaultdict
import pickle
from multiprocessing import Pool 
import warnings
warnings.filterwarnings("ignore")


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

# customized data helper functions 
from datahelper.timeseriesfeatures import *
from datahelper.plotdata import *
from datahelper.plotmodelresults import *
from datahelper.multiclassmetric import *
from datahelper.probabilisticmetric import *
from datahelper.multiclassclassifier import *
from datahelper.util import *


'''
# This section only needs to be run once, once the features are stored in input.pkl, 
# we can comment out this section

# path for pkl data and meta
data_pkl_path = os.environ['DATA_PKL_PATH']
meta_pkl_path = os.environ['META_PKL_PATH']

# this pickle file contains one year long measurement of interpolated data from AHUs
with open(data_pkl_path + "clean_data.pkl",'rb') as f:
    base_time,X,y,raw_tag,df = pickle.load(f)
# meta pickle file contains the mappings of the points to the building and site information    
with open(meta_pkl_path + 'clean_meta.pkl','rb') as f:
    site_mapping,sp_meta = pickle.load(f)

# generate features
feature_funs = [getF_1994_Li, getF_2015_Gao, getF_2015_Hong,
         getF_2015_Bhattacharya, getF_2015_Balaji, getF_2016_Koh]
F_names = ["F1: Li et al. 1994","F2: Gao et al. 2015","F3: Hong et al. 2015",
           "F4: Bhattacharya et al. 2015","F5: Balaji et al. 2015","F6: Koh et al. 2016","F7: Combination"]

F_features = [f(X) for f in feature_funs] 

F_features.append(np.hstack(F_features))


# Output Variables
# F_features
# F_names
# y
# raw_tag
# df
# site_mapping
# sp_meta

# save outputs to pickle files to avoid computing them every time
pickle.dump((F_features,F_names,y,raw_tag,df,site_mapping,sp_meta),
            open('input.pkl','wb'))
'''

cur_dir = os.path.dirname(os.path.realpath(__file__))

# generate output variables
F_features,F_names,y,raw_tag,df,site_mapping,sp_meta =\
                pickle.load(open(cur_dir+'/input.pkl','rb'))
    

# build clfs    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# clfs
# given the run time of SVM is too slow, we ignore it.
clfs={"kNN": make_pipeline(KNeighborsClassifier(n_neighbors=1, n_jobs=-1)),
 "Naive Bayes": make_pipeline(GaussianNB()),
 "Logistic": make_pipeline(LogisticRegression(C=1e5, n_jobs=-1)),
   "LDA": make_pipeline(LDA(solver='lsqr',shrinkage='auto')),
 "Decision Tree": make_pipeline(tree.DecisionTreeClassifier(max_depth=10)),
 "Random Forest": make_pipeline(RandomForestClassifier(max_depth=10,n_estimators=20,max_features='auto',criterion='gini',n_jobs=-1)),
 "AdaBoost": make_pipeline(AdaBoostClassifier(n_estimators=100))}
 # "SVM": make_pipeline(svm.SVC(kernel='rbf',C=1, probability=True)),
