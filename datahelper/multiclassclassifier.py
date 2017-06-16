"""MCclassifier model to handle multi class classification"""
from __future__ import print_function
import logging
from datetime import datetime
from time import time
import numpy as np
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from .util import *
from .multiclassmetric import *

# logger = logging.getLogger(__name__)

# we use 50% data from one zone to train and rest to test
# In this case, we can compare training and testing on different zones
def evaluate_zone_wise_S2(F,syn_y,syn_tag,syn_site_label,syn_zone_label,syn_zone_tag,syn_zones,clf,train_ratio=.5):
    '''train on one zone, evaluate on the others'''
    Result = {}

    start = time()
    for train_z in syn_zones:
        base_zone_ix = np.where(syn_zone_label==train_z)[0]
        base_zoneX = F[base_zone_ix,:]
        base_zone_y = syn_y[base_zone_ix]
        
        F_train = base_zoneX
        
        # we randomly split base_zoneX into training and testing
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio)
        for temp_train, temp_test in sss.split(F_train, base_zone_y):
            this_trainX, this_trainy, base_zone_testX, base_zone_testy = F_train[temp_train], base_zone_y[temp_train],\
                                                                F_train[temp_test], base_zone_y[temp_test]
        y_train_unq = np.unique(base_zone_y)

        this_metric, this_y_pred, this_y_prob = generate_metric_for_clf(clf, this_trainX, this_trainy, \
                                                                             base_zone_testX, base_zone_testy, syn_tag)
        Result[(train_z,train_z)] = (this_metric,this_y_pred, this_y_prob)

        # we use the same training data to build the model and test on rest zones
        F_train = this_trainX
        y_train = this_trainy

        for test_z in syn_zones:
            if test_z != train_z:
                test_ix = np.where(syn_zone_label!=test_z)[0]
                testX = F[test_ix,:]
                y_test = syn_y[test_ix]

                y_test_unq = np.unique(y_test)

                new_eles = set(y_test_unq) - set(y_train_unq)
                # if len(new_eles) == 0:
                #     # good
                #     pass
                # else:
                #     # abnormal
                #     print('Abnormal Case to handle!!!')
                #     print((train_z,test_z))
                #     break         
                F_test = testX
                metric, y_pred, y_prob = generate_metric_for_clf(clf, F_train, y_train, F_test, y_test, syn_tag)
                Result[(train_z,test_z)] = (metric,y_pred,y_prob)
    return Result


def get_S2_metrics(new_df,F_features,F_names,yy,new_tag,clfs,verbose=True):
    site_label,site_tag = pd.factorize(new_df['customer'])

    S2 = nested_dict()
    start = time()
    for j in range(len(F_names)):
        temp_feature = F_features[j]
        for k,c in clfs.items():
            temp = classifier_leave_one_group_out(c,temp_feature,yy,site_label,new_tag)
            for i in range(len(site_tag)):
                S2[site_tag[i]][F_names[j]][k] = temp[0][i],temp[1][i]
        if verbose:
            print('%d - working on %s\t time:%ds' % (j+1,F_names[j],time()-start))
    return S2

def get_S2_metrics_wrapper(args):
    return get_S2_metrics(*args)

def get_S1_metrics(new_df,F_features,F_names,yy,new_tag,clfs,train_size=.2,n_iter=2,verbose=True):
    site_label,site_tag = pd.factorize(new_df['customer'])
    S1 = nested_dict()
    # pool = Pool()
    start = time()
    for i in range(len(site_tag)):
        if verbose:
            print('%d - working on %s...\t time:%ds' % (i+1,site_tag[i],time()-start))
        temp_site_ix = np.where(site_label==i)[0]
        temp_site_F = [f[temp_site_ix] for f in F_features]
        temp_site_y = yy[temp_site_ix]
        _,temp_site_y,temp_tag = validity_check(temp_site_F[0], temp_site_y, new_tag, train_size)
        if len(temp_tag) < 2:
            if verbose:
                print('Not enough samples')
            continue
        else:
            temp_site_F = [validity_check(f, yy[temp_site_ix], new_tag, train_size)[0] for f in temp_site_F]
        for j in range(len(F_names)):
            temp_feature = temp_site_F[j]
            for k,c in clfs.items():
                S1[site_tag[i]][F_names[j]][k] = classifier_trainratio(c,
                                                                       temp_feature, 
                                                                       temp_site_y, 
                                                                       new_tag,
                                                                       train_size,
                                                                       n_iter=n_iter)
    return S1

def get_S1_metrics_wrapper(args):
    return get_S1_metrics(*args)


def classifier_leave_one_group_out(clf,X,Y,groups,tag):
    metrics = []
    raw_pred = []
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    for train, test in logo.split(X,Y, groups=groups):
        X_train, y_train, X_test, y_test = X[train], Y[train],\
                                            X[test], Y[test]
        metric, y_pred, y_prob  = generate_metric_for_clf(clf,
                                                          X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test,
                                                          tag)
        metrics.append(metric)
        raw_pred.append((y_pred, y_prob, y_test))

    return metrics, raw_pred


def generate_metric_for_clf(clf,X_train,y_train, X_test, y_test, tags):

    start_time = time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = np.array(clf.predict_proba(X_test))
    run_time = (time()-start_time)
    
    metric = get_MCmetric(y_test, y_pred, y_prob, tags)
    metric['run time(s)'] = run_time
    
    return metric,y_pred, y_prob

def classifier_trainratio(clf,X,Y,tag,train_size=.5,n_iter=20):
    metrics = []
    raw_pred = []
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_iter,test_size=1.-train_size)
    for train, test in sss.split(X,Y):
        X_train, y_train, X_test, y_test = X[train], Y[train],\
                                            X[test], Y[test]
        metric, y_pred, y_prob  = generate_metric_for_clf(clf,
                                                          X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test,
                                                          tag)
        metrics.append(metric)
        raw_pred.append((y_pred, y_prob, y_test))

    return ave_metric(metrics), metrics, raw_pred

def validity_check(X, y, tag, train_size):
    # check if y has enough samples for this site before feeding into classifier_trainratio
    min_count = max(np.ceil(1/train_size),np.ceil(1/(1-train_size)))
    c = Counter(y)
    rm_ix = [i for i in range(len(y)) if c[y[i]]<min_count]
    X = np.delete(X, rm_ix, axis=0)
    y = np.delete(y, rm_ix, axis=0)
    
    tag = [tag[i] for i in np.unique(y)]
    
    return X,y,tag


def classifier_trainratio_wrapper(args):
    return classifier_trainratio(*args)