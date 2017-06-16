'''This module contains functions to calculate metric for multi-class
classification '''
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, f1_score

def get_MCmetric(y_true,y_pred,y_prob,tags):

    names = ['accuracy score',
             'confusion matrix',
             'report',
             'f1-macro',
             'f1-weighted',
             'f1-micro',
             'f1']
    metric = {}

    # accuracy score
    metric[names[0]] = accuracy_score(y_true, y_pred)

    # confusion matrix
    metric[names[1]] = confusion_matrix(y_true, y_pred)

    # classification report
    temp_report = classification_report(y_true, y_pred)
    temp = [i.split('  ') for i in str(temp_report)
            .replace('\n\n', '\n').split('\n')]
    temp = [list(filter(None, i)) for i in temp]
    temp[0].insert(0, 'tag')
    del temp[-1]
    report = np.reshape(np.array(temp), (-1, 5))
    df = pd.DataFrame(report[1:, 1:].astype(np.float),
                  index=[tags[int(i)] for i in report[1:-1,0]]+[report[-1,0]],
                  columns=report[0, 1:])
    metric[names[2]] = df

    ######
    ## different f1_score
    #####
    # "macro" simply calculates the mean of the binary metrics, giving equal weight to each class.
    # In problems where infrequent classes are nonetheless important, macro-averaging may be a means
    # of highlighting their performance. On the other hand, the assumption that all classes are equally
    # important is often untrue, such that macro-averaging will over-emphasize the typically low
    # performance on an infrequent class.
    # f1-macro
    metric[names[3]] = f1_score(y_true, y_pred, average='macro') if len(np.unique(y_true))!=2 and len(np.unique(y_pred)!=2) else np.mean(f1_score(y_true, y_pred, average=None))
    # "weighted" accounts for class imbalance by computing the average of binary metrics in which each
    #  class s score is weighted by its presence in the true data sample
    # f1-weighted
    metric[names[4]] = f1_score(y_true, y_pred, average='weighted') if len(np.unique(y_true))!=2 and len(np.unique(y_pred)!=2) else np.mean(f1_score(y_true, y_pred, average=None))
    # 'micro' -  micro-averaging may be preferred in multilabel settings,
    # including multiclass classification where a majority class is to be
    # ignored.
    # f1-micro
    metric[names[5]] = f1_score(y_true, y_pred, average='micro') if len(np.unique(y_true))!=2 and len(np.unique(y_pred)!=2) else np.mean(f1_score(y_true, y_pred, average=None))
    # "None" will return an array with the score for each class
    # f1
    metric[names[6]] = f1_score(y_true, y_pred, average=None)

    ######
    ## ROC (receiver operation charateristics)
    ######

    return metric

def print_metric(metric):
    for key,val in metric.items():
        print(key,':\n',val)

def ave_metric(metrics):
    '''metrics should be a list of metric'''
    '''we also calculate the variation of the metric with single values'''
    '''we don't report the average metric for confusion matrix and report'''
    n = len(metrics)
    keys = metrics[0].keys()
    ave_metric = dict.fromkeys(keys,0.0)
    for me in metrics:
        for key in keys:
            if key != 'confusion matrix' and key != 'report':
                ave_metric[key] += me[key]
    for key in keys:
        if key != 'run time(s)' and key != 'confusion matrix' and key != 'report':
            ave_metric[key] /= n

    return ave_metric

if __name__ == '__main__':
    '''unit test'''
    yy_true = np.array([0, 1, 2,1])
    yy_pred = np.ones((4,1))
    yy_prob = np.array([[.2,.8,0,0],
                        [.1,.5,.2,.2],
                        [0,.9,0,.1],
                        [0,1,0,0]])
    tags = ['a','b','c']
    metric = get_MCmetric(yy_true,yy_pred,yy_prob,tags)
    print_metric(metric)
