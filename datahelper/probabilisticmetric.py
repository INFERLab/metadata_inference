import numpy as np
import pandas as pd


def get_tolerance_metric(y_prob, y_true, tols):
    y_pred_order = np.array([np.argsort(i)[::-1] for i in y_prob])
    tol_metric = [np.sum([y_true[i] in y_pred_order[i,:t] for i in range(len(y_true))])*1./len(y_true) for t in tols]
    tol_metric_df = pd.DataFrame(tol_metric,
                                index = tols,
                                columns = ['accuracy'])
    return tol_metric_df


def get_metric_given_threshold(y_prob, y_true, prob_threshold):
    prob_matrix = y_prob
    y_pred = np.argmax(y_prob,1)
    # consider the prediction above prob_threshold
    ix = np.unique(np.where(prob_matrix>prob_threshold)[0])

    metric_coverage = len(ix)*1./len(y_true)
    
    metric_correct_pred = np.sum(y_pred[ix] == y_true[ix])*1./len(ix)
        
    return metric_coverage, metric_correct_pred

def get_prob_metric_df(y_prob, y_true, probs):
    prob_metric = []

    for p in probs:
        prob_metric.append(get_metric_given_threshold(y_prob, y_true, p))

    prob_metric_df  = pd.DataFrame(prob_metric, 
                                   index = probs,
                                   columns=["coverage","accuracy"])
    return prob_metric_df