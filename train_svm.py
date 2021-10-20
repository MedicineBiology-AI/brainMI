# -*- coding: utf-8 -*-
from functools import reduce

from sklearn.preprocessing import minmax_scale, scale

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, average_precision_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor as Executor

import numpy as np
import argparse

np.random.seed(22)

def set_args():
    parser = argparse.ArgumentParser(description='Train SVM') 

    parser.add_argument('--x_path', type=str, default=None,
                        help='The path to feature files.', nargs='+')
    parser.add_argument('--y_path', type=str, default=None,
                        help='The path to label files.')

    args = parser.parse_args()

    return args


def features_labels_load(feature_path, y_path):

    features = [np.load(i, allow_pickle=True)["features"] for i in feature_path]
    symbol_all = [np.load(i, allow_pickle=True)["symbol"] for i in feature_path]

    if len(symbol_all) > 1:
        symbol = list(set(reduce(lambda x,y: list(x)+list(y), list(symbol_all))))
        symbol.sort()

        dict_symbol = dict(zip(symbol, range(len(symbol))))
        
        features_union = np.zeros((len(symbol_all), len(symbol), features[0].shape[1]))
        for i in range(len(symbol_all)):
            mask = np.array([dict_symbol[j] for j in symbol_all[i]])
            features_union[i][mask] = features[i]
            
        features = np.concatenate(features_union, axis=1)
    else:
        symbol = symbol_all[0]
        features = features[0]

    postive_symbol = np.load(y_path)
    label = np.array([i in postive_symbol for i in symbol])
    
    pos_index, neg_index = [i for i, j in enumerate(label) if j == 1], [i for i, j in enumerate(label) if j == 0]
    neg_index_choice = np.random.choice(neg_index, len(pos_index), replace=False)
    all_index = np.concatenate([pos_index, neg_index_choice], axis=0)
    features = features[all_index]
    label = label[all_index]

    # print(label.shape)

    return features, label

def worker(x_train, x_valid, y_train, y_valid, C=1.0):
# def worker(x_train, x_valid, y_train, y_valid,
#            mustlinks):
    model = SVC(C=C, kernel='precomputed', class_weight='balanced', random_state=10, probability=True, verbose=False)

    model.fit(x_train, y_train)

    y_valid_score = model.predict_proba(x_valid)[:, 1]

    aupr, auc = average_precision_score(y_valid, y_valid_score), roc_auc_score(y_valid, y_valid_score)
    acc, f1_ =  accuracy_score(y_valid, y_valid_score>0.5), f1_score(y_valid, y_valid_score>0.5)
    

    return auc, aupr, acc, f1_


def run(X, y, gamma, C):
    X = rbf_kernel(X, gamma=gamma)

    kfold = StratifiedKFold(n_splits=5, random_state=22, shuffle=True)

    tasks, results, true_pred = [], [], []
    with Executor(max_workers=5) as executor:

        for train_index, valid_index in kfold.split(X, y):
            x_train, x_valid = X[train_index][:, train_index], X[valid_index][:, train_index]
            y_train, y_valid = y[train_index], y[valid_index]
            
            tasks.append(executor.submit(worker, x_train, x_valid, y_train, y_valid, C))

        for future in as_completed(tasks):
            results.append(list(future.result()))

    # print("#########  gamma: {}, C: {} ###########".format(gamma, C))
    # print(np.mean(np.array(results), axis=0))

    return np.mean(np.array(results), axis=0)


def main():

    args = set_args()
    x_path, y_path = args.x_path, args.y_path
    X, y = features_labels_load(x_path, y_path)

    gammas = [0.001, 0.01, 0.1, 1.0]
    Cs = [0.1, 1.0, 10.0, 100.0]
    
    best_auc = 0.
    best_C, best_gamma = 0., 0.
    for gamma in gammas:
        for C in Cs:
            result = run(X, y, gamma, C)
            if result[0] > best_auc:
                best_auc = result[0]

                best_C, best_gamma = C, gamma 
                best_result = result 
    
    print("### best_gamma: {}, best_C: {} ###".format(best_gamma, best_C))
    # print(best_result)
    print('AUROC: {:.3f}\tAUPRC: {:.3f}\tAcc: {:.3f}\tF1: {:.3f}'.format(best_result[0], best_result[1], best_result[2], best_result[3]))

if __name__ == '__main__':
    main()
