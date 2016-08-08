import numpy as np
import pandas as pd
import msgpack
import msgpack_numpy as m
import pickle
import base64

from time import time

import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('fivethirtyeight')
#import seaborn as sns
from itertools import cycle
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
#from sklearn.neural_network import MLPClassifier

from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib



#######################    MODEL PROCESSING   ###########################

def train_online(model_file, smpls, clss, classes=[0, 1]):
    clf = joblib.load(model_file)
    clf.partial_fit(smpls, clss, classes=classes)
    joblib.dump(clf, model_file)
    return clf


def predict_online(model_file, smpls, proba=None):
    clf = joblib.load(model_file)
    if proba == None:
        return clf.predict(smpls)
    else:
        return clf.predict_proba(smpls)[:,proba]



#######################    DATA PROCESSING   ###########################

def decode_b(obj):
    try:
        if b'nd' in obj:
            if obj[b'nd'] is True:
                return np.fromstring(obj[b'data'], dtype=np.dtype(obj[b'type'])).reshape(obj[b'shape'])
            else:
                return np.fromstring(obj[b'data'], dtype=np.dtype(obj[b'type']))[0]
        elif b'complex' in obj:
            return complex(obj[b'data'])
        else:
            return obj
    except KeyError:
        return obj


def decode_unpack(var):
    return msgpack.unpackb(base64.b64decode(var), object_hook=decode_b)

#
def col_split (df,col,ind):
    return pd.DataFrame.from_records(df[col],
                                     index = df[ind].apply(lambda x: x.decode()),
                                     columns=pd.MultiIndex.from_tuples([(col.decode(),i) for i in range(len(df[col][0]))],
                                                                       names=['key', 'ind']))

def process_log_data (ser):
    requestDecoded = pd.DataFrame.from_records(ser.apply(decode_unpack))
    requestDecoded = pd.concat([col_split(requestDecoded, b'bid_request_body', b'requestID'),
                                col_split(requestDecoded, b'weight_funnel_stage_0', b'requestID')], axis=1)
    return requestDecoded



###################   DATA VISUALIZATION   ###########################

# Plot result in 2D by PCA
def plot_2d(F, ttl='PCA of dataset'):
    plt.figure(figsize=(20, 10))
    plt.scatter(F[:, 0], F[:, 1])
    # plt.legend()
    plt.title(ttl)
    plt.show()


def plot_2d_labled(F, L, ttl='PCA of dataset'):
    plt.figure(figsize=(20, 10))
    for i, c in zip(range(len(set(L))), colors):
        plt.scatter(F[L.values == i, 0], F[L.values == i, 1], c=c, label=i)
    plt.legend()
    plt.title(ttl)
    # set axes range
    plt.xlim(F[:, 0].min(), F[:, 0].max())
    plt.ylim(F[:, 1].min(), F[:, 1].max())
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None):
    plt.figure(figsize=(20, 10))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    t0 = time()

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)

    train_time = time() - t0
    plt.title("%s - train time: %0.3fs" % (title, train_time))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    print("Stratified 3-Fold cross-validation \ntrain_scores_mean:")
    print(train_scores_mean)
    print("test_scores_mean:")
    print(test_scores_mean)
    return plt


def plot_validation_curve(estimator, X, y, param_name, param_range=np.logspace(-6, -1, 5), title=None, ylim=None, cv=None, n_jobs=1, scoring=None):
    plt.figure(figsize=(20,10))
    #param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()
    print("Stratified 3-Fold cross-validation \ntrain_scores_mean:")
    print(train_scores_mean)
    print("test_scores_mean:")
    print(test_scores_mean)
    return plt

