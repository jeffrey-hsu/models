import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import time

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

def load_data(path):
    ## load pandas dataframe
    df = read_csv(os.path.expanduser(path))
    ## The Image column has pixel values separated by space
    ## convert the pixel values to numpy arrays
    df["Image"] = df["Image"].apply(lambda im: np.fromstring(im, sep=" "))
    ## drop all rows that have missing values in them
    #df = df.dropna()
    return df

def loadXY(df):
    X = np.vstack(df["Image"].values) / 255
    Y = (df[df.columns[:-1]].values - 48) / 48
    X, Y = shuffle(X, Y)
    X = X.astype(theano.config.floatX)
    Y = Y.astype(theano.config.floatX)
    return X, Y

def subset_data(X, Y, full_keypoints, cols):
    """
    The function subsets the label data based on the keypoints
    specified and returns the training data and subsetted
    training labels with complete keypoint position labeling.

    Input:
        - x: training image data. (ndarray)
        - y: training keypoint label data. (ndarray)
        - keypoints: list of column names wanted
    Output:
        - x: training image data with complete keypoint labeling. (ndarray)
        - y: subsetted keypoint label data. (ndarray)
    """
    col_index = []
    for col in cols:
        if col in full_keypoints:
            col_index.append(full_keypoints.index(col))
    Y_subset = Y[:,col_index]
    ## only select the examples that has complete keypoint values
    complete = ( np.sum( (np.isnan(Y_subset) == True )*1, axis=1) == 0).tolist()
    X, Y = X[complete,:], Y_subset[complete,:]
    return X, Y

def load_2d_images(data, imageWidth):
    data = data.reshape(-1, 1, imageWidth, imageWidth)
    return data
