import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback

from pandas import DataFrame
from pandas.io.parsers import read_csv
from datetime import datetime

from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from scipy import ndimage

from collections import OrderedDict

from keras.models import model_from_json
import cv2


#FTRAIN = '~/data/training.csv'
#FTEST = '~/data/test.csv'
#FLOOKUP = '~/data/IdLookupTable.csv'

FTRAIN = os.getcwd()+'/data/training.csv'
FTEST = os.getcwd()+'/data/test.csv'
FLOOKUP = os.getcwd()+'/data/IdLookupTable.csv'



from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler, EarlyStopping

import theano


def float32(k):
    return k.astype(theano.config.floatX)#np.cast['float32'](k)


def load(test=False, cols=None,filter_bad=True):
    print 'Loading data.....'
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]
    
    if not test:
        #drop bad images 1747, 1877, 1907, 2090, 2199
        if filter_bad:
            image_indices = [1747, 1877, 1907, 2090, 2199]
            count=0
            for index in image_indices:
                df = df.drop(df.index[index-1-count])
                count +=1
        df = df.dropna()  # drop all rows that have missing values in them
    #print(df.count())  # prints the number of values for each column

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    print 'Loading data.....Done'
    return X, y 


def load2d(test=False, cols=None,blur=False,filter_bad=True):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    if blur:
        for i in range(X.shape[0]):
            X[i] = ndimage.gaussian_filter(X[i], sigma=1)
    return X, y

def get_split_data(X,y):
    train_data, dev_data, train_labels, dev_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_data, dev_data, train_labels, dev_labels


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)#Create a matrix of pixels
    axis.imshow(img, cmap='gray')#Plot the pixels in gray scale
    #Mark the x and y coordinates of the key point with an 'x' in red color
    #Since the coordinates are scaled, we multiply and add 48 to get the original values
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=20,c='red')


def plot_model_performance(data):
    plt.plot(data.history['loss'], linewidth=3, label='train')
    plt.plot(data.history['val_loss'], linewidth=3, label='validation')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-5, 1e-2)
    plt.yscale('log')
    plt.show()
    

def save_model(model,name):
    json_string = model.to_json()
    architecture = name+'_architecture.json' 
    weights = name+'_weights.h5'
    open(architecture, 'w').write(json_string)
    model.save_weights(weights)


def retrieve_model(name,weights=True):
    print 'Retrieving model: {0}'.format(name)
    architecture = name+'_architecture.json' 
    model_saved = model_from_json(open(architecture).read())
    
    if weights:
        weights = name+'_weights.h5'    
        model_saved.load_weights(weights)
    return model_saved
    

def get_error(model,X,y):
    mse = mean_squared_error(model.predict(X), y)
    error = np.sqrt(mse) * 48
    print '**Mean Squared Error : {0}'.format(error)
    return error


def generate_submission_file(specialists):
    X = load2d(test=True)[0]
    y_pred = np.empty((X.shape[0], 0))

    for model in specialists.values():
        y_pred1 = model.predict(X)
        y_pred = np.hstack([y_pred, y_pred1])

    columns = ()
    for cols in specialists.keys():
        columns += cols

    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns=columns)

    lookup_table = read_csv(os.path.expanduser(FLOOKUP))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print '**Generated File:: {0}'.format(filename)
    

    
def get_saved_specialists(prefix):
    specialists = OrderedDict()
    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        name = '{0}_{1}'.format(prefix,cols[0])
        model = retrieve_model(name,weights=True)
        specialists[cols] = model
    return specialists


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
    
    
def get_all_images():
    print 'Loading data.....'
    fname = FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48  # scale target coordinates to [-1, 1]
    y = y.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)

    print 'Loading data.....Done'
    return X, y 


SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
        ),
        flip_indices=((0, 2), (1, 3)),
    ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
        ),
        flip_indices=(),
    ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        ),
        flip_indices=((0, 2), (1, 3)),
    ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
        ),
        flip_indices=(),
    ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
    ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
    ),
]
    
    