import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from six.moves import cPickle

from sklearn.metrics import classification_report
import time

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d


def load_data(path):
    ## load pandas dataframe
    df = read_csv(os.path.expanduser(path))
    
    ## The Image column has pixel values separated by space
    ## convert the pixel values to numpy arrays
    df["Image"] = df["Image"].apply(lambda im: np.fromstring(im, sep=" "))
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


class convNetBuilder:
    """
    The convNetBuilder class supports creating 
    Convolutional Neural Network models given 
    the architectures defined.
    
    When called, the __init__ method creates 
    object attributes based on the architectural 
    parameters specified. For the list of 
    parameters required for initializing the 
    convNetBuilder object, please see description
    under __init__ method.
    
    To use the class:
    1. initialize a convNetBuilder object
    2. call init_params to create the weight matrices 
    3. reshape the training data and dev data 
       by reshape_data method
    4. build the convolution, polling and activation 
       model of layers by the model method
    5. call the desired gradient descent method
    """
    
    def __init__(self, numClasses, patchSize = [3,3],
                 featureMapLayers = [32,64,128], numHiddenNodes = 600,
                 numNNLayer = 2, imageWidth = 96, poolingSize = 2,
                 train_dropout_rate = [0.,0.]):
        """
        The following parameters are required:
        - numClasses: y label dimensions
        - miniBatchSize: positive integar >= 1
        - patchSize: the filter size in both width and height as a list (=> 3)
        - featureMapLayers: list of number of feature map per convolutional layers
        - numHiddenNodes: number of hidden nodes per fully connected layers
        - numNNlayer: number of fully connected layers
        - imageWidth: original input image width
        - poolingSize: size of the pooling (e.g. 2,2) for feature reduction
        """
        self.numFeatureMaps = featureMapLayers
        self.numConvLayers = len(featureMapLayers)
        self.patchWidth = patchSize[0]
        self.patchHeight = patchSize[1]
        self.numHiddenNodes = numHiddenNodes
        self.numNNLayer = numNNLayer
        self.imageWidth = imageWidth
        self.poolingSize = poolingSize
        self.numClasses = numClasses
        #self.miniBatchSize = miniBatchSize
        self.params = self.__init_param()
        self.X = T.tensor4()
        self.Y = T.matrix()
        self.train_dropout_rate = train_dropout_rate
        self.Y_hat_train = self.__init_model(self.X, self.params, self.train_dropout_rate)
        self.Y_hat_predict = self.__init_model(self.X, self.params)
        #self.training_cost = self.__training_cost()
        
    def __init_param(self):
        """
        This method creates the weight parameters used 
        building the model. The inputs required are 
        specified per creation of the object. By calling 
        the init_param() method, it returns a list of 
        parameters with each elements representing the 
        weight tensor in each layer of the neural network.
        For the Convolutional layers, the weights are
        represented as a 4D tensor. For the connected 
        neural network layers, the weights are represemted
        by a 2D matrix.
        """
        params = [0.0 for i in range(self.numConvLayers + self.numNNLayer)]
        for convlayer in range(self.numConvLayers):
            
            if convlayer == 0:
                params[convlayer] = theano.shared( 
                    np.asarray( 
                        (np.random.randn(
                         *(self.numFeatureMaps[convlayer], 1, self.patchWidth, self.patchHeight))*.01) ))
            else:
                params[convlayer] = theano.shared(
                    np.asarray(
                        (np.random.randn(
                         *(self.numFeatureMaps[convlayer], self.numFeatureMaps[convlayer-1], 
                              self.patchWidth, self.patchHeight))*.01) ))
        
        firstNNLayer = int(self.numFeatureMaps[self.numConvLayers-1] * \
        (self.imageWidth / (self.poolingSize ** self.numConvLayers))**2)
        
        for nnlayer in [i + self.numConvLayers for i in range(self.numNNLayer)]:
            
            if nnlayer == self.numConvLayers:
                params[nnlayer] = theano.shared(
                    np.asarray(
                        (np.random.randn(
                         *(firstNNLayer, self.numHiddenNodes))*.01) ))
            
            elif nnlayer == (self.numConvLayers+self.numNNLayer-1):
                params[nnlayer] = theano.shared(
                    np.asarray(
                        (np.random.randn(
                         *(self.numHiddenNodes, self.numClasses))*.01) ))
            
            else:
                params[nnlayer] = theano.shared(
                    np.asarray(
                        (np.random.randn(*(self.numHiddenNodes, self.numHiddenNodes))*.01) ))
        return params
    
    def __dropout(self, X, p=0.):
        srng = RandomStreams()
        if p > 0:
            X *= srng.binomial(X.shape, p = 1-p)
            X /= 1-p
        return X
    
    def __init_model(self, X, params, dropout_rate = [0., 0.]):
        l = X 
        borderX = int(self.patchWidth / 2)
        borderY = int(self.patchHeight / 2)
        p1 = dropout_rate[0]
        p2 = dropout_rate[1]
        
        for i in range(self.numConvLayers - 1):
            l = self.__dropout(pool_2d(
                T.maximum(conv2d(l, params[i], border_mode=(borderX,borderY)),0.),
                (self.poolingSize, self.poolingSize), ignore_border=True), p1)
        
        l = self.__dropout(T.flatten(pool_2d(
            T.maximum(conv2d(l, params[self.numConvLayers - 1], border_mode=(borderX,borderY)), 0.),
            (self.poolingSize, self.poolingSize), ignore_border=True), outdim=2), p1)    
        
        for i in range(self.numNNLayer - 1):
            l = self.__dropout(T.maximum(T.dot(l, params[self.numConvLayers + i]), 0.), p2)
        
        model = T.dot(l, params[self.numConvLayers + self.numNNLayer - 1])
        
        return model
    
    def __training_cost(self, N):
        return ((self.Y_hat_train - self.Y)**2).sum() / (2 * N)
    
    def SGD(self, X, Y,
            update_rule = "backprop", epochs = 10, 
            miniBatchSize = 1, learning_rate = 0.01,
            validation = []):
        """
        Stochastic Gradient Descent:
        - X: training examples
        - Y: training labels
        - update_rule: "backprop", "nesterov_momentum", "rmsprop"
        - epochs: positive integar >= 1
        - miniBatchSize: positive integar >= 1
        - learning_rate
        - validation: [dev_data, dev_labels]
        """
        N = miniBatchSize
        trainTime = 0.0
        predictTime = 0.0
        start_time = time.time()
        
        ## initialize cost variable
        self.training_cost = self.__training_cost(miniBatchSize)
        
        ## blank results
        train_result = np.zeros((epochs,4))
        val_result = np.zeros((epochs,2))
        
        ## initialize velocity for momentum update
        #v = []
        #for i in range(len(self.params)):
        #    v.append(theano.shared( self.params[i].get_value() * 0. )

        ## Training function
        train = theano.function(inputs=[self.X, self.Y],
                                outputs=self.__training_cost(miniBatchSize),
                                updates=self.__weight_update(miniBatchSize, learning_rate,
                                                             update_rule),
                                allow_input_downcast=True)
        for i in range(epochs):
            train_result[i,0] = i+1
            epochStartTime = time.time()
            for batchStart, batchEnd in zip(range(0, len(X), N), range(N, len(X), N)):
                cost = train(X[batchStart:batchEnd], Y[batchStart:batchEnd])
            train_result[i,1] = cost
            train_result[i,2] = np.sqrt(cost)*48
            train_result[i,3] = time.time() - epochStartTime
            if validation != []:
                pred, val_loss, val_rmse = self.predict(validation[0], validation[1])
                val_result[i,0] = val_loss
                val_result[i,1] = val_rmse
            print("\nEpoch:", i+1, "/", epochs)
            print("training time:", train_result[i,3], "s, -----",
                  "loss:", train_result[i,1], ", RMSE:", train_result[i,2])
            if validation != []:
                print("validation loss:", val_result[i,0], ", val RMSE:", val_result[i,1])
        return train_result, val_result
    
    def __weight_update(self, miniBatchSize, learning_rate, update_rule):
        if update_rule == "backprop":
            return self.__backprop(miniBatchSize, learning_rate)
        elif update_rule == "rmsprop":
            return self.__rmsprop(miniBatchSize, learning_rate)
        elif update_rule == "nesterov_momentum":
            return self.__nesterov_momentum(miniBatchSize, learning_rate)
        else:
            return self.__backprop(miniBatchSize, learning_rate)
    
    def __backprop(self, miniBatchSize, learning_rate):
        grads = T.grad(cost=self.__training_cost(miniBatchSize), wrt=self.params)
        updates = []
        for w, grad in zip(self.params, grads):
            updates.append((w, w - grad * learning_rate))
        return updates
    
    def __nesterov_momentum(self, miniBatchSize, learning_rate, mu = 0.5):
        grads = T.grad(cost=self.__training_cost(miniBatchSize), wrt=self.params)
        updates = []
        for w, grad in zip(self.params, grads):
            #v = theano.shared( w.get_value() * 0. )
            #v1_new = mu*v1 - learning_rate*grad
            #delta = (1 + mu)*v1_new - mu*v1
            #updates.append((v1, v1_new))
            #updates.append((w, w + delta))
            updates.append((w, w - grad * learning_rate))
        return updates         
    
    def __rmsprop(self, miniBatchSize, learning_rate, decay=0.9, epsilon=1e-7):
        grads = T.grad(cost=self.__training_cost(miniBatchSize), wrt=self.params)
        updates = []
        for w, grad in zip(self.params, grads):
            ## acc is the cached accumulated gradient square
            acc = theano.shared( w.get_value() )
            acc_new = decay * acc + (1-decay) * (grad**2)
            ## adding gradient scaling
            gradient_scaling = T.sqrt(acc_new) + epsilon
            grad = grad / gradient_scaling
            ## append updates for shared variable acc and weights
            updates.append((acc, acc_new))
            updates.append((w, w - grad*learning_rate))
        return updates
    
    def __get_validation_error(self, pred, Y):
        loss = np.sum( (pred - Y)**2) / (2 * Y.shape[0])
        rmse = np.sqrt(loss)*48
        return loss, rmse    
    
    def predict(self, X, Y):
        predict = theano.function(inputs=[self.X], outputs=self.Y_hat_predict,
                               allow_input_downcast=True)
        pred = predict(X)
        loss, rmse = self.__get_validation_error(pred, Y)
        return pred, loss, rmse
        

def save_layer_params(obj, filename): 
    filename = filename + "_weights"
    f = open(filename, 'wb') 
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close() 

    
def load_saved_params(filename):
    f = open(filename, 'rb')
    layer_params =  cPickle.load(f)
    f.close()
    return layer_params    
       
    
def plot_performance(train_result, validation_result):
    plt.plot(train_result[:,1], linewidth=3, label="train")
    plt.plot(validation_result[:,0], linewidth=3, label="validation")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylim(1e-4, 1e-2)
    #plt.yscale('log')
    plt.show()