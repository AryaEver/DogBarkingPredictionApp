from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import os
import sys
import time
import math
import getopt

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data_conv
import csv
import matplotlib.pyplot as plt

# Confusion matrix imports
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

from input_data_conv import read_dataset
from input_data_conv import read_datasets
from input_data_conv import save_csv

# Confusion matrix functions

def confusion_matrix_table(tlabels, plabels):
    tlabels = pd.Series(tlabels)
    plabels = pd.Series(plabels)

    df_confusion = pd.crosstab(tlabels, plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)

    return df_confusion
    
def error_rate(predictions, labels):
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


param = []

param.append(" --num_hidden 40 --num_epochs 10 --learning_rate 0.002233333 --batch_size 5 --eval_h_batch_size 32 --patch_size 3 --depth 32 --strides 2 --ksize 1  --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 70 --learning_rate 0.00185 --batch_size 8 --eval_h_batch_size 24 --patch_size 7 --depth 20 --strides 1 --ksize 3  --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 70 --learning_rate 0.001466667 --batch_size 5 --eval_h_batch_size 16 --patch_size 4 --depth 32 --strides 1 --ksize 3  --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 20 --learning_rate 0.002233333 --batch_size 7 --eval_h_batch_size 28 --patch_size 7 --depth 32 --strides 2 --ksize 3  --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 10 --learning_rate 0.0007 --batch_size 6 --eval_h_batch_size 8 --patch_size 6 --depth 24 --strides 1 --ksize 1  --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 30 --learning_rate 0.001083333 --batch_size 6 --eval_h_batch_size 32 --patch_size 3 --depth 28 --strides 1 --ksize 1  --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 40 --num_epochs 50 --learning_rate 0.002616667 --batch_size 6 --eval_h_batch_size 28 --patch_size 7 --depth 20 --strides 1 --ksize 1  --padding 1 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 30 --learning_rate 0.0007 --batch_size 5 --eval_h_batch_size 28 --patch_size 4 --depth 16 --strides 2 --ksize 1  --padding 2 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 60 --learning_rate 0.0007 --batch_size 6 --eval_h_batch_size 24 --patch_size 3 --depth 32 --strides 3 --ksize 2  --padding 2 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 10 --learning_rate 0.002616667 --batch_size 9 --eval_h_batch_size 12 --patch_size 5 --depth 32 --strides 1 --ksize 1  --padding 1 --optimizer 1 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 50 --learning_rate 0.001466667 --batch_size 11 --eval_h_batch_size 20 --patch_size 5 --depth 24 --strides 2 --ksize 3  --padding 2 --optimizer 1 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 50 --learning_rate 0.002233333 --batch_size 10 --eval_h_batch_size 16 --patch_size 4 --depth 24 --strides 2 --ksize 1  --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 50 --learning_rate 0.001083333 --batch_size 10 --eval_h_batch_size 8 --patch_size 5 --depth 32 --strides 1 --ksize 1  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 40 --learning_rate 0.002233333 --batch_size 6 --eval_h_batch_size 12 --patch_size 7 --depth 28 --strides 2 --ksize 1  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 50 --learning_rate 0.00185 --batch_size 10 --eval_h_batch_size 12 --patch_size 4 --depth 16 --strides 1 --ksize 2  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 40 --learning_rate 0.001466667 --batch_size 7 --eval_h_batch_size 24 --patch_size 6 --depth 16 --strides 1 --ksize 2  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 40 --learning_rate 0.001083333 --batch_size 9 --eval_h_batch_size 16 --patch_size 3 --depth 20 --strides 1 --ksize 2  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 60 --learning_rate 0.001466667 --batch_size 7 --eval_h_batch_size 20 --patch_size 3 --depth 24 --strides 1 --ksize 2  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 70 --learning_rate 0.002233333 --batch_size 11 --eval_h_batch_size 8 --patch_size 3 --depth 20 --strides 1 --ksize 3  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 70 --learning_rate 0.001466667 --batch_size 10 --eval_h_batch_size 28 --patch_size 6 --depth 28 --strides 1 --ksize 3  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 10 --learning_rate 0.003 --batch_size 10 --eval_h_batch_size 8 --patch_size 7 --depth 32 --strides 3 --ksize 3  --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 70 --learning_rate 0.00185 --batch_size 5 --eval_h_batch_size 12 --patch_size 6 --depth 20 --strides 1 --ksize 1 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 40 --num_epochs 20 --learning_rate 0.003 --batch_size 10 --eval_h_batch_size 24 --patch_size 4 --depth 24 --strides 2 --ksize 1 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 20 --learning_rate 0.00185 --batch_size 7 --eval_h_batch_size 12 --patch_size 5 --depth 24 --strides 2 --ksize 2 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 40 --num_epochs 60 --learning_rate 0.0007 --batch_size 7 --eval_h_batch_size 8 --patch_size 6 --depth 28 --strides 2 --ksize 2 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 70 --learning_rate 0.003 --batch_size 6 --eval_h_batch_size 12 --patch_size 5 --depth 32 --strides 2 --ksize 3 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 10 --learning_rate 0.003 --batch_size 11 --eval_h_batch_size 16 --patch_size 7 --depth 16 --strides 3 --ksize 1 --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 50 --learning_rate 0.0007 --batch_size 9 --eval_h_batch_size 12 --patch_size 4 --depth 16 --strides 2 --ksize 2 --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 40 --num_epochs 30 --learning_rate 0.001466667 --batch_size 8 --eval_h_batch_size 12 --patch_size 5 --depth 20 --strides 3 --ksize 3 --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 60 --learning_rate 0.003 --batch_size 8 --eval_h_batch_size 12 --patch_size 4 --depth 28 --strides 1 --ksize 2 --padding 1 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 20 --learning_rate 0.00185 --batch_size 6 --eval_h_batch_size 16 --patch_size 6 --depth 28 --strides 2 --ksize 2 --padding 1 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 50 --learning_rate 0.001083333 --batch_size 8 --eval_h_batch_size 28 --patch_size 6 --depth 28 --strides 3 --ksize 2 --padding 1 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 30 --learning_rate 0.003 --batch_size 10 --eval_h_batch_size 20 --patch_size 7 --depth 20 --strides 1 --ksize 2 --padding 1 --optimizer 1 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 40 --learning_rate 0.003 --batch_size 9 --eval_h_batch_size 28 --patch_size 3 --depth 32 --strides 2 --ksize 2 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 40 --learning_rate 0.00185 --batch_size 8 --eval_h_batch_size 32 --patch_size 3 --depth 16 --strides 2 --ksize 3 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 10 --learning_rate 0.001083333 --batch_size 6 --eval_h_batch_size 24 --patch_size 7 --depth 28 --strides 2 --ksize 3 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 70 --learning_rate 0.002616667 --batch_size 7 --eval_h_batch_size 28 --patch_size 5 --depth 24 --strides 3 --ksize 3 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 40 --learning_rate 0.002616667 --batch_size 6 --eval_h_batch_size 8 --patch_size 4 --depth 16 --strides 1 --ksize 3 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 40 --num_epochs 40 --learning_rate 0.00185 --batch_size 11 --eval_h_batch_size 16 --patch_size 4 --depth 16 --strides 1 --ksize 1 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 60 --learning_rate 0.003 --batch_size 5 --eval_h_batch_size 24 --patch_size 6 --depth 24 --strides 2 --ksize 1 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 30 --learning_rate 0.00185 --batch_size 9 --eval_h_batch_size 8 --patch_size 5 --depth 28 --strides 2 --ksize 2 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 60 --learning_rate 0.001083333 --batch_size 5 --eval_h_batch_size 16 --patch_size 5 --depth 28 --strides 2 --ksize 2 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 30 --learning_rate 0.001466667 --batch_size 9 --eval_h_batch_size 32 --patch_size 7 --depth 24 --strides 3 --ksize 2 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 60 --learning_rate 0.002233333 --batch_size 9 --eval_h_batch_size 24 --patch_size 5 --depth 16 --strides 1 --ksize 3 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 40 --learning_rate 0.002233333 --batch_size 5 --eval_h_batch_size 20 --patch_size 7 --depth 24 --strides 3 --ksize 1 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 40 --num_epochs 70 --learning_rate 0.001083333 --batch_size 9 --eval_h_batch_size 20 --patch_size 4 --depth 16 --strides 2 --ksize 2 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 60 --learning_rate 0.002616667 --batch_size 10 --eval_h_batch_size 32 --patch_size 3 --depth 20 --strides 3 --ksize 2 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 40 --learning_rate 0.0007 --batch_size 10 --eval_h_batch_size 20 --patch_size 5 --depth 32 --strides 3 --ksize 2 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 50 --learning_rate 0.002616667 --batch_size 5 --eval_h_batch_size 24 --patch_size 3 --depth 16 --strides 3 --ksize 3 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 30 --learning_rate 0.0007 --batch_size 7 --eval_h_batch_size 16 --patch_size 4 --depth 28 --strides 3 --ksize 3 --padding 1 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 20 --learning_rate 0.001466667 --batch_size 6 --eval_h_batch_size 20 --patch_size 4 --depth 20 --strides 1 --ksize 1 --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 10 --learning_rate 0.002616667 --batch_size 8 --eval_h_batch_size 16 --patch_size 4 --depth 24 --strides 1 --ksize 1 --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 64 --num_epochs 10 --learning_rate 0.001083333 --batch_size 7 --eval_h_batch_size 28 --patch_size 4 --depth 24 --strides 2 --ksize 1 --padding 2 --optimizer 1 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 88 --num_epochs 10 --learning_rate 0.0007 --batch_size 11 --eval_h_batch_size 32 --patch_size 3 --depth 20 --strides 1 --ksize 2 --padding 1 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 70 --learning_rate 0.0007 --batch_size 8 --eval_h_batch_size 32 --patch_size 5 --depth 24 --strides 3 --ksize 3 --padding 2 --optimizer 2 --activation 1 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 20 --learning_rate 0.001083333 --batch_size 11 --eval_h_batch_size 12 --patch_size 3 --depth 16 --strides 3 --ksize 3 --padding 1 --optimizer 1 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 72 --num_epochs 50 --learning_rate 0.003 --batch_size 7 --eval_h_batch_size 32 --patch_size 4 --depth 20 --strides 3 --ksize 3 --padding 2 --optimizer 1 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 60 --learning_rate 0.00185 --batch_size 11 --eval_h_batch_size 28 --patch_size 7 --depth 32 --strides 3 --ksize 3 --padding 2 --optimizer 1 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 80 --num_epochs 20 --learning_rate 0.002616667 --batch_size 9 --eval_h_batch_size 32 --patch_size 6 --depth 16 --strides 2 --ksize 1 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 48 --num_epochs 30 --learning_rate 0.002233333 --batch_size 11 --eval_h_batch_size 24 --patch_size 6 --depth 32 --strides 3 --ksize 2 --padding 1 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 32 --num_epochs 30 --learning_rate 0.002616667 --batch_size 11 --eval_h_batch_size 20 --patch_size 6 --depth 28 --strides 1 --ksize 1 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 10 --learning_rate 0.00185 --batch_size 8 --eval_h_batch_size 20 --patch_size 4 --depth 32 --strides 3 --ksize 1 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 96 --num_epochs 20 --learning_rate 0.002233333 --batch_size 8 --eval_h_batch_size 8 --patch_size 3 --depth 16 --strides 2 --ksize 2 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")
param.append(" --num_hidden 56 --num_epochs 20 --learning_rate 0.0007 --batch_size 5 --eval_h_batch_size 8 --patch_size 7 --depth 20 --strides 2 --ksize 3 --padding 2 --optimizer 2 --activation 2 --convolution2d 0 --pool 0 --loss 1")

scores = []

NUM_CHANNELS = 1
SEED = 66478  # Set to None for random seed.
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

num_epochs = 10
base_learning_rate = 0.002233333
patch_size = 3
depth = 32
batch_size = 5
eval_batch_size = 32
num_hidden = 40
pool_type = 0
optimizer_type = 1
loss_type = 1
activation = 1
train_type = 0
convolution = 0
strides = 2
ksize = 1
padding = 'SAME'



goodscores = [0]
#for x in range(0,len(param)):
for x in goodscores:
    print("params#", x ," = ",param[x])
    split = param[x].split()
    
    num_epochs =  int(split[3])
    print(('num_epochs = ', num_epochs))
    
    base_learning_rate = float(split[5])
    print(('base_learning_rate = ', base_learning_rate))
    
    batch_size    =  int(split[7])
    print(('batch_size = ', batch_size))
    
    patch_size =  int(split[11])
    print(('patch_size = ', patch_size))
    
    depth =  int(split[13])
    print(('depth = ', depth))
    
    eval_batch_size =  int(split[9])
    print(('eval_batch_size = ', eval_batch_size))
    
    num_hidden =  int(split[1])
    print(('num_hidden = ', num_hidden))
    
    optimizer_type =  int(split[21])
    print(('optimizer_type = ', optimizer_type))
    
    activation =  int(split[23])
    print(('activation = ', activation))
    
    convolution =  int(split[25])
    print(('convolution = ', convolution))
    
    strides =  int(split[15])
    print(('strides = ', strides))
    
    ksize =  int(split[17])
    print(('ksize = ', ksize))
    
    if split[19] == 0:
        padding = 'SAME'
        print(('padding = ', padding))
    else:
        padding = 'VALID'
        print(('padding = ', padding))
        
    pool_type =  int(split[27])
    print(('pool_type = ', pool_type))
    
    loss_type =  int(split[29])
    print(('pool_type = ', pool_type))
            
    #data_sets = read_dataset('c:\\openSMILE-2.1.0/data/500best.csv')
    #data_sets = read_dataset('c:\\openSMILE-2.1.0/data/mudiraw.csv')
    #data_sets = read_datasets('DataBase/Test/train_data.csv', 'DataBase/Test/validation_data.csv', 'DataBase/Test/test_data.csv')
    
data_features = numpy.load('dogspect.npy')
data_labels = numpy.loadtxt('./mudidogsindividuos.csv',delimiter=',')
            
            
#Split in to test and train        
xtrain = data_features[:4629]
xvalid = data_features[4629:5952]
xtest = data_features[5952:]

ytrain = data_labels[:4629]
yvalid = data_labels[4629:5952]
ytest = data_labels[5952:]

#reshape CNN input
xtrain = numpy.array([x.reshape((1,2816,1)) for x in xtrain])
xvalid = numpy.array([x.reshape((1,2816,1)) for x in xvalid])
xtest = numpy.array([x.reshape((1,2816,1)) for x in xtest])

#One-Hot encoding for classes
#y_train = numpy.array(keras.utils.to_categorical(ytrain,7))
#y_test = numpy.array(keras.utils.to_categorical(ytest,7))
    
for i in range(8):
    plt.subplot(2, 4, i+1)    
    plt.imshow(data_features[i],aspect='auto')
    plt.colorbar();
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
plt.show()  
    
train_data = xtrain#data_sets.train
print('train_data.shape',train_data.shape)
validation_data = xvalid#data_sets.validation
('validation_data.shape',validation_data.shape)
print('validation_data.shape',validation_data.shape)
test_data = xtest#data_sets.test
('test_data.shape',test_data.shape)


train_labels = numpy.asarray(ytrain) #data_sets.train_labels
print('train_labels.shape',train_labels.shape)
validation_labels = numpy.asarray(yvalid) #data_sets.validation_labels
print('validation_labels.shape',validation_labels.shape)
test_labels = numpy.asarray(ytest) #data_sets.test_labels
print('test_labels.shape',test_labels.shape)

train_size = train_labels.shape[0]
print('train_size',train_size)
NUM_LABELS = len(numpy.unique(train_labels))
print('NUM_LABELS',NUM_LABELS)
ATTRIBUTES = 128*22;#train_data.shape[2:4]
print('ATTRIBUTES',ATTRIBUTES)

if padding == 'SAME':
    output_size = int(math.ceil(math.ceil(ATTRIBUTES / strides) / strides))
    print('output_size',output_size)
else:
    output_size = (((ATTRIBUTES - ksize) // strides + 1) - ksize) // strides + 1
        

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
train_data_node = tf.placeholder(tf.float32, shape=(batch_size, 1, ATTRIBUTES, NUM_CHANNELS))
print('placeholder train_data_node',train_data_node)
train_labels_node = tf.placeholder(tf.int64, shape=(batch_size,))
print('placeholder train_labels_node',train_labels_node)
eval_data = tf.placeholder(tf.float32, shape=(eval_batch_size, 1, ATTRIBUTES, NUM_CHANNELS))
print('placeholder eval_data',eval_data)
# The variables below hold all the trainable weights. They are passed an
 # initial value which will be assigned when we call:

# {tf.initialize_all_variables().run()}        
conv1_weights = tf.Variable(tf.truncated_normal([1, patch_size, NUM_CHANNELS, depth], stddev=0.1, seed=SEED, dtype=tf.float32)) # 5x5 filter, depth 32.
print('conv1_weights.shape',conv1_weights.shape)
conv1_biases = tf.Variable(tf.zeros([depth]), dtype=tf.float32)
print('conv1_biases.shape',conv1_biases.shape)
conv2_weights = tf.Variable(tf.truncated_normal([1, patch_size, depth, num_hidden], stddev=0.1, seed=SEED, dtype=tf.float32))
print('conv2_weights.shape',conv2_weights.shape)
conv2_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden], dtype=tf.float32))
print('conv2_biases.shape',conv2_biases.shape)
fc1_weights = tf.Variable(tf.truncated_normal([output_size * num_hidden, 512], stddev=0.1, seed=SEED, dtype=tf.float32)) # fully connected, depth 512. 
print('fc1_weights.shape',fc1_weights.shape)
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
print('fc1_biases.shape',fc1_biases.shape)
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32))
print('fc2_weights.shape',fc2_weights.shape)
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))
print('fc2_biases.shape',fc2_biases.shape)

    
# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train=False):
    # Select convolution
    if convolution == 0:
        print('data.shape',data.shape)
        print('conv1_weights.shape',conv1_weights.shape)
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  
    elif convolution == 1:
        filter1 =  tf.Variable(tf.truncated_normal([patch_size, patch_size, 1, depth], stddev=0.1, seed=SEED)) 
        conv = tf.nn.depthwise_conv2d(data, filter1, strides=[1, 1, 1, 1], padding='SAME')
    elif convolution == 2:
        depthwise_filter =  tf.Variable(tf.truncated_normal([1, patch_size, 1, 1], stddev=0.1, seed=SEED))
        pointwise_filter =  tf.Variable(tf.truncated_normal([1, 1, 1, depth], stddev=0.1, seed=SEED))
        conv = tf.nn.separable_conv2d(data, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')
    
    # Select activation
    if activation == 0:
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        print('conv.shape',conv.shape)
        print('conv1_biases.shape',conv1_biases.shape)
    elif activation == 1:
        relu = tf.nn.relu6(tf.nn.bias_add(conv, conv1_biases))
    elif activation == 2:
        relu = tf.nn.elu(tf.nn.bias_add(conv, conv1_biases))    

    # Select first pooling type
    if pool_type == 0:
        pool = tf.nn.max_pool(relu, ksize=[1, 1, ksize, 1], strides=[1, 1, strides, 1], padding=padding)
        print('1st pool.shape',pool.shape)
    elif pool_type == 1:
        pool = tf.nn.avg_pool(relu, ksize=[1, 1, ksize, 1], strides=[1, 1, strides, 1], padding=padding)
    # Select convolution
    if convolution == 0:
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    elif convolution == 1:
        channel_multiplier = num_hidden // depth
        print(channel_multiplier)
        filter2 = tf.Variable(tf.truncated_normal([1, patch_size,  depth, channel_multiplier], stddev=0.1, seed=SEED))
        conv = tf.nn.depthwise_conv2d(pool, filter2, strides=[1, 1, 1, 1], padding='SAME')
    elif convolution == 2:
        depthwise_filter2 =  tf.Variable(tf.truncated_normal([1, patch_size, depth, 1], stddev=0.1, seed=SEED))
        pointwise_filter2 =  tf.Variable(tf.truncated_normal([1, 1, depth, num_hidden], stddev=0.1, seed=SEED))
        conv = tf.nn.separable_conv2d(pool, depthwise_filter2, pointwise_filter2, strides=[1, 1, 1, 1], padding='SAME')
        
    # Select activation
    if activation == 0:
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        print('conv2_biases.shape',conv2_biases.shape)
        print('conv.shape',conv.shape)
        print('relu.shape',relu.shape)
    elif activation == 1:
        relu = tf.nn.relu6(tf.nn.bias_add(conv, conv2_biases))
    elif activation == 2:
        relu = tf.nn.elu(tf.nn.bias_add(conv, conv2_biases))    
    
    # Select second pooling type
    if pool_type == 0:
        pool = tf.nn.max_pool(relu, ksize=[1, 1, ksize, 1], strides=[1, 1, strides, 1], padding=padding)
        print('2nd pool.shape',pool.shape)
    elif pool_type == 1:
        pool = tf.nn.avg_pool(relu, ksize=[1, 1, ksize, 1], strides=[1, 1, strides, 1], padding=padding)    
        
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    print('final pool_shape',pool_shape)
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    print('reshape.shape',reshape.shape)
    print('x fc1_weights.shape',fc1_weights.shape)
    print('x fc1_biases.shape',fc1_biases.shape)
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    print(hidden)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        
    return tf.matmul(hidden, fc2_weights) + fc2_biases

# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, True)
print('logits =',logits)
print('train_labels_node =',train_labels_node,train_labels)
# Select loss type
if loss_type == 0:
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_node,logits=logits))
    #loss = tf.reduce_max(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))
elif loss_type == 1:
    print('loss 1 logits',logits)
    print('train_labels)_node',train_labels_node)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node,logits=logits))
    #tf.nn.softmax_cross_entropy_with_logits(logits = yPredbyNN, labels=Y)
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype=tf.float32)
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
    base_learning_rate,                # Base learning rate.
    batch * batch_size,  # Current index into the dataset.
    train_size,          # Decay step.
    0.95,                # Decay rate.
    staircase=True)
    
    
# Select momentum
if optimizer_type == 0:
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
elif optimizer_type == 1:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
elif optimizer_type == 2:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
elif optimizer_type == 3:
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=batch)
    
# Select training predictions
train_prediction = tf.nn.softmax(logits)

# Predictions for the test and validation, which we'll compute less often.
eval_prediction = tf.nn.softmax(model(eval_data))

# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < eval_batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, eval_batch_size):
        end = begin + eval_batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: data[-eval_batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
    merged = tf.summary.merge_all
    print(merged)
    train_writer = tf.summary.FileWriter( 'Data/Graph', sess.graph)
    print(train_writer)
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // batch_size):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * batch_size) % (train_size - batch_size)
        batch_data = train_data[offset:(offset + batch_size), ...]
        
        batch_labels = train_labels[offset:(offset + batch_size)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
        #print("batch_data",batch_data.shape)
        #print("batch_labels",batch_labels.shape)
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Param# %d Step %d of %d, (epoch %.2f), of %d, %.1f ms' % (x,step, int(num_epochs * train_size) // batch_size, float(step) * batch_size / train_size , num_epochs ,  1000 * elapsed_time / EVAL_FREQUENCY))
            #print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            #print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
            #print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
            sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('\nTest error: %.1f%%' % test_error)
    
    # Validation Results
    y_pred = numpy.argmax(eval_in_batches(validation_data, sess), 1)
    y_true = validation_labels
    
    print('\nValidation Results')
    val_result = 100.0  * accuracy_score(y_true, y_pred)
    print("Validation accuracy:  %.1f%%" % (val_result))
    print("\nConfusion_matrix")
    print(confusion_matrix_table(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred))    
    
    # Test Results
    y_pred = numpy.argmax(eval_in_batches(test_data, sess), 1)
    y_true = test_labels
    
    print('\nTest Results')
    print('\nX Results')
    print(y_true)
    print('\nY Results')
    print(y_pred)
    test_result = 100.0  * accuracy_score(y_true, y_pred)
    print("Test accuracy:  %.1f%%" % (100.0  * accuracy_score(y_true, y_pred)))
    scores.append(test_result)
    print("\nConfusion_matrix")
    print(confusion_matrix_table(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred))    
    
    
    for i in range(7):
        plt.subplot(2, 4, i+1)    
        weight = sess.run(fc1_weights)[:,i]
        plt.title(i)
        weight = weight.reshape([704,40])
        image = weight[:,0]
        image = image.reshape([32,22])
        plt.imshow(image, cmap=plt.get_cmap('seismic'),aspect='auto')
        plt.colorbar();
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.show()  
    
    for i in range(7):
        plt.subplot(2, 4, i+1)    
        weight = sess.run(fc2_weights)[:,i]
        plt.title(i)
        image = weight.reshape([512,1])
        plt.plot(image)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.show()
    
    # data = numpy.concatenate((train_data, validation_data, test_data))
    # labels = numpy.concatenate((train_labels, validation_labels, test_labels))

    # y_pred = numpy.argmax(eval_in_batches(data, sess), 1)
    # y_true = labels
        
    # print('\nData Results')
    # print("Data accuracy:  %.1f%%" % (100.0  * accuracy_score(y_true, y_pred)))
    # print("\nConfusion_matrix")
    # print(confusion_matrix_table(y_true, y_pred))
    # print("\n", classification_report(y_true, y_pred))
        
    # Save data to csv
    # save_csv(data_sets)

#if __name__ == '__main__':
#    tf.app.run()


