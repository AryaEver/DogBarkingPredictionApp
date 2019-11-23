
# -*- coding: utf-8 -*-
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

x_data = np.array(pd.read_csv('./all_contexteven_LLDs.csv'),dtype='str')[:,:500]
y_data = np.array(pd.read_csv('./all_contexteven_LLDs.csv'),dtype='str')[:,500]
#y_data = (pd.read_csv('./mudi_labels.csv', header=None).values[:,1]).astype(str)

#print(np.vstack((np.unique(y_data.astype(int)), np.bincount(y_data.astype(int)))).T)
print('data plus labels shape =',x_data.shape)

#scramble
sss = StratifiedShuffleSplit(10,test_size=0.99, random_state = 15)
for train_idx, test_idx in sss.split(x_data, y_data):
    x_data_tmp , x_test = x_data[train_idx], x_data[test_idx]
    y_data_tmp , y_test = y_data[train_idx], y_data[test_idx]
del x_data_tmp, y_data_tmp
del x_data, y_data

#print(np.vstack((np.unique(y_test.astype(int)), np.bincount(y_test.astype(int)))).T)



#normalize
x_test = x_test.astype(float)
rows, columns = x_test.shape
print('data plus labels shape =',x_test.shape)
for i in range(0,columns):
    x = x_test[:,i]
    x_normed = (x - x.min(0)) / x.ptp(0)
    x_test[:,i] = x_normed
del x_normed,x

#x_test = np.reshape(x_test,(len(x_test),25,20))


PATH = os.getcwd() #"\\UKNOWN\\Documents\\GitHub\\_Tesis" #os.getcwd()

LOG_DIR = PATH + '/tb_dimension_tsne_logdir'

'''
label = ['L-A', 'L-D', 'L-H', 'L-O', 'L-P', 'L-PA', 'L-S1', 'L-S2', 'L-S3',
       'L-TA', 'alone', 'ball', 'fight', 'food', 'play', 'stranger',
       'walk']


label = ['temor', 'otros', 'bienvenida', 'otros2', 'juego2', 'paseo', '+agresivo', 'agresivo', 'peligro',
       'triste', 'solo', 'pelota', 'pelea', 'comida', 'juego', 'extrano'
       'caminar']
'''
#mnist = input_data.read_data_sets(PATH + "/tb_mnist_logdir/data/", one_hot=True)

images = x_test
#labels = np.eye(17)[y_test.reshape(-1).astype(int)]


images = tf.Variable(images, name='images')
#def save_metadata(file):
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
with open(metadata, 'w') as metadata_file:
    for row in range(len(x_test)):
        #c = np.nonzero(labels[::1])[1:][0][row]
        c = y_test[row]#label[y_test[row].astype(int)]
        metadata_file.write('{}\n'.format(c))



with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
    
