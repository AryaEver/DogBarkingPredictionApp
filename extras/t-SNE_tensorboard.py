import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

x_data = np.array(pd.read_csv('./all_context_LLDs.csv'),dtype='float')[:,:500]
y_data = np.array(pd.read_csv('./all_context_LLDs.csv'),dtype='float')[:,500]


#scramble
sss = StratifiedShuffleSplit(10,test_size=0.001, random_state = 15)
for train_idx, test_idx in sss.split(x_data, y_data):
    x_data_tmp , x_test = x_data[train_idx], x_data[test_idx]
    y_data_tmp , y_test = y_data[train_idx], y_data[test_idx]
del x_data_tmp, y_data_tmp
del x_data, y_data
#normalize
rows, columns = x_test.shape
print('data plus labels shape =',x_test.shape)
for i in range(0,columns):
    x = x_test[:,i]
    x_normed = (x - x.min(0)) / x.ptp(0)
    x_test[:,i] = x_normed
del x_normed,x

x_test = np.reshape(x_test,(len(x_test),25,20))

    
    
logdir = 'C:\\Users\\Oblivion\\Documents\\GitHub\\_Tesis\\tb_dimension_tsne_logdir'
summary_writer = tf.summary.FileWriter(logdir)

embedding_var = tf.Variable(x_test, name="all_context_embedding")

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.join(logdir,'metadata.stv')
embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')

embedding.sprite.single_image_dim.extend([25,20])

projector.visualize_embeddings(summary_writer, config)

with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sesh, os.path.join(logdir,'model.ckpt'))


cols, rows = 25,20

label = ['L-A', 'L-D', 'L-H', 'L-O', 'L-P', 'L-PA', 'L-S1', 'L-S2', 'L-S3',
       'L-TA', 'alone', 'ball', 'fight', 'food', 'play', 'stranger',
       'walk']

sprite_dim = int(np.sqrt(x_test.shape[0]))

sprite_image = np.ones((cols*sprite_dim,rows*sprite_dim))

labels = []
for i in range(sprite_dim):
    for j in range(sprite_dim):
        index = i * j       
        labels.append(label[int(y_test[index])])        
        sprite_image[i * cols: (i+1) * cols, j * rows: (j +1) * rows] = x_test[index].reshape(25,20) * -1 +1
        
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index,label))

plt.imsave(embedding.sprite.image_path,sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray',aspect='auto')
plt.show()