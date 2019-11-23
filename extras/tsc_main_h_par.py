

"""
LSTM for time series classification
Made: 30 march 2016

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""
# Force matplotlib to not use any Xwindows backend.
import sys
import getopt
import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

import os
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

"""Hyperparamaters"""

num_epochs = 30
learning_rate = 0.002233333
batch_size = 11             #Batch size
dropout = 0.7          #Keep probability of the dropout wrapper
max_grad_norm = 7          #Clipping of the gradient before update
num_layers = 2           #Number of stacked LSTM layers
num_steps =  16            #Number of steps to backprop over at every batch
hidden_size = 10            #Number of entries of the cell state of the LSTM
#max_iterations = 1005     #Maximum iterations to train


def confusion_matrix_table(tlabels, plabels):
  tlabels = pd.Series(tlabels)
  plabels = pd.Series(plabels)

  df_confusion = pd.crosstab(tlabels, plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)

  return df_confusion

def main(argv=None):
  print ('Number of arguments:', len(sys.argv), 'arguments.')
  print ('Argument List:', str(sys.argv))
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["max_grad_norm=", "num_epochs=", "learning_rate=" ,"dropout=", "num_layers=",  "num_steps=", "hidden_size=", "batch_size="])
  except getopt.GetoptError:
    print ('tsc_main_h_par.py --max_grad_norm <> --num_epochs <> --learning_rate <> --dropout <> --num_layers <> --num_steps <> --hidden_size <> --batch_size <>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print ('tsc_main_h_par.py --max_grad_norm <> --num_epochs <> --learning_rate <> --dropout <> --num_layers <> --num_steps <> --hidden_size <> --batch_size <>')
      sys.exit()
    elif opt == '--max_grad_norm':
      global max_grad_norm
      max_grad_norm =  int(arg)
    elif opt == '--num_epochs':
      global num_epochs
      num_epochs =  int(arg)
    elif opt == '--learning_rate':
      global learning_rate
      learning_rate = float(arg)
    elif opt == '--dropout':
      global dropout
      dropout=  float(arg)
    elif opt == '--num_layers':
      global num_layers
      num_layers =  int(arg)
    elif opt == '--num_steps':
      global num_steps
      num_steps =  int(arg)
    elif opt == '--hidden_size':
      global hidden_size
      hidden_size =  int(arg)
    elif opt == '--batch_size':
      global batch_size
      batch_size  =  int(arg)

  def normalize_matrix(matrix):
  	columns = matrix.shape[1]
  	
  	for i in range(0, columns):
  		x = matrix[:, i]
  		x_normed = (x - x.min(0)) / x.ptp(0)
  		matrix[:,i] = x_normed
  	return matrix

  def read_datasets(train_csv, validation_csv, test_csv):
  	class Data(object): pass
  	
  	data_sets = Data()
  	
  	train = np.genfromtxt(train_csv, delimiter=',', dtype=float)
  	train = train.astype(np.float)
  	validation = np.genfromtxt(validation_csv, delimiter=',', dtype=float)
  	validation = validation.astype(np.float)
  	test = np.genfromtxt(test_csv, delimiter=',', dtype=float)
  	test = test.astype(np.float)
  	
  	rows, columns = train.shape
  	arr = np.arange(rows)
  	np.random.shuffle(arr)
  	matrix = np.zeros((rows, columns))
  	
  	for i in range (0, rows):
  		matrix[i] = train[arr[i],:]

  	mTrain = matrix[:, 0:-1]
  	train_labels = matrix[:, columns - 1]
  	
  	rows, columns = validation.shape
  	arr = np.arange(rows)
  	np.random.shuffle(arr)
  	matrix = np.zeros((rows, columns))
  	
  	for i in range (0, rows):
  		matrix[i] = validation[arr[i],:]

  	mValidation = matrix[:, 0:-1]
  	validation_labels = matrix[:, columns - 1]
  	
  	rows, columns = test.shape
  	arr = np.arange(rows)
  	np.random.shuffle(arr)
  	matrix = np.zeros((rows, columns))
  	
  	for i in range (0, rows):
  		matrix[i] = test[arr[i],:]

  	mTest = matrix[:, 0:-1]
  	test_labels = matrix[:, columns - 1]	
  	
  	mData = np.concatenate((mTrain, mValidation, mTest))
  	mData = normalize_matrix(mData)

  	trainSize = len(mTrain)
  	validationSize = len(mValidation)
  	testSize = len(mTest)
  	
  	mTrain = mData[0:trainSize, :]
  	mValidation = mData[trainSize:trainSize + validationSize, :]
  	mTest = mData[-testSize:, :]

  	train_labels = train_labels.astype(np.uint8)
  	validation_labels = validation_labels.astype(np.uint8)
  	test_labels = test_labels.astype(np.uint8)
  	
  	data_sets.train = mTrain
  	data_sets.validation = mValidation
  	data_sets.test = mTest
  	data_sets.train_labels = train_labels
  	data_sets.validation_labels = validation_labels
  	data_sets.test_labels = test_labels

  	return data_sets

  def sample_batch(X_train,y_train,batch_size,num_steps):
    """ Function to sample a batch for training"""
    N,data_len = X_train.shape
    ind_N = np.random.choice(N,batch_size,replace=False).astype(int)[:]
    ind_start = np.random.choice(data_len-num_steps,1).astype(int)[0]
    X_batch = X_train[ind_N,ind_start:ind_start+num_steps]
    y_batch = y_train[ind_N]
    
    return X_batch,y_batch

  def check_test(X_test,y_test,batch_size,num_steps):
    """ Function to check the test_accuracy on the entire test set"""
    N = X_test.shape[0]
    num_batch = np.floor(N/batch_size).astype(int)
    test_acc = np.zeros(num_batch)
    test_predictions=[]
    for i in range(num_batch):
      X_batch, y_batch = sample_batch(X_test,y_test,batch_size,num_steps)
      test_acc[i], test_pred = sess.run([accuracy, predictions], feed_dict = {input_data: X_batch, targets: y_batch, keep_prob:1})
      test_predictions =  np.append(test_predictions,test_pred)   
    return np.mean(test_acc), test_predictions





  """Load the data"""
  # dummy = True
  # if dummy:
    # data_train = np.loadtxt(dir_path + 'UCR_TS_Archive_2015/Two_Patterns/Two_Patterns_TRAIN',delimiter=',')
    # data_test_val = np.loadtxt(dir_path + 'UCR_TS_Archive_2015/Two_Patterns/Two_Patterns_TEST',delimiter=',')
  # else:
    # data_train = np.loadtxt('data_train_dummy',delimiter=',')
    # data_test_val = np.loadtxt('data_test_dummy',delimiter=',')
  # data_test,data_val = np.split(data_test_val,2)
  # X_train = data_train[:,1:]
  # X_val = data_val[:,1:]
  # X_test = data_test[:,1:]
  # N = X_train.shape[0]
  # Ntest = X_test.shape[0]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  # y_train = data_train[:,0]-1
  # y_val = data_val[:,0]-1
  # y_test = data_test[:,0]-1
  # num_classes = len(np.unique(y_train))

  data_sets = read_datasets('Data/Test/train_data.csv','Data/Test/validation_data.csv', 'Data/Test/test_data.csv')

  X_train = data_sets.train
  train_size = X_train.shape[0]
  max_iterations = int((num_epochs * train_size) // batch_size)
  X_val = data_sets.validation
  X_test = data_sets.test

  N = X_train.shape[0]
  Ntest = X_test.shape[0]

  y_train = data_sets.train_labels
  y_val = data_sets.validation_labels
  y_test = data_sets.test_labels

  num_classes = len(np.unique(y_train))

  # Collect the costs in a numpy fashion
  epochs = np.floor(batch_size*max_iterations / N)
  print('Train with approximately %d epochs' %(epochs))
  if max_iterations%100 == 0:
    perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))
  else:
   perf_collect = np.zeros((3,int(np.floor(max_iterations /100)) + 1 )) 

  """Place holders"""
  input_data = tf.placeholder(tf.float32, shape=(batch_size, num_steps), name = 'input_data')
  print(input_data)
  targets = tf.placeholder(tf.int64, shape=(batch_size), name='Targets')
  print(targets)
  #Used later on for drop_out. At testtime, we pass 1.0
  keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

  with tf.name_scope("LSTM_setup") as scope:
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    #We have only one input dimension, but we generalize our code for future expansion
    inputs = tf.expand_dims(input_data, 2)

  #Define the recurrent nature of the LSTM
  with tf.name_scope("LSTM") as scope:
    outputs = []
    state = initial_state
    with tf.variable_scope("LSTM_state"):
      for time_step in range(num_steps):
       if time_step > 0: tf.get_variable_scope().reuse_variables() #Re-use variables only after first time-step
       (cell_output, state) = cell(inputs[:, time_step, :], state)
       outputs.append(cell_output)       #Now cell_output is size [batch_size x hidden_size]
    output = tf.reduce_mean(tf.pack(outputs),0)


  #Generate a classification from the last cell_output
  #Note, this is where timeseries classification differs from sequence to sequence
  #modelling. We only output to Softmax at last time step
  with tf.name_scope("Softmax") as scope:
    with tf.variable_scope("Softmax_params"):
      softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
      softmax_b = tf.get_variable("softmax_b", [num_classes])
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    #Use sparse Softmax because we have mutually exclusive classes
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,targets,name = 'Sparse_softmax')
    cost = tf.reduce_sum(loss) / batch_size
  with tf.name_scope("Evaluating_accuracy") as scope:
    predictions  =  tf.argmax(logits,1)
    correct_prediction = tf.equal(predictions,targets)
    accuracy  = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    
  """Optimizer"""
  with tf.name_scope("Optimizer") as scope:
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_op = optimizer.apply_gradients(gradients)
    # Add histograms for variables, gradients and gradient norms.
    # The for-loop loops over all entries of the gradient and plots
    # a histogram. We cut of
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      h1 = tf.histogram_summary(variable.name, variable)
      h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
      h3 = tf.histogram_summary(variable.name + "/gradient_norm", tf.global_norm([grad_values]))

  #Final code for the TensorBoard
  merged = tf.merge_all_summaries()

  """Session time"""
  sess = tf.Session() #Depending on your use, do not forget to close the session
  writer = tf.train.SummaryWriter(dir_path + "/logs/log_tb")
  sess.run(tf.initialize_all_variables())


  step = 0
  cost_train_ma = -np.log(1/float(num_classes)+1e-9)
  for i in range(max_iterations):
    # Calculate some sizes
    N = X_train.shape[0]
    #Sample batch for training
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size,num_steps)

    #Next line does the actual training
    cost_train, _ = sess.run([cost,train_op],feed_dict = {input_data: X_batch,targets: y_batch,keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    if i%100 == 0:
      #Evaluate training performance
      perf_collect[0,step] = cost_train
      #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size,num_steps)
      result = sess.run([cost,merged,accuracy],feed_dict = {input_data: X_batch, targets: y_batch, keep_prob:1})
      cost_val = result[0]
      perf_collect[1,step] = cost_val
      acc_val = result[2]
      perf_collect[2,step] = acc_val
      print('At %5.0f out of %5.0f: Cost is TRAIN %.3f(%.3f) VAL %.3f and val acc is %.3f' %(i,max_iterations,cost_train,cost_train_ma,cost_val,acc_val))


      #Write information to TensorBoard
      summary_str = result[1]
      writer.add_summary(summary_str, i)
      writer.flush()

      step +=1
  acc_test, predictions = check_test(X_test,y_test,batch_size,num_steps)

  """Additional plots"""
  print('The accuracy on the test data is %.3f' %(acc_test))
  plt.plot(perf_collect[0],label='Train')
  plt.plot(perf_collect[1],label = 'Valid')
  plt.plot(perf_collect[2],label = 'Valid accuracy')
  plt.axis([0, step, 0, np.max(perf_collect)])
  plt.legend()
  plt.show()
  #y_val = y_val[1849:3724]
  print('\nY Results')
  print('Test accuracy is:  %.1f%%' % (100.0  * accuracy_score(y_test[0:predictions.shape[0]],predictions)))
  print('\nConfusion_matrix')
  print(confusion_matrix_table(y_test[0:predictions.shape[0]],predictions))
  print('\n', classification_report(y_test[0:predictions.shape[0]],predictions)) 

if __name__ == '__main__':
  tf.app.run()