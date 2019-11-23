#List of data bases to evaluate --options: 
'''
data_set=(['all_context','all_individ','all_breed','all_sex','all_age','mudi_context','mudi_individ','mudi_sex','mudi_age','mescalina_context',
'mescalina_individ','mescalina_context','mescalina_breed','mescalina_sex','mescalina_age',
'mescalina2017_context','mescalina2017_individ','mescalina2017_context','mescalina2017_breed',
'mescalina2017_sex','urban','mnist'])
'''
data_set=['all_4context']
#List of input data types to feed the network --options: input_data=['Raw','Spect','LLDs']
input_data=['LLDs']
#generate_data = False generate .csv containing desired input type representation to be loaded (slows down processsing)
#List of neural networks to use --options:
#neural_network=['lstm','gru','conv_lstm','conv_gru','convolutional','gan']
neural_network=['convolutional']
#List of Parameters to Tune
epochs=200 #max epoch to train
#training paramteres

# Important Warning !!! all parameter values should be arranged from max to min
# due Tensorflow lack of gpu max memory allocation reset has no fix yet
# if this error shows, decrease max values from some parameters until it runs
###############################################################################
batch_size=[256]
learning_rate=[1e-3]
drop_out=[0.5]
#optimizer=['adam','sgd','rmsprop','adamax']
optimizer=['adam']
#architecure hyper-parameters
#dense_layers=[3,1,0]
dense_layers=[3]
#hidden_neurons=[1000,850,750,512,256,128,90,72]
hidden_neurons=[1000]
#convolutional parameters only applied if convolutional selected
#conv_layers=[3,2,1] #at least 1 conv layer  is needed
conv_layers=[3]
filter_depth=[6]
#filter_depth=[32,24,12]
kernel_size=[3]
#kernel_size=[6,5,3]
pooling_size=[1]
#pooling_size=[4,3,2,1]

#Run_options special actions
use_augmented_dataset = False #use Augmented data set High RAM
use_validation_set = True #use 10% of data to validate, 10% for test and 80% to train
shuffle_data_sets = True #stratified shuffling so all the sets contain random samples but even distributions
normalize_data = True #normalize data range values between 1 and 0
save_load_models = True #save trained models and load existing models if already trained
early_stopping = True #if there is no improvment after several epochs stop training 
get_learned_features = False #get learned features on first layers (slow processing) 
get_saliency_map = False #gets activation weights for each categories filtered for better visulizaiton (slows down processing)
save_pictures = True #save all plot as files
show_summary = True #shows summary of model architecture in console for each combination
display_pictures = False #show plots graphics if using graphic os

###############################################################################
############################ MODULE IMPORT ####################################
###############################################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras import backend as k
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, CuDNNLSTM, CuDNNGRU,TimeDistributed
#from keras.preprocessing.image import ImageDataGenerator
import pickle
import time
import operator
import gc
from numba import cuda
from get_confusion_matrix import get_confusion_matrix
from visualize_layer_features import visualize_layer_features
from saliency_map import saliency_map
###############################################################################
############################ LOAD DATABASE ####################################
###############################################################################
chunksize = 1000
num_of_runs = len(data_set)*len(input_data)
current_run = num_of_runs
print(device_lib.list_local_devices())
final_list = []
if display_pictures == False:
    plt.ioff()
for input_type in input_data:
    for name in data_set:
        print(name,input_type)
#   #   #
        try:
            size = None
            size = 6615 if 'mudi' in name else size
            size = 6077 if 'mescalina' in name else size
            size = 8732 if 'urban' in name else size
            size = 8970 if '2017' in name else size
            size = 14000 if 'all' in name else size

            size = size*2 if use_augmented_dataset == True else size
            start_time = time.time()
            
            file = ['./'+name+'_'+input_type+'.csv','./'+name+'_AugmentedNoise_'+input_type+'.csv']
            if use_augmented_dataset == True:
                file = file[1] 
            else: 
                file = file[0]
            chunks = []
            for idx, chunk in enumerate(pd.read_csv(file,chunksize=chunksize, header=None)):
                if idx ==0:
                    slicesize = int(size/chunksize)
                    print('loading',file,' data by chunk size ',chunksize, ' of ',size, ' total')
                chunks.append(chunk)
                out_file = "chunk/data_{}".format(idx)
                with open(out_file, "wb") as f:
                    pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)    
                if idx == 0:
                    totaltime = (slicesize + 1)*(time.time()-start_time)
                print('data loaded =',(idx+1)*chunksize,' time remaining =',totaltime-(time.time()-start_time))
                
            x_train = pd.DataFrame()    
            print("concatenating as data frame")
            x_train = pd.concat(chunks)
            x_train = x_train.as_matrix()
            name = name+'_AugmentedNoise' if use_augmented_dataset == True else name

            ###############################################################################
            ############################ DATA PROCESSING ##################################
            ###############################################################################
            rows, columns = x_train.shape
            print('data plus labels shape =',x_train.shape)
            if normalize_data == True:
                for i in range(0,columns-1):
                    x = x_train[:,i]
                    x_normed = (x - x.min(0)) / x.ptp(0)
                    x_train [:,i] = x_normed

            #this block asumes that last column in .csv contains the labels for each category
#   #   #   #
            try:
                labels = None
                labels = np.unique(pd.read_csv('./mudi_labels.csv', header=None).values[:,1]) if 'mudi_context' in name else labels
                labels = np.unique(pd.read_csv('./mudi_labels.csv', header=None).values[:,3]) if 'mudi_individ' in name else labels
                labels = np.unique(pd.read_csv('./labels_mudidogs_all.csv', ).values[:,5]) if 'mudi_sex' in name else labels
                labels = np.unique(pd.read_csv('./mudi_labels.csv' ).values[:,5]) if 'mudi_age' in name else labels
               
                labels = np.unique(pd.read_csv('./mescalina_labels.csv', header=None).values[:,3]) if 'mescalina_context' in name else labels
                labels = np.unique(pd.read_csv('./mescalina_labels.csv', header=None).values[:,1]) if 'mescalina_individ' in name else labels
                labels= np.unique(pd.read_csv('./mescalina_labels.csv', header=None).values[:,7]) if 'mescalina_breed' in name else labels
                labels= np.unique(pd.read_csv('./mescalina_labels.csv', header=None).values[:,5]) if 'mescalina_sex' in name else labels
                labels= np.unique(pd.read_csv('./mescalina_labels.csv', header=None).values[:,6]) if 'mescalina_age' in name else labels
                
                labels= np.unique(pd.read_csv('./mescalina2017.csv', header=None).values[:,2]) if 'mescalina2017_individ' in name else labels
                labels= np.unique(pd.read_csv('./mescalina2017.csv', header=None).values[:,1]) if 'mescalina2017_context' in name else labels
                labels= np.unique(pd.read_csv('./mescalina2017.csv', header=None).values[:,4]) if 'mescalina2017_breed' in name else labels
                labels= np.unique(pd.read_csv('./mescalina2017.csv', header=None).values[:,3]) if 'mescalina2017_sex' in name else labels

                labels = np.unique(pd.read_csv('./urban_labels.csv', header=None).values[:,1]) if 'urban' in name else labels
                n_classes = len(labels)

            except:
                print('no label file found, enumerated categories will be shown in plots')
                labels = np.unique(x_train[:,x_train.shape[1]-1])
                n_classes = len(labels)
#   #   #   #    
            try:
                print((np.vstack((np.unique(x_train[:,columns-1]), np.bincount((x_train[:,columns-1]).astype(int)))).T).astype(int))
                if use_validation_set == True:
                    train,validation,test = int(rows*0.8),int(rows*0.1),int(rows*0.1)
                    x_test = x_train[train+validation:,:]
                    x_validation = x_train[train:train+validation,:]
                    x_train = x_train[:train,:]
                    
                    rows, columns = x_validation.shape
                    y_validation = x_validation[:,columns-1].astype(np.int64)
                    x_validation = x_validation[:,0:columns-1]
                    
                else:
                    train,test = int(rows*0.9),int(rows*0.1)
                    x_test = x_train[train:,:]
                    x_train = x_train[:train,:]
                
                #carry on
                rows, columns = x_train.shape
                y_train = x_train[:,columns-1].astype(np.int64)
                x_train = x_train[:,0:columns-1]
                   
                rows, columns = x_test.shape
                y_test = x_test[:,columns-1].astype(np.int64)
                x_test = x_test[:,0:columns-1]
    
                #defining size for plots, shape for network input and filter size for visualizations
                if input_type == 'LLDs' or input_type =="Raw":
                    img_shape = (len(x_test[0]),1,1)
                    img_size = (len(x_test[0]),1)
                    img_filter_size = (1,len(x_test[0]),1,1)
                if input_type == 'Spect':
                    img_shape = (128,18,1)
                    img_size = (128,18)
                    img_filter_size = (1,128,18,1)
                    
                x_train = np.array([x.reshape( (img_shape) ) for x in x_train])
                print('x_train.shape',x_train.shape)
                x_test = np.array([x.reshape( (img_shape) ) for x in x_test])
                print('x_test.shape',x_test.shape)
                if use_validation_set == True:
                    x_validation = np.array([x.reshape( (img_shape) ) for x in x_validation])
                    print('x_validation.shape',x_validation.shape)
                
            except Exception as e:
                print(e)
                sys.exit()
                
#   #   #        
        except:
            if name == 'mnist':
                input_type = 'Img'
                use_validation_set = False
                
                print('loading mnist DataSet')                
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
                x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
                input_shape = (28, 28, 1)
                # Making sure that the values are float so that we can get decimal points after division
                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')
                # Normalizing the RGB codes by dividing it to the max RGB value.
                x_train /= 255
                x_test /= 255
                print('x_train shape:', x_train.shape)
                print('Number of images in x_train', x_train.shape[0])
                print('Number of images in x_test', x_test.shape[0])
                img_shape = (28,28,1)
                img_size = (28,28)
                img_filter_size = (1,28,28,1)
                labels = np.unique(y_train)
                n_classes = len(labels)
            else:
                print('file not found -> name format should be:',file)
                sys.exit()
        
        ######################### Plot First 9 Samples ########################
        fig, ax = plt.subplots(3, 3, figsize = (18, 9))
        fig.suptitle('First 36 images in dataset')
        fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
        for x, y in [(i, j) for i in range(3) for j in range(3)]:
            if input_type == 'LLDs' or input_type =="Raw":
                ax[x, y].plot(x_train[x + y * 3].reshape(img_size))
            if input_type == 'Spect' or input_type == 'Img':
                ax[x, y].imshow(x_train[x + y * 3].reshape(img_size),aspect='auto')
        if save_pictures == True:
            plt.savefig(name+input_type+'Inputs.png', bbox_inches='tight')
        if display_pictures == True:
            fig.show()
            plt.show()   
        plt.clf()
        plt.cla()
        plt.close()
        ####################### Shuffle Even Distribution #####################
        if shuffle_data_sets == True:
            print('Stratified shuffling...')
            if use_validation_set == True:
                x_train = np.concatenate((x_test,x_validation,x_train),axis=0)
                y_train = np.concatenate((y_test,y_validation,y_train),axis=0)
                
                sss = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state = 15)
                for train_idx, test_idx in sss.split(x_train, y_train):
                    x_train_tmp, x_test = x_train[train_idx], x_train[test_idx]
                    y_train_tmp, y_test = y_train[train_idx], y_train[test_idx]
                split = int(len(x_test)/2)
                x_train = x_train_tmp
                del x_train_tmp
                y_train = y_train_tmp
                del y_train_tmp
                x_validation =  x_test[split:,:]
                y_validation =  y_test[split:]
                x_test =  x_test[:split,:]
                y_test =  y_test[:split]
                
            elif use_validation_set == False:
                x_train = np.concatenate((x_test,x_train),axis=0)
                y_train = np.concatenate((y_test,y_train),axis=0)
                sss = StratifiedShuffleSplit(10, random_state = 15)
                for train_idx, test_idx in sss.split(x_train, y_train):
                    x_train_tmp, x_test = x_train[train_idx], x_train[test_idx]
                    y_train_tmp, y_test = y_train[train_idx], y_train[test_idx]
                x_train = x_train_tmp
                del x_train_tmp
                y_train = y_train_tmp
                del y_train_tmp
                
            print('Finish stratified shuffling...')

        if use_validation_set == True:
            print('train',x_train.shape,'val',x_validation.shape,'test',x_test.shape)
        elif use_validation_set == False:
            print('train',x_train.shape,'test',x_test.shape)
        
        print('Number of occurence for each number in training data:')
        print(np.vstack((np.unique(y_validation), np.bincount(y_validation))).T)
        plt.figure()
        fig1 = sns.countplot(y_train)
        if save_pictures == True:
            plt.savefig(name+input_type+'Train.png', bbox_inches='tight')
        if display_pictures == True:
            fig.show()
            plt.show() 
        plt.clf()
        plt.cla()
        plt.close()
        if use_validation_set == True:
            print('Number of occurence for each number in validation data')
            print(np.vstack((np.unique(y_test), np.bincount(y_test))).T)
            plt.figure()
            fig1 = sns.countplot(y_test)
            if save_pictures == True:
                plt.savefig(name+input_type+'validation.png', bbox_inches='tight')
            if display_pictures == True:
                fig.show()
                plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        print('Number of occurence for each number in test data')
        print(np.vstack((np.unique(y_test), np.bincount(y_test))).T)
        plt.figure()
        fig1 = sns.countplot(y_test)
        if save_pictures == True:
            plt.savefig(name+input_type+'Test.png', bbox_inches='tight')
        if display_pictures == True:
            fig.show()
            plt.show() 
        plt.clf()
        plt.cla()
        plt.close()
        # transform training label to one-hot encoding
        y_train = np.array(keras.utils.to_categorical(y_train, n_classes))
        y_test = np.array(keras.utils.to_categorical(y_test, n_classes))
        if use_validation_set == True:
            y_validation = np.array(keras.utils.to_categorical(y_validation, n_classes))
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_train)
        #y_train = lb.transform(y_train)
        ###############################################################################
        ############################ NEURAL NETWORK ###################################
        ###############################################################################
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        k.tensorflow_backend.set_session(sess)
       #print(cuda.list_devices(),cuda.get_current_device(),cuda.cudadrv.driver.Device(0).compute_capability)
       #print(cuda.current_context().get_memory_info())
        print('backend=',k.backend())
        #early stopping configuration
        estop = keras.callbacks.EarlyStopping(monitor='val_acc',min_delta=1e-4,patience=5,mode='auto')
#   #   #iterate each network in list
        for network in neural_network:
            
            model_params = False
            output_scores = []
            output_values = []
            model_list = []
            index = 0
            train_combinations = len(batch_size)*len(learning_rate)*len(drop_out)
            if 'conv' in network:
                hyper_combinations = len(filter_depth)*len(pooling_size)*len(kernel_size)*len(dense_layers)*len(hidden_neurons)*len(conv_layers)
            else:
                hyper_combinations = len(dense_layers)*len(hidden_neurons)
                
            remaining_combinations = train_combinations + hyper_combinations -1
            #Find best training parameters for best network architecture
            batch = batch_size
            learning = learning_rate
            drop = drop_out
            dense = dense_layers
            hidden = hidden_neurons
            
            if 'conv' in network:
                convolutions = conv_layers
                filtering = filter_depth
                kernel = kernel_size
                pooling = pooling_size
                
            if 'conv_lstm' == network or 'conv_gru' == network:
                x_train = np.array([x.reshape( (img_filter_size) ) for x in x_train])
                x_test = np.array([x.reshape( (img_filter_size) ) for x in x_test])
                if use_validation_set == True:
                    x_validation = np.array([x.reshape( (img_filter_size) ) for x in x_validation])
            
            if 'lstm' == network or 'gru' == network:        
                x_train = np.reshape(x_train,(len(x_train),img_size[0],img_size[1]))
                if use_validation_set == True:
                    x_validation = np.reshape(x_validation,(len(x_validation),img_size[0],img_size[1]))
                x_test = np.reshape(x_test,(len(x_test),img_size[0],img_size[1]))
                
            if 'convolutional' == network:
                x_train = np.array([x.reshape( (img_shape) ) for x in x_train])
                x_test = np.array([x.reshape( (img_shape) ) for x in x_test])
                if use_validation_set == True:
                    x_validation = np.array([x.reshape( (img_shape) ) for x in x_validation])
            
            for lr in learning_rate:
                for bs in batch_size:                   
                    for do in drop_out:
#   #   #   #   #   #   #Find best hyper-parameter for network architecture
                        
                        if model_params == True:
                            print('Searching for the best model training parameters')
                            #o = neurons,layer,conv,lr,bs,do,fd,ps,ks,log_name
                            if 'conv' in network:
                                o0,o1,o2,o3,o4,o5,o6,o7,o8,o9 = output_values[index]
                                dense = [o2]
                                hidden = [o0]
                                convolutions = [o3]
                                filtering = [o6]
                                kernel = [o8]
                                pooling = [o7]
                            #o = neurons,layer,lr,bs,do,log_name
                            if 'lstm' ==network or 'gru' == network:
                                o0,o1,o2,o3,o4,o5 = output_values[index]
                                dense = [o1]
                                hidden = [o0]   
                        if model_params == False:
                            print('Searching for the best model Hyper-parameters')

#   #   #   #   #   #   #Load selected networks
                        if 'lstm' == network or 'gru' == network:
                            for layer in dense:
                                for neurons in hidden:
                                    log_name = ("{}neurons_{}dense_{}Lrate_{}batch_{}dout").format(neurons,layer,lr,bs,do)
                                    
                                    try:
                                        model = load_model('./models/'+network+'_'+name+'_'+input_type+log_name+'.h5')
                                            
                                    except:
                                        print('Building model...',log_name)
                                        log_name = ("{}neurons_{}dense_{}Lrate_{}batch_{}dout").format(neurons,layer,lr,bs,do)
                                        
                                        
                                        model = Sequential()
                                        if 'lstm' == network:
                                            model.add(CuDNNLSTM(neurons, input_shape=(img_size), return_sequences=True))
                                            #model.add(CuDNNLSTM(neurons, input_shape=(img_size)))
                                        if 'gru' == network:
                                            model.add(CuDNNGRU(neurons, input_shape=(img_size), return_sequences=True))
                                            
                                        model.add(Dropout(do))

                                        if 'lstm' == network:
                                            model.add(CuDNNLSTM(neurons))
                                            #model.add(CuDNNLSTM(neurons, input_shape=(img_size)))
                                        if 'gru' == network:
                                            model.add(CuDNNGRU(neurons))
                                        
                                        model.add(Dropout(do))
                                        
                                        
                                        #Generating dense layers
                                        for l in range(layer):
                                            model.add(Dense(neurons))
                                            model.add(Activation('relu'))
                                            model.add(Dropout(do))
                                            
                                        model.add(Dense(n_classes))
                                        model.add(Activation('softmax'))
                                        
                                        opt = keras.optimizers.Adam(lr=lr, decay=1e-6)
                                        #model training
                                        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
                                        
                                        #tensorboard graph (visualize by typing in console: tensorboard log_dir=logs)
                                        if 'lstm' == network:
                                            tb = keras.callbacks.TensorBoard(log_dir='./logs_lstm_'+name+'_'+input_type+'/{}'.format(log_name))
                                        if 'gru' == network:
                                            tb = keras.callbacks.TensorBoard(log_dir='./logs_gru_'+name+'_'+input_type+'/{}'.format(log_name))
                                        
                                        callbacks_list = [tb] if early_stopping == False else [tb,estop]
                                        val_data = (x_validation,y_validation) if use_validation_set == True else (x_test,y_test)
    
                                        if show_summary == True:
                                            model.summary()
                                        model.fit(x_train,
                                                  y_train,
                                                  batch_size=bs,
                                                  epochs=epochs,
                                                  validation_data=val_data,
                                                  callbacks=callbacks_list)
                                        
                                        model.save('./models/'+network+'_'+name+'_'+input_type+log_name+'.h5')
    
                                    values = neurons,layer,lr,bs,do,log_name

                                    print('Testing model...')
                                    score, acc = model.evaluate(x_test, y_test, verbose=1)
                                    print('\nLoss:', score, '\nAcc:', acc)
                                    test_score = (acc*100).astype(str)
                                    output_scores.append(test_score)
                                    output_values.append(values)
                                    current_model = str(current_run)+'-'+str(remaining_combinations)+'-'+test_score+'%'+' \t'+network+'_'+name+'_'+input_type+'\t'+'N'+str(neurons)+' Dense'+str(layer)+' Lr'+str(lr)+' Bsz'+str(bs)+' Dout'+str(do)
                                    model_list.append(current_model)
                                    final_list.append(current_model)
                                    print(*model_list, sep='\n')
                                    remaining_combinations = remaining_combinations -1
                                    
                                    del model
                                    gc.collect()
                                    k.clear_session()
                                    tf.keras.backend.clear_session()
                                    tf.reset_default_graph()
                                   #print(cuda.current_context().get_memory_info())
    
#   #   #   #   #   #   #   #                                                                    
                            if model_params == False:
                                index, score = max(enumerate(output_scores), key=operator.itemgetter(1))
                                print('keeping the best score =',score,' params =',output_values[index])
                                model_params = True
                                        
                        if 'conv' in network:
                            for fd in filtering:
                                for ps in pooling:
                                    for ks in kernel:
                                        for conv in convolutions:
                                            for layer in dense:
                                                for neurons in hidden:
                                                    log_name = ("{}neurons_{}dense_{}conv_{}Lrate_{}batch_{}dout_{}depth_{}pool_{}kernel").format(neurons,layer,conv,lr,bs,do,fd,ps,ks)
    
                                                    try:
                                                        print('Loading model '+network+'_'+name+'_'+input_type+'.h5')
                                                        model = load_model('./models/'+network+'_'+name+'_'+input_type+log_name+'.h5')

                                                    except:
                                                        print('Building model...',log_name) 
    
                                                        model = Sequential()
                                                        #Convlutional layer input shapes
                                                        
                                                        #First conv layer
                                                        if input_type == 'LLDs' or input_type == "Raw":
                                                            if 'convolutional' == network:
                                                                model.add(Conv2D(fd, (ks, 1), input_shape = img_shape, kernel_initializer = 'normal'))
                                                            else:
                                                                model.add(TimeDistributed(Conv2D(fd, (ks, 1),  kernel_initializer = 'normal'), input_shape = img_filter_size))
                                                        if input_type == 'Spect' or input_type == 'Img':
                                                            if 'convolutional' == network:
                                                                model.add(Conv2D(fd, (ks, ks), input_shape = img_shape, kernel_initializer = 'normal'))
                                                            else:
                                                                model.add(TimeDistributed(Conv2D(fd, (ks, ks), kernel_initializer = 'normal'), input_shape = img_filter_size))
                                                                     
                                                        model.add(Activation('relu'))
                                                        
                                                        #First Pooling layer resampling shapes
                                                        if input_type == 'LLDs' or input_type == "Raw":
                                                            if 'convolutional' == network:
                                                                model.add(MaxPooling2D(pool_size = (ps, 1)))
                                                            else:
                                                                model.add(TimeDistributed(MaxPooling2D(pool_size = (ps, 1))))
                                                        if input_type == 'Spect' or input_type == 'Img':
                                                            if 'convolutional' == network:
                                                                model.add(MaxPooling2D(pool_size = (ps, ps)))
                                                            else:
                                                                model.add(TimeDistributed(MaxPooling2D(pool_size = (ps, ps))))
                                                       
                                                        for l in range(int(conv)-1):
                                                            #Convlutional layer input shapes
                                                            if input_type == 'LLDs' or input_type == "Raw":
                                                                if 'convolutional' == network:
                                                                    model.add(Conv2D(fd, (ks, 1), kernel_initializer = 'normal'))
                                                                else:
                                                                    model.add(TimeDistributed(Conv2D(fd*2, (ks, 1), kernel_initializer = 'normal')))
                                                            if input_type == 'Spect'or input_type == 'Img':
                                                                if 'convolutional' == network:
                                                                    model.add(Conv2D(fd, (ks, ks), kernel_initializer = 'normal'))
                                                                else:
                                                                    model.add(TimeDistributed(Conv2D(fd*2, (ks, ks), kernel_initializer = 'normal')))
                                                                     
                                                            model.add(Activation('relu'))
                                                            #Pooling layer resampling shapes
                                                            if input_type == 'LLDs' or input_type == "Raw":
                                                                if 'convolutional' == network:
                                                                    model.add(MaxPooling2D(pool_size = (ps, 1)))
                                                                else:
                                                                    model.add(TimeDistributed(MaxPooling2D(pool_size = (ps, 1))))
                                                            if input_type == 'Spect' or input_type == 'Img':
                                                                if 'convolutional' == network:
                                                                    model.add(MaxPooling2D(pool_size = (ps, ps)))
                                                                else:
                                                                    model.add(TimeDistributed(MaxPooling2D(pool_size = (ps, ps))))
                                                            #drop out for convolutional layer
                                                            model.add(Dropout(do))
                                                        
                                                        if 'convolutional' == network:
                                                            model.add(Flatten())
                                                        else:
                                                            model.add(TimeDistributed(Flatten()))   
                                                        
                                                        if 'conv_lstm' == network:
                                                            model.add(CuDNNLSTM(neurons))
                                                            model.add(Dropout(do))
                                                        
                                                        if 'conv_gru' == network:
                                                            model.add(CuDNNGRU(neurons))
                                                            model.add(Dropout(do))
                                                            
                                                        #Generating dense layers
                                                        for l in range(layer):
                                                            model.add(Dense(neurons))
                                                            model.add(Activation('relu'))
                                                            model.add(Dropout(do))
       
                                                        model.add(Dense(n_classes))
                                                        model.add(Activation('softmax'))
                                                        opt = keras.optimizers.Adam(lr=lr, decay=1e-6)
                                                        #model training
                                                        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
                                                        
                                                        #tensorboard graph (visualize by typing in console: tensorboard log_dir=logs)
                                                        tb = keras.callbacks.TensorBoard(log_dir='./logs_'+network+'_'+name+'_'+input_type+'/{}'.format(log_name))
                                                        callbacks_list = [tb] if early_stopping == False else [tb,estop]
                                                        val_data = (x_validation,y_validation) if use_validation_set == True else (x_test,y_test)
                                                            
                                                        if show_summary == True:
                                                            model.summary()
                                                            
                                                        model.fit(x_train,
                                                                  y_train,
                                                                  batch_size=bs,
                                                                  epochs=epochs,
                                                                  validation_data=val_data,
                                                                  callbacks=callbacks_list)
                                                        
                                                        model.save('./models/'+network+'_'+name+'_'+input_type+log_name+'.h5')
                                                        
                                                    values = neurons,layer,conv,lr,bs,do,fd,ps,ks,log_name           
                                                    
                                                    print('Testing model...')
                                                    score, acc = model.evaluate(x_test, y_test, verbose=1)
                                                    print('\nLoss:', score, '\nAcc:', acc)
                                                    test_score = (acc*100).astype(str)
                                                    output_scores.append(test_score)
                                                    output_values.append(values)
                                                    current_model = str(current_run)+'-'+str(remaining_combinations)+'-'+test_score+'%'+' \t'+network+'_'+name+'_'+input_type+'\t'+'N'+str(neurons)+' Dense'+str(layer)+' Conv'+str(conv)+' Fdpth'+str(fd)+' Psz'+str(ps)+' Ksz'+str(ks)+' Lr'+str(lr)+' Bsz'+str(bs)+' Dout'+str(do)
                                                    model_list.append(current_model)
                                                    final_list.append(current_model)
                                                    print(*model_list, sep='\n')
                                                    remaining_combinations = remaining_combinations -1
                                                    
                                                    del model
                                                    gc.collect()
                                                    k.clear_session()
                                                    tf.keras.backend.clear_session()
                                                    tf.reset_default_graph()
                                                   #print(cuda.current_context().get_memory_info())
                                                    
#   #   #   #   #   #   #   #
                            if model_params == False:
                                index, score = max(enumerate(output_scores), key=operator.itemgetter(1))
                                print('keeping the best score =',score,' params =',output_values[index])
                                model_params = True  

#   #   #   #   #
                #Select best score and load model
                index, score = max(enumerate(output_scores), key=operator.itemgetter(1))
                print('keeping the best score =',score,' params =',output_values[index])
                if 'conv' in network:
                    o0,o1,o2,o3,o4,o5,o6,o7,o8,log_name = output_values[index]
                else:
                    o0,o1,o2,o3,o4,log_name = output_values[index]
                    
                best_model = network+'_'+name+'_'+input_type+log_name                    
                np.savetxt('./LogLastRun___'+best_model+'.txt',model_list,delimiter=",", fmt="%s")
                
                #confusion
                get_confusion_matrix(best_model,x_test,y_test,n_classes,labels,save_pictures,display_pictures)
                
                if get_learned_features == True:
                    visualize_layer_features(best_model,x_train,img_filter_size,img_size,save_pictures,display_pictures)
                    
                if get_saliency_map == True:
                    saliency_map(best_model,x_train,y_train,n_classes,img_size,save_pictures,display_pictures)
#   #   #
        final_list.append('Best Score = '+score+' '+best_model) 
        final_list.append(' ')      
        current_run = current_run - 1
        #if use_validation_set == True:
            #del x_validation, y_validation
        #del x_train, x_test, y_train, y_test
        
np.savetxt('./LogLastRunTotal.txt',final_list,delimiter=",", fmt="%s")