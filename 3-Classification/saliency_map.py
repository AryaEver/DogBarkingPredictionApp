###############################################################################
############################## SALIENCY MAP ###################################
###############################################################################

# Saliency map
# https://github.com/experiencor/deep-viz-keras/blob/master/saliency.py
from keras.layers import Input, Conv2DTranspose
from keras.models import Model, load_model
from keras.initializers import Ones, Zeros
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras import backend as K

def saliency_map(best_model,x_train,y_train,n_classes,img_size,save_pictures,display_pictures):
    try:
        #lb = preprocessing.LabelBinarizer()
         #y_train_label = lb.inverse_transform(y_train).astype(np.int32)
        y_train_label = np.argmax(y_train, axis=1)
        
        global model 
        model = load_model('./models/'+best_model+'.h5')
        fig, ax = plt.subplots(n_classes, 5, figsize = (18, 9))
        fig.suptitle('vanilla gradient')
        for i in range(n_classes):
            img = np.array(x_train[i])
    
            vanilla = GradientSaliency(model, y_train_label[i])
            mask = vanilla.get_mask(img)
            filter_mask = (mask > 0.0).reshape((img_size))
            smooth_mask = vanilla.get_smoothed_mask(img)
            filter_smoothed_mask = (smooth_mask > 0.0).reshape((img_size))
        
            if 'Spect' in best_model:
                ax[i, 0].imshow(img.reshape((img_size)), cmap = 'gray',aspect='auto')
                cax = ax[i, 1].imshow(mask.reshape((img_size)), cmap = 'jet',aspect='auto')
                fig.colorbar(cax, ax = ax[i, 1])
                ax[i, 2].imshow(mask.reshape((img_size)) * filter_mask, cmap = 'gray',aspect='auto')
                cax = ax[i, 3].imshow(mask.reshape((img_size)), cmap = 'jet',aspect='auto')
                fig.colorbar(cax, ax = ax[i, 3])
                ax[i, 4].imshow(smooth_mask.reshape((img_size)) * filter_smoothed_mask, cmap = 'gray',aspect='auto')
                
            if 'LLDs' or 'Raw' in best_model:
                ax[i, 0].plot(img.reshape(img_size))
                ax[i, 0].set_title('input image class:%d' %i,fontsize=6)
                
                cax = ax[i, 1].plot(mask.reshape((img_size)))
                #fig.colorbar(cax, ax = ax[i, 1])
                ax[i, 1].set_title('mask class:%d' %i,fontsize=6)
                
                ax[i, 2].plot(mask.reshape((img_size)) * filter_mask)
                ax[i, 2].set_title('filter mask class:%d' %i,fontsize=6)
                
                cax = ax[i, 3].plot(mask.reshape((img_size)))
                ax[i, 3].set_title('Smooth mask clas:%d' %i,fontsize=6)
                #fig.colorbar(cax, ax = ax[i, 3])
                
                ax[i, 4].plot(smooth_mask.reshape((img_size)) * filter_smoothed_mask)
                ax[i, 4].set_title('Filter smooth  mask class:%d' %i, fontsize=6)
                
        if save_pictures == True:
            plt.savefig('saliency_map_'+best_model+'.png', bbox_inches='tight')
        if display_pictures == True:
            plt.show()
        plt.clf()
        plt.cla()
        plt.close()
    except Exception as e:
        print('error in saliency map')
class SaliencyMask(object):
    def __init__(self, model, output_index=0):
        pass

    def get_mask(self, input_image):
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class GradientSaliency(SaliencyMask):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients

# https://github.com/experiencor/deep-viz-keras/blob/master/visual_backprop.py
class VisualBackprop(SaliencyMask):
    def __init__(self, model, output_index = 0):
        inps = [model.input]           # input placeholder
        outs = [layer.output for layer in model.layers]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function
        
        self.model = model

    def get_mask(self, input_image):
        x_value = np.expand_dims(input_image, axis=0)
        
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        for i in range(len(self.model.layers) - 1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis = 3, keepdims = True)
                layer = layer - np.min(layer)
                layer = layer / (np.max(layer) - np.min(layer) + 1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        x = Input(shape = (None, None, 1))
        y = Conv2DTranspose(filters = 1, 
                            kernel_size = (3, 3), 
                            strides = (2, 2), 
                            padding = 'same', 
                            kernel_initializer = Ones(), 
                            bias_initializer = Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]
