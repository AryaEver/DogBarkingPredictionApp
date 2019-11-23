###############################################################################
##################### VISUALIZE LAYER FEATURES ################################
###############################################################################

# Visualizating filters
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def visualize_layer_features(best_model,x_train,img_filter_size,img_size,save_pictures,display_pictures):
    
    img = np.array(x_train[0]).reshape((img_filter_size)).astype(np.float64)
    K.set_learning_phase(1)
    global model 
    model = load_model('./models/'+best_model+'.h5')
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    #print('Layer dict', layer_dict)
    #print(model.summary())
    #put algoritm for more layers 
    try:
        vis_img_in_filter(0,img,img_size,layer_dict,best_model,x_train,save_pictures,display_pictures)
    except Exception as e:
        print(e)

#util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vis_img_in_filter(layer_num,img,img_size,layer_dict,best_model,x_train,save_pictures,display_pictures):
    layer_name = model.layers[layer_num].name
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    
    #convolutional shows all convolutional layers if avalible
    if 'convolutional' in best_model:
        for filter_index in range(layer_output.shape[3]):
            # build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            loss = K.mean(layer_output[:, :, :, filter_index])
    
            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, model.input)[0]
    
            # normalization trick: we normalize the gradient
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
            # this function returns the loss and grads given the input picture
            iterate = K.function([model.input], [loss, grads])
    
            # step size for gradient ascent
            step = 5.
    
            img_asc = np.array(img)
            # run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([img_asc])
                img_asc += grads_value * step
    
            img_asc = img_asc[0]
            img_ascs.append(deprocess_image(img_asc).reshape((img_size)))
            
        if layer_output.shape[3] >= 35:
            plot_x, plot_y = 6, 6
        elif layer_output.shape[3] >= 23:
            plot_x, plot_y = 4, 6
        elif layer_output.shape[3] >= 11:
            plot_x, plot_y = 2, 6
        else:
            plot_x, plot_y = 1, 2
        if 'spect' in best_model:
            fig, ax = plt.subplots(plot_x, plot_y, figsize = (18, 9))
            ax[0, 0].imshow(img.reshape((img_size)), cmap = 'gray', aspect ='auto')
            ax[0, 0].set_title('Input image')
            fig.suptitle('Input image and %s filters' % (layer_name,))
            fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
            for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
                if x == 0 and y == 0:
                    continue
                ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray',aspect='auto')
                ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
        
        if 'LLDs' or 'Raw' in best_model:
            fig, ax = plt.subplots(plot_x, plot_y, figsize = (18, 9))
            ax[0, 0].plot(img.reshape(img_size))
            ax[0, 0].set_title('Input image')
            fig.suptitle('Input image and %s filters' % (layer_name,))
            fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
            for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
                if x == 0 and y == 0:
                    continue
                ax[x, y].plot(img_ascs[x * plot_y + y - 1])
                ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
        try:
            if save_pictures == True:
                plt.savefig('layer_features_'+best_model+'.png', bbox_inches='tight')
            if display_pictures == True:
                plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        except Exception as e:
            print(e)
    #lstm only shows first hidden layer    
    if 'lstm' in best_model:
            num_inputs=[0,1,2,3,4]
            fig, ax = plt.subplots(len(num_inputs), 2, figsize = (18, 9))
            fig.suptitle('Input image and %s activation' % (layer_name))
            for j in range(len(num_inputs)):
                img = np.array(x_train[j]).reshape((img_size)).astype(np.float64)
                img_ascs = list()
                # build a loss function that maximizes the activation
                # of the nth filter of the layer considered
                loss = K.mean(layer_output[:, :,:])
            
                # compute the gradient of the input picture wrt this loss
                grads = K.gradients(loss, model.input)[0]
            
                # normalization trick: we normalize the gradient
                grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            
                # this function returns the loss and grads given the input picture
                iterate = K.function([model.input], [loss, grads])
            
                # step size for gradient ascent
                step = 5.
            
                img_asc = np.array(img)
                #print("img_asc.shape",img_asc.shape)
                # run gradient ascent for 20 steps
                for i in range(20):
                    loss_value, grads_value = iterate([img_asc])
                    img_asc += grads_value * step
            
                img_asc = img_asc[:,:,0]
                img_ascs.append(deprocess_image(img_asc).reshape((img_size)))    
                if 'LLDs' or 'Raw' in best_model:                    
                    ax[j, 0].plot(img.reshape((img_size)))
                    ax[j, 0].set_title('Input image %d' %j,fontsize=6)
                    
                    ax[j, 1].plot(img_ascs[0])
                    ax[j, 1].set_title('Layer %s optimal activation ' %layer_name,fontsize=6)
                
                if 'Spect' in best_model:                    
                    ax[j, 0].plot(img.reshape((img_size)))
                    ax[j, 0].set_title('Input image %d' %j,fontsize=6)
                    
                    ax[j, 1].plot(img_ascs[0])
                    ax[j, 1].set_title('Layer %s optimal activation ' %layer_name,fontsize=6)
            try:
                if save_pictures == True:
                    plt.savefig('layer_features_'+best_model+'.png', bbox_inches='tight')
                if display_pictures == True:
                    plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            except Exception as e:
                print(e)