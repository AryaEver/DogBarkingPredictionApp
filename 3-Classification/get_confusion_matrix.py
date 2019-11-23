###############################################################################
############################ CONFUSION MATRIX #################################
###############################################################################              
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
        
def get_confusion_matrix(best_model,x_test,y_test,n_classes,labels,save_pictures,display_pictures):
    
    try:
        #Confusion Matrix
        print('Loading model the best model...')
        model = load_model('./models/'+best_model+'.h5')
        print('Testing model...')
        score, acc = model.evaluate(x_test, y_test, verbose=1)
        print('\nLoss:', score, '\nAcc:', acc)
        testscore = (acc*100).astype(str)
        testsize = str(len(y_test))
        y_pred = model.predict(x_test)
        y_test_pred = np.argmax(y_pred, axis=1)#lb.inverse_transform(y_pred)
        y_test_real = np.argmax(y_test, axis=1)#np.argmax(y_test, axis=1)#lb.inverse_transform(y_val)
        cm = confusion_matrix(y_test_real, y_test_pred)
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        plt.figure(figsize = (26, 15))
        plt.imshow(cm)
        mod = best_model.split("_")
        plt.title('Normalized confusion matrix \n'+mod[0]+mod[1]+mod[2]+mod[3]+'\nTest size = '+ testsize +'\n Test accuaracy = '+ testscore +'%')
        plt.colorbar()
        tick_marks = np.arange(n_classes)
        plt.tight_layout()
        plt.xticks(tick_marks, np.arange(n_classes))
        plt.yticks(tick_marks, labels)
        fsize = 14
        fsize = 5 if n_classes > 15 else fsize
        fsize = 8 if n_classes > 10 else fsize
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center", fontsize = fsize,
                     color="black" if cm[i, j] > thresh else "white")
        if save_pictures == True:
            plt.savefig('confusion_matrix_'+best_model+'.png', bbox_inches='tight')
        if display_pictures == True:
            plt.show() 
        plt.clf()
        plt.cla()
        plt.close()
    except Exception as e:
        print('error in confusion matrix')
        