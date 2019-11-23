import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import seaborn as sns
#import numpy as np
import operator


scores = [
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv (LLD's)", 84.56),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv (LLD's)", 95.46),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv (LLD's)", 82.89),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv (LLD's)", 87.66),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv (LLD's)", 85.45),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv (MelSpect)", 65.35),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv (MelSpect)", 88.35),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv (MelSpect)", 76.48),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv (MelSpect)", 86.5),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv (MelSpect)", 77.7),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv (Raw Audio)", 21.93),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv (Raw Audio)", 23.93),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv (Raw Audio)", 58.55),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv (Raw Audio)", 37.82),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv (Raw Audio)", 37.45),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "LSTM (LLD's)", 55.5),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "LSTM (LLD's)", 69),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "LSTM (LLD's)", 75.3),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "LSTM (LLD's)", 77.8),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "LSTM (LLD's)", 67.1),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "LSTM (MelSpect)", 57.1),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "LSTM (MelSpect)", 81.4),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "LSTM (MelSpect)", 75.6),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "LSTM (MelSpect)", 71.7),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "LSTM (MelSpect)", 71.7),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "LSTM (Raw Audio)", 10.13),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "LSTM (Raw Audio)", 11.15),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "LSTM (Raw Audio)", 33.44),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "LSTM (Raw Audio)", 16.76),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "LSTM (Raw audio)", 11.91),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv-Lstm (LLD's)", 77.0),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv-Lstm (LLD's)", 93.79),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv-Lstm (LLD's)", 80.26),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv-Lstm (LLD's)", 86.51),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv-Lstm (LLD's)", 81.55),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv-Lstm (Raw audio)", 19.14),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv-Lstm (Raw audio)", 20.08),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv-Lstm (Raw audio)", 51.52),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv-Lstm (Raw audio)", 32.56),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv-Lstm (Raw audio)", 37.11),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Gru (LLD's)", 33.88),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Gru (LLD's)", 45.83),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Gru (LLD's)", 58.52),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Gru (LLD's)", 58.88),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Gru (LLD's)", 49.59),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv-Gru (LLD's)", 81.54),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv-Gru (LLD's)", 94.70),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv-Gru (LLD's)", 76.48),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv-Gru (LLD's)", 84.53),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv-Gru (LLD's)", 80.98),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Gru (Raw audio)", 0),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Gru (Raw audio)", 0),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Gru (Raw audio)", 0),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Gru (Raw audio)", 0),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Gru (Raw audio)",0 ),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "Conv-Gru (Raw audio)", 0),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Conv-Gru (Raw audio)", 0),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "Conv-Gru (Raw audio)", 0),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "Conv-Gru (Raw audio)", 0),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv-Gru (Raw audio)", 0),
        
        ("Database: Mudi Dogs \n Task: 7 Context categories", "SVM (LLD's)\nAutomatic classification\n of context in induced \nbarking (2015)", 75,1),
        ("Database: Mescalina Dogs \n Task: 4 Context categories", "RandomForest (LLDs)\n Tuning the Parameters of\na ConvolutionalArtificial\nNeural Network by Using\nCoveringArrays (2016)", 79,1),
        ("Database: Mudi Dogs \n Task: 11 Individual categories", "Bayesian (Spectrogram)\nClassification of dog barks:\na machine learning approach\n(2008)", 52,1),
        ("Database: Mescalina Dogs \n Task: 36 Individual categories", "SVM (LLD's)\nAutomatic individual dog\n recognition based on the \nacoustic properties of \nits barks (2018)", 90.5,1),
        ("Database: UrbanSound8k \n Task: 10 Context categories", "Conv Augmented (MelSpect)\n Deep Convolutional\nNeural Network and Data\nAugmentation for Environmental\nSound Classification (2017)", 79,1)
        ]
       
def results(database):
    plt.figure(figsize=(18,9))
    db_plot = []
    for i in scores:
        if i[0] == database:
            db_plot.append(i)
    db_plot.sort(key=operator.itemgetter(2))
    bar = plt.barh([x[1] for x in db_plot],[x[2] for x in db_plot],color=(0.2, 0.4, 0.6))
    for rect in bar:
        height = rect.get_y() + rect.get_height()/2.0
        width = rect.get_width()
        plt.text(width+1.5, height, '%.1f%%' % width, ha='center', va='bottom')
    for x in range(len(db_plot)):
        try:
            if db_plot[x][3] == 1:
                bar.patches[x].set_facecolor('y')
        except Exception as e: print(e)
            
    plt.ylabel('Method (Input Data)')
    plt.xlabel('Test Accuracy')
    plt.title(db_plot[0][0])
    plt.show()
    name = database.split()
    plt.savefig('Scores'+name[1]+name[5]+'.png', bbox_inches='tight')
    return db_plot
    
results("Database: Mudi Dogs \n Task: 7 Context categories") 
results("Database: Mudi Dogs \n Task: 11 Individual categories")
results("Database: Mescalina Dogs \n Task: 4 Context categories")
results("Database: Mescalina Dogs \n Task: 36 Individual categories")
results("Database: UrbanSound8k \n Task: 10 Context categories")
