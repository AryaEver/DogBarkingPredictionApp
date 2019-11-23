#https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from ggplot import *

mnist = tf.keras.datasets.mnist
(xdata,ydata),(xtest,ytest)= mnist.load_data()
#concatenate 60k+10k = 70k
xdata= np.concatenate([xdata,xtest])
ydata= np.concatenate([ydata,ytest])
del xtest,ytest
#reshape data and normalize by dividing
xdata = np.reshape(xdata,(len(xdata),784))/255

print(xdata.shape, ydata.shape)

feat_cols = [ 'pixel'+str(i) for i in range(xdata.shape[1]) ]

df = pd.DataFrame(xdata,columns=feat_cols)
df['label'] = ydata
df['label'] = df['label'].apply(lambda i: str(i))

xdata, ydata = None, None
print('Size of the dataframe: {}'.format(df.shape))

rndperm = np.random.permutation(df.shape[0])
'''
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#plot PCA



chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=30,alpha=0.6) \
        + ggtitle("First and Second Principal Components colored by digit")
#write chart in console to visulize chart
        '''
#N_SNE
import time

from sklearn.manifold import TSNE

n_sne = 700*1

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]
df_tsne['z-tsne'] = tsne_results[:,2]

chart2 = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne',z='z-tsne', color='label') ) \
        + geom_point(size=30,alpha=0.6) \
        + ggtitle("tSNE dimensions colored by digit")
#write chart2 in console to visulize chart2


colors= np.asarray(list(map(int, df_tsne['label'].values)))
group = np.unique(colors.astype(str))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], alpha=0.8, c=colors,cmap='prism',label=group,marker='.')
#for x,y,z,i in zip(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2],range(len(tsne_results[:,0]))):
#    ax.text(x,y,z,colors[i],fontsize=5)
plt.title('Tsne 3d scatter plot')
plt.legend(loc=2)
plt.show()        

