#https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
import numpy as np
import numpy
import pandas as pd
from pandas import Timestamp
import ggplot
from ggplot import geom_point, ggtitle, aes
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
#from ggplot import *
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
chunks = []
import matplotlib.patches as mpatches
chunksize = 1000
size = 6615

#labels = list(np.unique(pd.read_csv('./mudidogs.csv').values[:,1]))

"""
slicesize = int(size/chunksize)
print('loading data by chunk size = ',chunksize, ' of ',size, ' total')
start_time = time.time()

for idx, chunk in enumerate(pd.read_csv('./mescalina_individ_raw.csv',chunksize=chunksize,header=None)):    
    chunks.append(chunk)
    out_file = "chunk/data_{}".format(idx)
    with open(out_file, "wb") as f:
        pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)    
    if idx == 0:
        totaltime = (slicesize + 1)*(time.time()-start_time)
    print('data loaded = ',idx*chunksize,' time remaining =',totaltime-(time.time()-start_time))
"""
train_data = pd.read_csv('./mudi_context_LLDs.csv',header=None).as_matrix();

print(train_data.shape)
rows, columns = train_data.shape
train,test = int(rows*0.9),int(rows*0.1)
test_data = train_data[train:,:]
train_data = train_data[:train,:]

rows, columns = train_data.shape
train_labels = train_data[:,columns-1].astype(int)
labelsize = pd.factorize(train_labels)
print(labelsize[1])
labelsize = len(labelsize[1])  
train_data = train_data[:,0:columns-1]
rows, columns = train_data.shape
train_data = numpy.reshape(train_data,(rows,columns))

rows, columns = test_data.shape
test_labels = test_data[:,columns-1].astype(int)
test_data = test_data[:,0:columns-1]
rows, columns = test_data.shape
test_data = numpy.reshape(test_data,(rows,columns))

xdata= np.concatenate([train_data,test_data])
ydata= np.concatenate([train_labels,test_labels])
del train_data,test_data

feat_cols = [ 'pixel'+str(i) for i in range(xdata.shape[1]) ]

df = pd.DataFrame(xdata,columns=feat_cols)
df['label'] = ydata
df['label'] = df['label'].apply(lambda i: str(i))

xdata, ydata = None, None
print('Size of the dataframe: {}'.format(df.shape))

rndperm = np.random.permutation(df.shape[0])

'''
#PCA 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


chart = ggplot( df.loc[rndperm[:8000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=30,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
#write chart in console to visulize chart

#PCA 3D
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

colors= np.asarray(list(map(int, df['label'].values)))
group = np.unique(colors.astype(str))
fig = plt.figure()
ax = Axes3D(fig,elev=300)
ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], alpha=0.8,c=colors,cmap='hsv',label=group,marker='.')
#box=['red','pink','blue','yellow','purple','green','orange','aqua','olive','magenta']
#for x,y,z,i in zip(pca_result[:,0], pca_result[:,1], pca_result[:,2],range(len(pca_result[:,0]))):
#    ax.text(x,y,z,colors[i],fontsize=3,bbox=dict(facecolor=box[colors[i]],pad=0.6, alpha=0.75,edgecolor='none',boxstyle='round'))
plt.title('PCA 3d Scatter plot\n From'+str(size)+'Dimension to 3 Dimensions ')
#plt.tight_layout()
plt.show()  
'''

#TSNE 2d
n_sne = int(size)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

#plot t-SNE 2d
chart2 = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=20,alpha=0.7) \
        + ggtitle("tSNE dimensions colored by digit")
#write chart in console to visulize chart2
chart2

        
'''
#TSNE 3d
n_sne = int(size)

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=400)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne = df.loc[rndperm[:n_sne],:].copy()

df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]
df_tsne['z-tsne'] = tsne_results[:,2]

colors= np.asarray(list(map(int, df_tsne['label'].values)))
group = np.unique(colors.astype(str))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], alpha=0.8, c=colors,cmap='hsv',label=group,marker='.')
#for x,y,z,i in zip(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2],range(len(tsne_results[:,0]))):
#    ax.text(x,y,z,colors[i],fontsize=5)
plt.title('Tsne 3d Scatter plot\n From'+str(size)+'Dimension to 3 Dimensions ')
plt.show()
df_tsne = np.column_stack((tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], colors))
np.savetxt('mudi_context_Tsne3d.csv', df_tsne, delimiter=',',fmt="%s")
'''


