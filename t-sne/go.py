import pylab as Plot
import numpy as np
import bhtsne
import scipy.io as sio
mat_ = sio.loadmat('/home/lwu/t-SNE/features/softmax.mat')
X = mat_['dataset_feats']

data = X[0:200,:]
data = np.array(data)
labels = mat_['dataset_labels']

labels = labels[0,0:200]
labels = np.array(labels)
l = np.unique(labels)


#Y = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])
#sio.savemat('Y.mat',{'Y':Y})
Y = sio.loadmat('Y.mat')
Y = Y['Y']
from matplotlib import pyplot as plt
a=plt.scatter(Y[:,0], Y[:,1], 20, labels)

plt.legend(a,l,loc = 0,)

plt.show()
