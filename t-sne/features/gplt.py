from sklearn import datasets
import matplotlib.pyplot as plt
import pylab as Plot
import numpy as np
import scipy.io as sio
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
mat_ = sio.loadmat('softmax.mat')
X = mat_['dataset_feats']
data = X[0:200,:]
data = np.array(data)
labels = mat_['dataset_labels']
labels = labels[0,0:200]
labels = np.array(labels)
l = np.unique(labels)
Y = sio.loadmat('Y.mat')
Y = Y['Y']
Y = np.array(Y)
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
idx = []
for i in range(len(colors)):
	idx = np.where(labels==i)
	plt.scatter(Y[idx,0], Y[idx,1],color = colors[i], label= l[i] , s = 20)
plt.legend(loc='upper right')
plt.show()