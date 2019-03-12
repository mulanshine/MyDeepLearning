import numpy as np
from numpy import linalg
from numpy.linalg import norm
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
randomState=13204
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
#sns.set_style('darkgrid')
#sns.set_palette('muted')
#sns.set_context("notebook",font_scale=1.5,rc={"lines.linewidth":2.5})


digits=load_digits()
print 'shape is ',digits.data.shape
pca = PCA(n_components=2)
xn=pca.fit_transform(digits.data)


nrows,ncols=2,5
plt.figure(figsize=(6,3))
plt.gray()
for i in range(ncols*nrows):
    ax=plt.subplot(nrows,ncols,i+1)
    ax.matshow(digits.images[i,...])
    plt.xticks([]);plt.yticks([])
    plt.title(digits.target[i])
    
plt.show()


X=np.vstack([digits.data[digits.target==i] for i in range(10)])

y=np.hstack([digits.target[digits.target==i] for i in range(10)])
projectedDigits = TSNE(random_state=randomState).fit_transform(X)

def CustomScatter(x,colors,i,title1,xl,yl):
    palette = np.array(sns.color_palette("hls",10))
    
    ax = plt.subplot(1, 2, i)
    plt.title(title1)
    sc=ax.scatter(x[:,0],x[:,1],lw=0,s=40,c=palette[colors.astype(np.int)])
    plt.xlim(xl,yl)
    plt.ylim(xl,yl)
    labels=[]
    for i in range(10):
        xtext,ytext=np.median(x[colors==i,:],axis=0)
        txt=ax.text(xtext,ytext,str(i),fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5,foreground="w"),PathEffects.Normal()])
        labels.append(txt)


CustomScatter(projectedDigits, y,1,'Dimension Reduction by TSNE',-25,25)

CustomScatter(xn,y,2,'Dimension Reduction by PCA',-40,30)
plt.show()


