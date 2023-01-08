import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from  matplotlib import pylab;
import copy;
import tempfile;    #创建临时文件和目录。
import os;

from sklearn.decomposition import PCA;
from sklearn.datasets import make_blobs;
"""

X,Y = make_blobs(n_samples=100 ,random_state=1, centers=[[-5,0],[5,5],[5,-5]]);
print (X.shape);
print (Y);

pca = PCA(n_components=2);
x_pca = pca.fit_transform(X);
print (x_pca.shape);
print("Y");
print (Y);
print (Y.shape);

pca_comp = pca.components_.T;
print (type(pca_comp));
print(pca_comp);

test_point = np.matrix([5,-2]);
test_point_pca = pca.transform(test_point);
print(test_point);

plt.subplot(1,2,1);
plt.scatter(X[:,0],X[:,1],c=Y, edgecolors='none');
plt.quiver(0, pca_comp[:,0],pca_comp[:,1], width=0.02, scale=5, color='orange');
plt.plot(test_point[0,0], test_point[0,1],'o');
plt.plot('Input dataset');

plt.subplot(1,2,2);
plt.scatter(X_pca[:,0],X_pca[:,1], c=Y, edgecolors='none');
plt.plot(test_point_pca[0,0], teste_point_pca[0,1],'o');
plt.title('After " lossless" PCA');

plt.show();
"""


### 使用 matplotlib.pyplot 画 散点图
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
data = np.random.seed(19680801)
print ("Data : " , data)
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
#
print ("X ; " , x)
print ("y : " , y)
print ("color: " , colors)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
print ("Area value : ", area)

"""# s 控制点的大小
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, 
    norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, 
    edgecolors=None, plotnonfinite=False, data=None, **kwargs)

    x, y 浮点或类数组。形式为(n,)
    alpha 的值介于0-1 ，调整透明关系 透明--不透明     
"""
#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.scatter(x,y,c=colors);
plt.show()