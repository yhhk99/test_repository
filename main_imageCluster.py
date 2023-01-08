
#对图像进行聚类的一种或几种方法


import numpy as np
from sklearn import cluster, datasets


iris = datasets.load_iris();
print (iris);
#print(iris.shape);


X_iris = iris.data;
#print(X_iris);
y_iris = iris.target;
print(y_iris);

k_means = cluster.KMeans(n_clusters=3);

#计算k_means 聚类，这里是对x进行聚类
k_means.fit(X_iris);


print(k_means.labels_[::10]);
print(y_iris[::10])


"""
###
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

#from skimage import  io, filter
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering



# #############################################################################
# Generate data
orig_coins = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)



#特征聚类
#对样本数据转置后进行聚类
digits = datasets.load_digits();
images = digits.images

X2 = np.reshape(images, (len(images), -1))
print (X2)

connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity = connectivity, n_clusters=32)
print (agglo)

agglo.fit(X2)

X_reduced = agglo.transform(X2)

X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)
"""

##
"""
解决 cannot import name ‘filter‘ from ‘skimage‘ 问题
https://blog.csdn.net/cst95295299/article/details/108663915

from skimage import data, io, filters
"""
image = data.coins()

edges = filters.sobel(image)
print(io.imshow(edges))
"""

##主成分分析  使用来转换数据，可以通过子空间的投影来降低数据的维数
x11 = np.random.normal(size=10)
x12 = np.random.normal(size=10)
x13 = x11 + x12
X123 = np.c_[x11,x12,x13]
print (X123)

#decomposition模块提供了矩阵分解算法，如PCA、NMF和ICA。该模块的大部分算法均可看作降维算法
from sklearn import decomposition

pca = decomposition.PCA()
pca.fit(X123)

print (pca.explained_variance_)

pca.n_components = 2;
X_reduced = pca.fit_transform(X123)
print("X123 D: ", X123.shape)
print("X_reduced : " ,X_reduced.shape)

#独立成分分析 可以提取数据信息中的独立成分，这些城府载荷包含了最多的独立信息
import numpy as np
from scipy import signal

time = np.linspace(0,10,2000)
s1 = np.sin(2*time) #信号1,     sinusoidal signal
s2 = np.sign(np.sin(3 * time)) #信号2  square signal
s3 = signal.sawtooth(2 * np.pi * time) #信号3 saw tooth signal

print ("s1: " , s1)
print ("s2: ", s2)
print ("s3: ", s3);

#np.c_[]  可以拼接多个数组，但要求拼接数组的行数必须相同
s = np.c_[s1,s2,s3] 

print("s : " , s)   

s += 0.2 * np.random.normal(size = s.shape);
print (s.shape)

s /= s.std(axis=0)
print (s)

A = np.array([[1,1,1],[0.5,2,1],[1.5,1,2]])

X = np.dot(s,A.T)
print(A.T)

ica = decomposition.FastICA()
s_ = ica.fit_transform(X)
A_ = ica.mixing_.T
datanp = np.allclose(X, np.dot(s_,A_) + ica.mean_)
print("s_ :", s_)
print("A_ :", A_)
print("datanp: " , datanp)
"""

