"""
MiniBatchK-Means  examples study 
解决 demoForComparsionOfTheKMeansAndMiniBatchKMeans中簇数不稳定问题
"""

from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
import numpy as np


X= np.array([
    [1,2],[1,4],[1,0],
    [4,2],[4,0],[4,4],
    [4,5],[0,5],[2,2],
    [3,2],[5,5],[1,-1]        
    ])

#manually fit on batches
kmeans = MiniBatchKMeans(n_clusters = 2, 
                         random_state=0,
                         batch_size=6)

kmeans = kmeans.partial_fit(X[0:6,:])
print ("kmeans : ", kmeans)
kmeans = kmeans.partial_fit(X[6:12,:])
print ("kmeans : ", kmeans)

mbk_cluster_centers = kmeans.cluster_centers_
print ("mbk_cluster_centers : ", mbk_cluster_centers)
