# -*- coding: utf-8 -*-
"""An unsupervised learning model based on PCA and Agglomerative"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import decomposition
from sklearn.utils.linear_assignment_ import linear_assignment

def cluster_acc(y_true, y_pred):
    """match labels from clustering and calculate accuracy"""
    assert y_pred.size == y_true.size
    dummyd = max(y_pred.max(), y_true.max())+1
    dummyw = np.zeros((dummyd, dummyd), dtype=np.int64)
    for i in range(y_pred.size):
        dummyw[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(dummyw.max() - dummyw)
    return sum([dummyw[i, j] for i, j in ind])*1.0/y_pred.size


class Agglomerative:
    """Unsupervised model based on PCA and Agglomerative Clustering

    Parameters
    ----------
    n_components: int
        The dimensions of embedded space
    n_cluster: int
        Estimate of class number

    Returns
    ---------
    TODO

    """
    def __init__(self, n_components=50, n_cluster=17):
        self.n_components = n_components
        self.n_cluster = n_cluster


    def pca_embeding(self, data_train):
        """use pca to embed features into embedded space"""
        pca = decomposition.PCA(self.n_components)
        pca.fit(data_train)
        return pca.transform(data_train)


    def cluster(self, data_train, y_train=None, semi_supervised=False):
        """cluster data points by Agglomerative method"""
        embed_train = None
        if not semi_supervised:
            embed_train = self.pca_embeding(data_train)
        else:
            embed_train = data_train
        if y_train is None:
            model_ac = AgglomerativeClustering(self.n_cluster)
            labels_ac = model_ac.fit_predict(embed_train)
        else:
            self.n_cluster = y_train.max()
            model_ac = AgglomerativeClustering(self.n_cluster)
            labels_ac = model_ac.fit_predict(embed_train)
            return labels_ac, cluster_acc(labels_ac, y_train)
        return labels_ac
