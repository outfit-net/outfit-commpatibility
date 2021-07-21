import numpy as np
import torch
import scipy.sparse as sp

from scipy.sparse import coo_matrix

# _row  = np.array([0, 3, 1, 2])
# _col  = np.array([1, 2, 0, 3])
# _data = np.array([4, 5, 4, 5])
# coo = coo_matrix((_data, (_row, _col)), shape=(4, 4), dtype=np.int)
# print(coo)
# z=sp.tril(coo).tocsr()
# print(z)
# r_i,r_j = np.nonzero(z)

pos_labels =[[1,2],[2,3],[3,4]]
pos_labels1 = [[1,3],[3,3],[3,2]]
x = np.append(pos_labels,pos_labels1)
print(x)

# pos_labels = np.array(pos_labels)
# n_pos = pos_labels.shape[0]  # number of positive edges
# perm = list(range(n_pos))
# np.random.shuffle(perm)
# pos_labels = pos_labels[perm]
# print(pos_labels)
# coo.todense()  # 通过toarray方法转化成密集矩阵(numpy.matrix)
# print(coo)
# coo.toarray()  # 通过toarray方法转化成密集矩阵(numpy.ndarray)
# print(coo)

# mean=None
#
# std=None
# reuse_mean = mean is not None and std is not None
# print(reuse_mean)
#
# fea = [[1,2],[2,3],[3,4]]
# fea =np.mat(fea)
# fea = fea.mean(axis=0)
# fea1 = np.std(fea,axis=0)
# print(fea1)
# print(fea)
#
# fe = [1,2,3]
# print(np.std(fe))