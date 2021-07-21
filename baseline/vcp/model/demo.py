import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
adj=sp.csr_matrix((data, (row, col)), shape=(3, 3))
print(type(adj))

x=np.zeros(None,None,dtype=float)
print(x)

# def get_degree_supports(adj, k, adj_self_con=False, verbose=True):  # k = 2
#  if verbose:
#   print('Computing adj matrices up to {}th degree'.format(k))
#  supports = [sp.identity(adj.shape[0])]  # 定义稀疏格式的单位矩阵
#  '''
#  identity(3).toarray()
#  array([[ 1.,  0.,  0.],
#         [ 0.,  1.,  0.],
#         [ 0.,  0.,  1.]])
#  '''
#  if k == 0:  # return Identity matrix (no message passing)
#   return supports
#  assert k > 0
#  supports = [sp.identity(adj.shape[0]), adj.astype(np.float64) + adj_self_con * sp.identity(adj.shape[0])]
#
#  prev_power = adj
#  for i in range(k - 1):
#   pow = prev_power.dot(adj)
#   new_adj = ((pow) == 1).astype(np.float64)
#   new_adj.setdiag(0)
#   new_adj.eliminate_zeros()
#   supports.append(new_adj)
#   prev_power = pow
#  return supports
#
# supports = get_degree_supports(adj,2,adj_self_con=False, verbose=True)
# # sess= tf.Session()
# print(supports)


# outputs = torch.zeros(2,2)
# print(outputs.shape)
# outputs = torch.squeeze(outputs)
# print(outputs)
# test = np.array([
# [1, 2, 3],
#  [2, 3, 4],
#  [5, 4, 3],
#  [8, 7, 2]])
# m = test[-1]
# print(m)

# isolate = np.random.choice(range(2), int(2), replace=False)
# print(isolate)
# z = tf.cast(test >= 0.0, tf.int64)
# print(z)
# sess = tf.Session()
# # print(sess.run(z))
# print(sess.run(m))

# act=lambda x: x
# print(act([3,4,5]))
# x_n = np.random.random([5,3])
#
# print(x_n)
# x_n = torch.nn.dropout(x_n, 0.2)
# # with session as  sess:
# #     print(sess(x_n))
# print(x_n)
# #
# _LAYER_UIDS = {'': 1}
# def get_layer_uid(layer_name=''):
#     """Helper function, assigns unique layer ID
#     """
#     if layer_name not in _LAYER_UIDS:
#         _LAYER_UIDS[layer_name] = 1
#         print(_LAYER_UIDS)
#         return 1
#     else:
#         _LAYER_UIDS[layer_name] += 1
#         print(_LAYER_UIDS)
#         return _LAYER_UIDS[layer_name]
#     print(_LAYER_UIDS)
# print(get_layer_uid())



# mp_pos_labels = [1,1,1]
# mp_pos_r_idx = [4,3,5]
# mp_pos_c_idx = [2,2,2]
# adj = sp.csr_matrix((
#                     np.hstack([mp_pos_labels, mp_pos_labels]),
#                     (np.hstack([mp_pos_r_idx, mp_pos_c_idx]), np.hstack([mp_pos_c_idx, mp_pos_r_idx]))
#                     # np.hstack([mp_pos_r_idx, mp_pos_c_idx])
#                 ),
#                 shape=(6,6)
#             )
# print(adj)
# print(adj.eliminate_zeros())
# eval_pos_labels = [1,1,1,1,1]
# neg_labels = np.zeros(10)
# labels = np.append(eval_pos_labels, neg_labels)
# print(labels)

# st = tf.SparseTensor(values=[1, 2, 3], indices=[[0, 0], [1, 1],[2,2]], dense_shape=[3, 3])
# print(st)
# dt = tf.ones(shape=[3,3],dtype=tf.int32)
# result = tf.sparse_tensor_dense_matmul(st,dt)
# # result = tf.matmul(st,dt)
# print(result)
# sess = tf.Session()
# with sess.as_default():
#     print(result.eval())
#     print(st.eval())
# x=np.arange(10)
# print(x)
# perm = list(range(10))
# np.random.shuffle(perm)
# print(perm)
# x= np.zeros(10)
# print(x)

# _LAYER_UIDS = {}
# def get_layer_uid(layer_name=''):
#     """Helper function, assigns unique layer IDs
#     """
#     if layer_name not in _LAYER_UIDS:
#         _LAYER_UIDS[layer_name] = 1
#         return 1
#     else:
#         _LAYER_UIDS[layer_name] += 1
#         return _LAYER_UIDS[layer_name]
#
# x=get_layer_uid()
# print(x)
# labels = [1,2,3,0]
# int_preds = [1,3,3,2]
# z = tf.confusion_matrix(labels, int_preds)
# print(z)
#
# with tf.Session() as sess:
#     print(sess.run(z))
# accuracy_all = torch.tensor([0,1,1,0,1,0],dtype=float)
#
# x = torch.mean(accuracy_all)
# print(x)
# with tf.Session() as sess:
#     print(sess.run(x))