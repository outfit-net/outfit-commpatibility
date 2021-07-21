import tensorflow as tf
import numpy as np
# import torch
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers
from load_data_NGNN import load_train_data
import json

# ftrain = open('F:\\NGNN\\NGNN\\data\\train_no_dup_hyper.json', 'r')
# outfit_list = json.load(ftrain)
# image_pos, image_neg, text_pos, text_neg, graph_pos, graph_neg, size_=load_train_data(0, 8, outfit_list)
# print(image_pos)
# print(graph_pos)
# sess = tf.InteractiveSession()
######################model##########################
def weights(name, hidden_size, i):
    image_stdv = np.sqrt(1. / (2048))
    text_stdv = np.sqrt(1. / (2757))
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'in_image':
        w = tf.get_variable(name='w/in_image_'+ str(i),
                            shape=[2048, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'out_image':
        w = tf.get_variable(name='w/out_image_' + str(i),
                            shape=[hidden_size, 2048],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'in_text':
        w = tf.get_variable(name='w/in_text_'+ str(i),
                            shape=[2757, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'out_text':
        w = tf.get_variable(name='w/out_text_' + str(i),
                            shape=[hidden_size, 2757],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'image_hidden_state_out':
        w = tf.get_variable(name='w/image_hidden_state_out' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_in':
        w = tf.get_variable(name='w/image_hidden_state_in_' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'text_hidden_state_out':
        w = tf.get_variable(name='w/text_hidden_state_out' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'text_hidden_state_in':
        w = tf.get_variable(name='w/text_hidden_state_in_' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))


    return w


def biases(name, hidden_size, i):
    image_stdv = np.sqrt(1. / (2048))
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'image_hidden_state_out':
        b = tf.get_variable(name='b/image_hidden_state_out', shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_in':
        b = tf.get_variable(name='b/image_hidden_state_in' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'text_hidden_state_out':
        b = tf.get_variable(name='b/text_hidden_state_out' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'text_hidden_state_in':
        b = tf.get_variable(name='b/text_hidden_state_in' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'out_image':
        b = tf.get_variable(name='b/out_image_' + str(i), shape=[2048],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'out_text':
        b = tf.get_variable(name='b/out_text_' + str(i), shape=[2757],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))

    return b


def message_pass(label, x, hidden_size, batch_size, num_category, graph):
    '''
    以输入的标签为image为例 进行注释  x= hi [16,120,12]
    '''

    w_hidden_state = weights(label + '_hidden_state_out', hidden_size, 0)   # [12,12]  第一个种类对应的输出矩阵 Wi(out)
    x_all = tf.reshape(tf.matmul(
        tf.reshape(x[:,0,:], [batch_size, hidden_size]),
        w_hidden_state),
                       [batch_size, hidden_size])
    for i in range(1, num_category):
        w_hidden_state = weights(label + '_hidden_state_out', hidden_size, i)
        x_all_ = tf.reshape(tf.matmul(
            tf.reshape(x[:, i, :], [batch_size, hidden_size]),
            w_hidden_state),
                           [batch_size, hidden_size])
        x_all = tf.concat([x_all, x_all_], 1)
    x_all = tf.reshape(x_all, [batch_size, num_category, hidden_size])
    x_all = tf.transpose(x_all, (0, 2, 1))


    x_ = x_all[0]
    graph_ = graph[0]
    x = tf.matmul(x_, graph_)   # [12,120]
    for i in range(1, batch_size):
        x_ = x_all[i]
        graph_ = graph[i]
        x_ = tf.matmul(x_, graph_)
        
        x = tf.concat([x, x_], 0)  # [12*16,120]
    x = tf.reshape(x, [batch_size, hidden_size, num_category])
    x = tf.transpose(x, (0, 2, 1))  # [16,120,12]

    x_ = tf.reshape(tf.matmul(x[:, 0, :], weights(label + '_hidden_state_in', hidden_size, 0)),
                    [batch_size, hidden_size])
    for j in range(1, num_category):
        _x = tf.reshape(tf.matmul(x[:, j, :], weights(label + '_hidden_state_in', hidden_size, j)),
                        [batch_size, hidden_size])  # hi(1)=A*Wi(out)*Wj(in)*hi(0)
        x_ = tf.concat([x_, _x], 1)  # hi(1)=A*Wi(out)*Wj(in)*hi(0)
    x = tf.reshape(x_, [batch_size, num_category, hidden_size])  # [16,120,12]

    return x


def GNN(label, data, batch_size, hidden_size, n_steps, num_category, graph):
    '''
    label 以image为例 data=image_pos[16,120,2048] ,graph=graph_pos[16,120,120]
    data[:,i,:]=image_pos[:,i,:] 指的是将每件item的image[1,2048]特征映射到大小为d=16的一个潜在空间为[16,2048]
    h0表示的是每件item的特征与所对应的种类矩阵相乘的结果 h0=ri(论文中)
    '''



    gru_cell = GRUCell(hidden_size)
    w_in = weights('in_' + label, hidden_size, 0)  # [2048,12]  种类0对应节点的特征输入矩阵W0 in
    h0 = tf.reshape(tf.matmul(data[:,0,:], w_in), [batch_size, hidden_size]) #[16,12] 种类为0的item在图空间中的表示

    for i in range(1, num_category):
        w_in = weights('in_' + label, hidden_size, i) # 种类1-120对应节点的输入矩阵Wi in
        h0 = tf.concat([h0, tf.reshape(
                tf.matmul(data[:,i,:], w_in), [batch_size, hidden_size])
                          ], 1)   # [16,12*120]
    h0 = tf.reshape(h0, [batch_size, num_category, hidden_size])  # h0: [batchsize, num_category, hidden_state] 每种种类对应的item在图空间中的表示
    ini = h0
    h0 = tf.nn.tanh(h0)  # 初始化 每个种类对应的item在图中的表示 第一维度指的是一套衣服   h0指的是每件item对应在图中节点以后的初始化向量

    state = h0  # [16,120,12]
    # print(state[2])
    sum_graph = tf.reduce_sum(graph, reduction_indices=1)   # 图关系按照第二维度进行求和  # [16,1,120]
    enable_node = tf.cast(tf.cast(sum_graph, dtype=bool), dtype=tf.float32)  #转化为0 1的序列
    # print(enable_node)
    '''
    tf.cast的用处将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
    那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
    '''

    with tf.variable_scope("gnn"):
        for step in range(n_steps):
            if step > 0: tf.get_variable_scope().reuse_variables()
            x = message_pass(label, state, hidden_size, batch_size, num_category, graph)  # 每经过一次message_pass以后套装对应的item的新表示
            (x_new, state_new) = gru_cell(x[0], state[0])  # 用GRU处理hi(0)和hi(1)
            state_new = tf.transpose(state_new, (1, 0))

            state_new = tf.multiply(state_new, enable_node[0])
            state_new = tf.transpose(state_new, (1, 0))
            for i in range(1, batch_size):
                (x_, state_) = gru_cell(x[i], state[i])  # #input of GRUCell must be 2 rank, not 3 rank
                state_ = tf.transpose(state_, (1, 0))
                state_ = tf.multiply(state_, enable_node[i])
                state_ = tf.transpose(state_, (1, 0))
                state_new = tf.concat([state_new, state_], 0)
            state = tf.reshape(state_new, [batch_size, num_category, hidden_size])  # #restore: 2 rank to 3 rank  得到每个套装经过三次message以后的节点表示
            



    return state, ini


