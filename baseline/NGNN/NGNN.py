import tensorflow as tf
import numpy as np
import json
from load_data_NGNN import load_num_category, load_graph, load_train_data, load_train_size, load_fitb_data, load_test_size, load_auc_data

from model_NGNN import GNN
from datetime import *

import os



##################load data###################
ftrain = open('/data/train_no_dup_hyper.json', 'r')
train_outfit_list = json.load(ftrain)
ftest = open('/data/test_no_dup_hyper.json', 'r')
test_outfit_list = json.load(ftest)

def cm_ggnn(batch_size, image_hidden_size, n_steps, learning_rate, G, num_category, opt, i):

    hidden_stdv = np.sqrt(1. / (image_hidden_size))  # 求1. / (image_hidden_size)的平方根 0.28867513459481287
    if i == 0:
        with tf.variable_scope("cm_ggnn", reuse=None):
            w_conf_image = tf.get_variable(name='gnn/w/conf_image', shape=[image_hidden_size, 1],
                                           initializer=tf.random_normal_initializer(hidden_stdv))
            w_score_image = tf.get_variable(name='gnn/w/score_image', shape=[image_hidden_size, 1],
                                            initializer=tf.random_normal_initializer(hidden_stdv))
    else:
        with tf.variable_scope("cm_ggnn"):
            tf.get_variable_scope().reuse_variables()

    #################feed#######################
    image_pos = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    image_neg = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    graph_pos = tf.placeholder(tf.float32, [batch_size, num_category, num_category])
    graph_neg = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

    ##################GGNN's output###################

    with tf.variable_scope("gnn_image", reuse=None):
        image_state_pos, image_ini = GNN('image', image_pos, batch_size, image_hidden_size, n_steps, num_category, graph_pos)  #output: [batch_size, num_category, 2048]

        tf.get_variable_scope().reuse_variables()
        image_state_neg, text_ini = GNN('image', image_neg, batch_size, image_hidden_size, n_steps, num_category, graph_neg)

    ##################predict positive###################
    for i in range(batch_size):

        image_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_pos[i], w_conf_image), [1, num_category]))
        image_score_pos = tf.reshape(tf.matmul(image_state_pos[i], w_score_image), [num_category, 1])  #
        image_score_pos = tf.maximum(0.01 * image_score_pos, image_score_pos)
        image_score_pos = tf.reshape(tf.matmul(image_conf_pos, image_score_pos), [1])
        
        image_conf_neg = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_neg[i], w_conf_image), [1, num_category]))  # [1,120]
        image_score_neg = tf.reshape(tf.matmul(image_state_neg[i], w_score_image), [num_category, 1])  # [120,1]
        image_score_neg = tf.maximum(0.01 * image_score_neg, image_score_neg)  # [120,1]
        image_score_neg = tf.reshape(tf.matmul(image_conf_neg, image_score_neg), [1])


        if i == 0:
            s_pos = image_score_pos
            s_neg = image_score_neg
        else:
            s_pos = tf.concat([s_pos, image_score_pos], 0)  # [16,1]
            s_neg = tf.concat([s_neg, image_score_neg], 0)

    s_pos = tf.reshape(s_pos, [batch_size, 1])
    s_neg = tf.reshape(s_neg, [batch_size, 1])

    s_pos_mean = tf.reduce_mean(s_pos)
    s_neg_mean = tf.reduce_mean(s_neg)

    ##################cost, optimizer###################
    cost_parameter = 0.
    num_parameter = 0.
    for variable in tf.trainable_variables():
        print (variable)
        cost_parameter += tf.contrib.layers.l2_regularizer(0.1)(variable)
        num_parameter += 1.
        print(num_parameter)


    cost_parameter /= num_parameter
    score = tf.nn.sigmoid(s_pos - s_neg)
    score_mean = tf.reduce_mean(score)
    cost = -score_mean
    if opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    if opt == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
    if opt == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    if opt == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        best_accurancy = 0.
        best_auc = 0.
        best_epoch = 0
        # saver = tf.train.Saver()

        train_size, train_size_ = load_train_size()
        print ('train_size is %d' % train_size_)
        train_batch = int(train_size_ / batch_size)
        print ('train_batch is %d' % train_batch)

        for epoch in range(30):
            #########train##########
            test_interval = 2000
            if epoch > 10:
                test_interval = 1000

            no_count = 0
            c_all = 0.
            score_all = 0.
            vt_all = 0.
            dis_pos_all = 0.
            dis_neg_all = 0.
            for i in range(train_batch):
                train_image_pos, train_image_neg, \
                train_graph_pos, train_graph_neg, size_ = load_train_data(i, batch_size, train_outfit_list)
                if size_ >= batch_size:
                    image_pos_ = train_image_pos[0: batch_size]
                    image_neg_ = train_image_neg[0: batch_size]
                    train_graph_pos_ = train_graph_pos[0: batch_size]
                    train_graph_neg_ = train_graph_neg[0: batch_size]

                    _, c, score,  dis_pos_, dis_neg_ = sess.run(
                            [optimizer, cost, score_mean, s_pos_mean, s_neg_mean],
                                    feed_dict={image_pos: image_pos_,
                                               image_neg: image_neg_,
                                               graph_pos: train_graph_pos_,
                                               graph_neg: train_graph_neg_})
                    c_all += c
                    score_all += score
                    dis_pos_all += dis_pos_
                    dis_neg_all += dis_neg_


                    if i % test_interval == 0:
                        print ('now batch: %d, total batch: %d' % (i, train_batch))

                        print('less than batch size: %d' % no_count)
                        c_average = c_all / (i + 1)
                        score_average = score_all / (i + 1)
                        dis_pos_average = dis_pos_all / (i + 1)
                        dis_neg_average = dis_neg_all / (i + 1)

                        ############test############
                        test_size_fitb = load_test_size()
                        batches = int((test_size_fitb * 4) / batch_size)
                        right = 0.
                        for ii in range(batches):
                            test_fitb = load_fitb_data(ii, batch_size, test_outfit_list)
                            answer = sess.run([s_pos], feed_dict={image_pos: test_fitb[0],
                                                                  graph_pos: test_fitb[1]})
                            answer = np.asarray(answer[0])

                            for j in range(int(batch_size / 4)):
                                a = []
                                for k in range(j * 4, (j + 1) * 4):
                                    a.append(answer[k][0])
                                if np.argmax(a) == 0:
                                    right += 1.

                        print('answer1:', answer)
                        accurancy = float(right / test_size_fitb)
                        if accurancy > best_accurancy:
                            best_accurancy = accurancy
                            best_epoch = epoch

                        ####### AUC #######
                        test_size_auc = load_test_size()
                        batches = int((test_size_auc * 2) / batch_size)
                        right = 0.
                        for ii in range(batches):
                            test_auc = load_auc_data(ii, batch_size, test_outfit_list)
                            answer = sess.run([s_pos], feed_dict={image_pos: test_auc[0],
                                                                    graph_pos: test_auc[1]})
                            answer = np.asarray(answer[0])

                            for j in range(int(batch_size / 2)):
                                a = []
                                for k in range(j * 2, (j + 1) * 2):
                                    a.append(answer[k][0])
                                if np.argmax(a) == 0:
                                    right += 1.

                        print('answer2:',answer)
                        auc = float(right / test_size_auc)

                        if auc > best_auc:
                            best_auc = auc
                            # saver.save(sess, "trained_model/cm_ggnn.ckpt")

                        print('now():' + str(datetime.now()))
                        print("Train Epoch:", '%d' % epoch, "Batch:", '%d' % i,
                              "total cost:", "{:.9f}".format(c_average),
                              "pred score distance:", "{:.9f}".format(score_average),
                              "postive score:", "{:.9f}".format(dis_pos_average),
                              "negative score:", "{:.9f}".format(dis_neg_average),
                              "accurancy:", "{:.9f}".format(accurancy), "auc:", "{:.9f}".format(auc))
                        print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
                              "Best auc: %f" % best_auc,
                              "Best epoch: %d" % best_epoch)
                        print("batch_size: %d, image_hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                            batch_size, image_hidden_size, n_steps, learning_rate))

                else:
                    no_count += 1

            c_average = c_all / train_batch
            score_average = score_all / train_batch
            dis_pos_average = dis_pos_all / train_batch
            dis_neg_average = dis_neg_all / train_batch

            print("Train Epoch:", '%d' % epoch, "finished",
                  "total cost:", "{:.9f}".format(c_average), "pred score distance:", "{:.9f}".format(score_average),
                  "postive score:", "{:.9f}".format(dis_pos_average),
                  "negative score:", "{:.9f}".format(dis_neg_average))
            print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy, "Best auc: %f" % best_auc,
                  "Best epoch: %d" % best_epoch)
            print("batch_size: %d, image_hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                batch_size, image_hidden_size,  n_steps, learning_rate))

            ############test############
            batches = int((test_size_fitb * 4) / batch_size)   # 674
            right = 0.
            for i in range(batches):
                test_fitb = load_fitb_data(i, batch_size, test_outfit_list)
                answer = sess.run([s_pos], feed_dict={image_pos: test_fitb[0],
                                            graph_pos: test_fitb[1]})

                answer = np.asarray(answer[0])

                for j in range(int(batch_size / 4)):
                    a = []
                    for k in range(j * 4, (j + 1) * 4):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.
            print('answer3:',answer)
            accurancy = float(right / test_size_fitb)

            ##### AUC #####
            batches = int((test_size_auc * 2) / batch_size)
            right = 0.
            for i in range(batches):
                test_auc = load_auc_data(i, batch_size, test_outfit_list)
                answer = sess.run([s_pos], feed_dict={image_pos: test_auc[0],
                                                       graph_pos: test_auc[1]})
                answer = np.asarray(answer[0])
                for j in range(int(batch_size / 2)):
                    a = []
                    for k in range(j * 2, (j + 1) * 2):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.

            print('answer4:',answer)
            auc = float(right / test_size_auc)

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch

            if accurancy > best_accurancy:
                best_accurancy = accurancy
                best_epoch = epoch
                # saver.save(sess, "multi_modal/cm_ggnn.ckpt")

            print("Test Epoch:", '%d' % epoch, "accuracy:", "{:.9f}".format(accurancy), "auc:", "{:.9f}".format(auc))

            print('now():' + str(datetime.now()))
            print("batch_size: %d, image_hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                batch_size, image_hidden_size,  n_steps, learning_rate))
            print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy, "Best auc: %f" % best_auc,
                  "Best epoch: %d" % best_epoch)

    return best_accurancy


def look_enable_node(graph):
    if_enable = np.sum(graph, axis=1)
    index_list = []
    for index, value in enumerate(if_enable):
        if value > 0:
            index_list.append(index)
    return index_list


if __name__ == '__main__':

    num_category = load_num_category()
    G = load_graph()
    best_accurancy = 0.
    i = 0
    batch_size = 16
    image_hidden_size = 12
    n_steps = 2
    learning_rate = 0.001
    opt = "RMSProp"
    accurancy = cm_ggnn(batch_size, image_hidden_size,  n_steps, learning_rate, G, num_category, opt, i)
    print("best parameter is batch_size, image_hidden_size, n_steps, learning_rate, optimizer:%d, %d , %d, %f, %s" % (batch_size,
    image_hidden_size,  n_steps, learning_rate, opt))


