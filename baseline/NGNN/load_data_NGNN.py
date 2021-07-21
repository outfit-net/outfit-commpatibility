import numpy as np
import json
import pickle
import random
# image_x = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
# image_y = tf.placeholder(tf.float32, [batch_size, 2048])
# category_y = tf.placeholder(tf.int32, [batch_size])
# grah = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

# fc = open('category_summarize_100.json', 'r')
# dict_list = json.load(fc)

fic = open('/home/ndn/桌面/HyperGCN/data/category_summarize_100.json', 'r')
category_item = json.load(fic)
num_category = len(category_item)
# print('more than 100 category: %d' % num_category)
fr = open('/home/ndn/桌面/HyperGCN/data/cid2rcid_100.json', 'r')
cid2rcid = json.load(fr)

image_feature_path='/home/ndn/桌面/HyperGCN/data/polyvore_image_vectors/'

# text_feature_path = 'F:\\NGNN\\NGNN\\data\\polyvore_text_onehot_vectors\\'

total_graph = np.zeros((num_category, num_category), dtype=np.float32)   # 构建一个item与item之间的图关系
per_outfit = 8


def load_graph():
    return total_graph

def load_num_category():
    return num_category

def load_train_data(i, batch_size, outfit_list):
    size_ = batch_size  # 16
    time = int(batch_size / per_outfit)  #2
    i = i * time  #0

    image_pos = np.zeros((size_, num_category, 2048), dtype=np.float32)  # [16,120,2048]
    image_neg = np.zeros((size_, num_category, 2048), dtype=np.float32)   # [16,120,2048]
    # text_pos = np.zeros((size_, num_category, 2757), dtype=np.float32)    # [16,120,2757]
    # text_neg = np.zeros((size_, num_category, 2757), dtype=np.float32)    # [16,120,2757]
    graph_pos = np.zeros((size_, num_category, num_category), dtype=np.float32)   # [16,120,120]
    graph_neg = np.zeros((size_, num_category, num_category), dtype=np.float32)   # [16,120,120]
    now_size = 0


    for outfit in outfit_list[i : i + time]:  # 每次只读取2套套装  \\\
        ii = outfit['items_index']   # 读取这套衣服所组成item的索引
        ci = outfit['items_category']  # 读取这套衣服组成item的种类
        sid = outfit['set_id']   # 读取这套衣服的id
        len_ = len(ii)  # 是由多少件item组成的
        j_list = []
        '''
        下面的注释都是以train_no_dup_new_100.json也就是训练集中的第一套衣服为例所给出
        '''
        for j in range(len_):
            j_list.append(j)  #[0,1,2,3,4,5,6]
        for j in range(per_outfit - len_):
            j_list.append(random.randint(0, len_ - 1))  #[0,1,2,3,4,5,6, 1(随机生成的0-6之间的一个数)]
        for j in j_list:
            list_ = [] # j=0的循环中，list_=[16,18,89,88,108,2]  每次循环list_都会清0
            for k in range(len(ii)):
                cid = ci[k]  # 读取套装的组成的第一件item种类
                iid = ii[k]  # 读取套装的组成的第一件item索引
                rcid = int(cid2rcid[str(cid)])  # 将第一件item种类转化为关系图上对应的数字
                image_feature = json.load(open(image_feature_path + str(sid)+ '_' + str(iid)+ '.json'))  # 读取对应每件item的图像特征
                if k == j:
                    image_pos[now_size][rcid] = image_feature  # 指的是每件item在对应种类上的特征信息
                    rcid_pos = rcid

                    i = random.choice(category_item)  # 随机选择一种衣服类别
                    rcid_neg = cid2rcid[str(i['id'])] # 读取衣服类别所对应的种类id
                    ri = random.choice(i['items'])    # 随机读取该种类中包含的一件item ID号
                    image_neg[now_size][rcid_neg] = json.load(open(image_feature_path + ri + '.json'))

                else:
                    image_neg[now_size][rcid] = image_feature
                    image_pos[now_size][rcid] = image_feature
                    list_.append(rcid)
                    '''
                    读取第一套服装中的每件item，并将它们对应的文本特征和图像特征分别添加到image_neg/pos，text_neg/pos中
                    '''

            list_pos = list_[:]  # [16,18,89,88,108,2]
            list_pos.append(rcid_pos)  # [16,18,89,88,108,2,118]
            for a in list_pos:
                for b in list_pos:
                    if b != a:
                        graph_pos[now_size][a][b] = 1.  # 将套装中的同时出现的两件item添加到graph_pos中，a，b指的是每件item的类别
                        total_graph[a][b] = 1.  # 将套装中的同时出现的两件item添加到total_graph中 生成套装的图关系
            list_neg = list_[:]  # [16,18,89,88,108,2]
            list_neg.append(rcid_neg)  # [16,18,89,88,108,2,m]
            for a in list_neg:
                for b in list_neg:
                    if b != a:
                        graph_neg[now_size][a][b] = 1.  # 将套装中的同时出现的两件item添加到graph_neg中

            now_size += 1

    return image_pos, image_neg, graph_pos, graph_neg, size_
'''
load_train_data函数指的是对于读取的每套套装，替换套装中的每件item，相当于生成了只更换一件item的不同的套
装然后生成对应套装的图像特征，文本特征和图关系
该函数用来预测兼容性
'''




def load_train_size():
    ftrain = open('/data/train_no_dup_hyper.json', 'r')
    train_list = json.load(ftrain)
    train_size = len(train_list)  # 16983
    train_size_ = per_outfit * train_size  # 135,864
    return train_size, train_size_

# train_size, train_size_=load_train_size()

def load_fitb_data(index, batch_size, outfit_list):
    time = int(batch_size / 4)  # 4
    num_category = load_num_category()  # 120
    image = np.zeros((batch_size, num_category, 2048), dtype=np.float32)   # [16,120,2048]
    graph = np.zeros((batch_size, num_category, num_category), dtype=np.float32)    # [16,120,120]
    outfit_list_ = outfit_list[index*time: (index + 1)*time]
    for i in range(len(outfit_list_)):  # 4
        outfit = outfit_list_[i]
        ii = outfit['items_index']  # 套装中每件item 的索引
        ci = outfit['items_category']   # 套装中每件item 的种类
        sid = outfit['set_id']   # 套装的ID
        rcid_list = []
        length = len(ii)  # 7
        blank_index = random.randint(0, length - 1)  # 随机生成(0, length - 1)中的一个数字
        for j in range(len(ii)):
            cid = ci[j]  # 得到第j件item的种类
            iid = ii[j]  # 得到第j件item的索引
            rcid = int(cid2rcid[str(cid)])  # 得到第j件item种类对应的在total_graph中的索引
            image_feature = json.load(open(image_feature_path + str(sid) + '_' + str(iid) + '.json'))
            # text_feature = json.load(open(text_feature_path + str(sid) + '_' + str(iid) + '.json'))
            # 得到该套装中第j件item的图像特征和文本特征
            if j == blank_index:
                rcid_pos = rcid
                image[i * 4][rcid] = image_feature
                r1 = random.choice(category_item)
                rcid_w1 = cid2rcid[str(r1['id'])]
                r11 = random.choice(r1['items'])
                image[i * 4 + 1][rcid_w1] = json.load(open(image_feature_path + r11 + '.json'))


                r2 = random.choice(category_item)
                rcid_w2 = cid2rcid[str(r2['id'])]
                r22 = random.choice(r2['items'])
                image[i * 4 + 2][rcid_w2] = json.load(open(image_feature_path + r22 + '.json'))


                r3 = random.choice(category_item)
                rcid_w3 = cid2rcid[str(r3['id'])]
                r33 = random.choice(r3['items'])
                image[i * 4 + 3][rcid_w3] = json.load(open(image_feature_path + r33 + '.json'))

            else:
                rcid_list.append(rcid)
                image[i * 4][rcid] = image_feature
                image[i * 4 + 1][rcid] = image_feature
                image[i * 4 + 2][rcid] = image_feature
                image[i * 4 + 3][rcid] = image_feature

        '''
        这个for循环指的是生成每套套装的图像特征和文本特征的同时，生成负样本的套装特征，也就是替换其中的一件item以后的套装特征
        每个batch_size都是改变blank_index的特征
        '''
        rl_pos = rcid_list[:]
        rl_pos.append(rcid_pos)  # 得到原本套装中item种类在图中对应的索引
        g1 = np.zeros((num_category, num_category), dtype=np.float32)  # [120,120]
        for a in rl_pos:
            for b in rl_pos:
                if b != a:
                    g1[a][b] = 1.
        graph[i * 4] = g1  # 生成每套套装的图关系

        rl_w1 = rcid_list[:]
        rl_w1.append(rcid_w1)
        g2 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_w1:
            for b in rl_w1:
                if b != a:
                    g2[a][b] = 1.
        graph[i * 4 + 1] = g2  # 生成不同维度的不匹配套装的图关系

        rl_w2 = rcid_list[:]
        rl_w2.append(rcid_w2)
        g3 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_w2:
            for b in rl_w2:
                if b != a:
                    g3[a][b] = 1.
        graph[i * 4 + 2] = g3   # 生成不同维度的不匹配套装的图关系

        rl_w3 = rcid_list[:]
        rl_w3.append(rcid_w3)
        g4 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_w3:
            for b in rl_w3:
                if b != a:
                    g4[a][b] = 1.
        graph[i * 4 + 3] = g4   # 生成不同维度的不匹配套装的图关系

    return image, graph
'''
load_fitb_data函数指的是对于一套pos套装，生成不同的三套neg套装，而且这四种套装中只有一件item是不同的，
然后生成这四套套装的图像特征，文本特征和图关系
该函数用来计算Fill-in-the-blank的准确率
'''
def load_test_size():

    ftest = open('/data/test_no_dup_hyper.json', 'r')
    test_list = json.load(ftest)
    test_size = len(test_list)  # 2697
    return test_size



def load_auc_data(index, batch_size, outfit_list):
    time = int(batch_size / 2)    # 8
    num_category = load_num_category()  # 120
    image = np.zeros((batch_size, num_category, 2048), dtype=np.float32)   # [16,120,2048]
    graph = np.zeros((batch_size, num_category, num_category), dtype=np.float32)   # [16,120,120]
    outfit_list_ = outfit_list[index*time: (index + 1)*time]
    for i in range(len(outfit_list_)):  # len(outfit_list_)=8
        outfit = outfit_list_[i]
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']

        length = len(ii)
        graph_pos = []
        graph_neg = []
        for j in range(length):
            cid = ci[j]
            iid = ii[j]
            rcid = int(cid2rcid[str(cid)])
            image_feature = json.load(open(image_feature_path + str(sid) + '_' + str(iid) + '.json'))
            image[i * 2][rcid] = image_feature
            graph_pos.append(rcid)  # 将套装中每件item的种类在图中对应的索引添加到graph_pos中
        for j in range(length):
            r = random.choice(category_item)
            rcid_w = cid2rcid[str(r['id'])]
            rr = random.choice(r['items'])
            image[i * 2 + 1][rcid_w] = json.load(open(image_feature_path + rr + '.json'))
            graph_neg.append(rcid_w)  # 生成与每件套装item数量相同，但是item种类不同的套装，并将不同种类的item索引添加到graph_neg中

        g = np.zeros((num_category, num_category), dtype=np.float32)  # [120,120]
        for a in graph_pos:
            for b in graph_pos:
                if b != a:
                    g[a][b] = 1.
        graph[i * 2] = g  # 生成graph_pos中对应的图关系

        g = np.zeros((num_category, num_category), dtype=np.float32)
        for a in graph_neg:
            for b in graph_neg:
                if b != a:
                    g[a][b] = 1.
        graph[i * 2 + 1] = g  # 生成graph_neg中对应的图关系

    return image,graph
'''
load_auc_data指的是读取每套pos套装，同时对应生成不同的neg套装，然后生成他们各自的图像特征，文本特征和图关系
该函数用来预测套装的兼容性
'''





