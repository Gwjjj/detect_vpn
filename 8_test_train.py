import re
import tensorflow as tf
import pickle
import numpy as np
import random
import sklearn as sk
import time;  # 引入time模块
import os
import traceback
import myUtils as mu


frames_min_threshold = int(mu.getConfig('parameter', 'frames_min_threshold'))  # 一个有效的会话中必须包含此数值以上的帧
frames_max_threshold = int(mu.getConfig('parameter', 'frames_max_threshold'))  # 一个有效的会话中最多包含此数值以上的帧


vpn_pickle_save_dir = mu.getConfig('URI', 'vpn_pickle_save_dir') 
novpn_pickle_save_dir = mu.getConfig('URI', 'novpn_pickle_save_dir') 
vpn_pickle_save_file = os.path.join(vpn_pickle_save_dir, 'vpn' +'_' + str(frames_min_threshold) +'_' + str(frames_max_threshold) + '.pkl')
novpn_pickle_save_file = os.path.join(novpn_pickle_save_dir, 'novpn' +'_' + str(frames_min_threshold) +'_' + str(frames_max_threshold) + '.pkl')




# 训练参数

frame_max_length = int(mu.getConfig('parameter', 'frame_max_length'))  # 流的最长字节数，大于截断，小于补0

num_classes = 2

sequence_length = int(mu.getConfig('parameter', 'frames_max_threshold'))  # 一个有效的会话中最多包含此数值以上的帧    

filter_sizes = [3, 4, 5]

num_filters = 64

batch_num = 60 


# 加载向量文件
def init_data():
    try:
        with open(vpn_pickle_save_file,'rb') as f:
            vpn_v = pickle.load(f)
        with open(novpn_pickle_save_file,'rb') as f:
            novpn_v = pickle.load(f)
        print("init dataset end")
        return vpn_v, novpn_v
    except Exception as ex:
        print(ex)
        print(traceback.format_exc())
    


def init_vaild_data():
    vaild_data_array = []
    X_ = vaild_vpn_list.copy()
    X_.extend(vaild_novpn_list)
    v_X = vaild_vpn_list
    nv_X = vaild_novpn_list
    Y_ = []
    v_Y = []
    nv_Y = []
    for _ in range(vaild_vpn_count):   
        Y_.append([0,1])
        v_Y.append([0,1])
    for _ in range(vaild_novpn_count):   
        Y_.append([1,0])
        nv_Y.append([1,0])

    vaild_data_array.append([X_,Y_])
    vaild_data_array.append([v_X,v_Y])
    vaild_data_array.append([nv_X,nv_Y])

    return vaild_data_array

vpn_vec, novpn_vec = init_data()

random.seed(8)

random.shuffle(vpn_vec)
random.shuffle(novpn_vec)


rate = 0.80
vpn_train_count = int(len(vpn_vec) * rate)
novpn_train_count = int(len(novpn_vec) * rate)
vaild_vpn_count = len(vpn_vec) - vpn_train_count
vaild_novpn_count = len(novpn_vec) - novpn_train_count

train_vpn_list =vpn_vec[: vpn_train_count]
train_novpn_list =novpn_vec[: novpn_train_count]
vaild_vpn_list =vpn_vec[vpn_train_count:]
vaild_novpn_list =novpn_vec[novpn_train_count:]


def get_train_label():
    # 是vpn
    if random.randint(0,1):
        vpn_c_eid = random.randint(0,vpn_train_count-1)
        X_ = train_vpn_list[vpn_c_eid]
        Y_ = np.array([0,1])
    else:
        # 不是vpn
        novpn_c_eid = random.randint(0,novpn_train_count-1)
        X_ = train_novpn_list[novpn_c_eid]
        Y_ = np.array([1,0])
    return X_, Y_


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')


# def max_pool(x):
#     return tf.nn.max_pool(x, ksize=[1, sequence_length-filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

# l2_loss = tf.constant(0.0)

x = tf.placeholder(tf.float32, [None, sequence_length, frame_max_length], name="x")
x_rs = tf.reshape(x, [-1, sequence_length, frame_max_length, 1])
y_ = tf.placeholder(tf.float32, [None, 2])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


x_pool = tf.nn.max_pool(
            x_rs,
            ksize=[1, 1, frame_max_length, 1], 
            strides=[1, 1, 1, 1],
            padding='VALID',
            )
x_pool_flat = tf.reshape(x_pool, [-1, sequence_length])
# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        # filter_size 分别为3 4 5
        filter_shape = [filter_size, frame_max_length, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d( # [None,56-3+1,1,128] [None,56-4+1,1,128] [None,56-5+1,1,128]
            x_rs,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool( 
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1], 
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        print(pooled)
        pooled_outputs.append(pooled)


# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
final_pool = tf.concat([x_pool_flat, h_pool_flat], 1)
num_filters_total += sequence_length
# 全连接dropout
h_drop = tf.nn.dropout(final_pool, dropout_keep_prob)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

# Calculate mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_)
    loss = tf.reduce_mean(losses)

# Accuracy
with tf.name_scope("accuracy"):
    y_true = tf.argmax(y_, 1)
    correct_predictions = tf.equal(predictions, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



# saver = tf.train.Saver(max_to_keep=1)
# model_saver = "vpn_novpn"

# ckpt_state = tf.train.get_checkpoint_state(model_saver)
# saver = tf.train.Saver()
# saver.restore(sess, ckpt_state.model_checkpoint_path)
# x_va = []
# y_va = []
# for vaild_rueid in vaild_rueid_list:
#     xv, yv = get_vaild_label(vaild_rueid, 1)
#     x_va.append(xv)
#     y_va.append(yv)
# for novaild_rueid in vaild_norueid_list:
#     xv, yv = get_vaild_label(novaild_rueid, 0)
#     x_va.append(xv)
#     y_va.append(yv)


vaild_data_array = init_vaild_data()


# 训练过程
acc_m = 0
for i in range(100000):
    X_train = []
    y = []
    for _ in range(batch_num):
       xx, yy = get_train_label()
       X_train.append(xx)
       y.append(yy)
    sess.run(train_step, feed_dict={x: X_train, y_: y, dropout_keep_prob: 0.5})
    if i % 500 == 0:
        acc = sess.run(accuracy, feed_dict={x: vaild_data_array[0][0], y_: vaild_data_array[0][1], dropout_keep_prob: 1})
        print(i, "iters: ", acc)
        localtime = time.asctime( time.localtime(time.time()) )
        print("本地时间为 :", localtime)
        if 0.945 < acc < 0.97:
            break

saver = tf.train.Saver(max_to_keep=1)
model_save_path = "vpn_novpn" + str(frames_min_threshold) +'_' + str(frames_max_threshold) + "/model"
saver.save(sess=sess, save_path=model_save_path)









for vaild_data in vaild_data_array[:1]:
    _, pre, y_t, sc = sess.run([accuracy, predictions, y_true, scores], feed_dict={x: vaild_data[0], y_: vaild_data[1], dropout_keep_prob: 1.0})
    acc = sk.metrics.accuracy_score(y_t, pre)
    Precision =  sk.metrics.precision_score(y_t, pre,average=None)
    Recall =  sk.metrics.recall_score(y_t, pre,average=None)
    f1_score =  sk.metrics.f1_score(y_t, pre,average=None)
    print("accuracy", str(acc))
    print("Precision", str(Precision))
    print("Recall", str(Recall))
    print("f1_score", str(f1_score))
