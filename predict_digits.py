import tensorflow as tf
import os
import config
import numpy as np
import cv2

NUM_CLASSES=config.DIGITS_NUM_CLASSES
SAVER_DIR=config.DIGITS_SAVER_DIR
saver=tf.train.import_meta_graph(os.path.join(SAVER_DIR,'model.ckpt.meta'))
HEIGHT=config.HEIGHT
WIDTH=config.WIDTH
CHANNEL_NUM=config.CHANNEL_NUM
x=tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,CHANNEL_NUM])
SIZE=config.SIZE
LETTERS_DIGITS=config.LETTERS_DIGITS
license_num=''
with tf.Session() as sess:
    model_file=tf.train.latest_checkpoint(SAVER_DIR)
    saver.restore(sess, model_file)
    conv1_w=sess.graph.get_tensor_by_name('layer1-conv1/weight:0')
    conv1_b=sess.graph.get_tensor_by_name('layer1-conv1/bias:0')

    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]

    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # 第二个卷积层
    conv2_w = sess.graph.get_tensor_by_name('layer3-conv2/weight:0')
    conv2_b = sess.graph.get_tensor_by_name('layer3-conv2/bias:0')
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = tf.nn.max_pool(relu2, [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME')

    # 全连接层
    fc1_w = sess.graph.get_tensor_by_name('layer5-fc1/weight:0')
    fc1_b = sess.graph.get_tensor_by_name('layer5-fc1/bias:0')
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [-1, nodes])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)

    fc2_w=sess.graph.get_tensor_by_name('layer6-fc2/weight:0')
    fc2_b=sess.graph.get_tensor_by_name('layer6-fc2/bias:0')

    result=tf.nn.softmax(tf.matmul(fc1, fc2_w) + fc2_b)

    img=cv2.imread('tf_car_license_dataset/test_images/7.bmp',0)
    print(img.shape)
    _,img_b=cv2.threshold(img,190,255,cv2.THRESH_BINARY_INV)

    img_data=np.reshape(img_b,[1,HEIGHT,WIDTH,CHANNEL_NUM])
    result = sess.run(result, feed_dict={x: np.array(img_data)})

    max = 0

    for j in range(NUM_CLASSES):
        if result[0][j] > max:
            max = result[0][j]
            max_index = j
            continue

    license_num = LETTERS_DIGITS[max_index]
    print("概率：  [%s %0.2f%%]" % (
        LETTERS_DIGITS[max_index], max * 100))
