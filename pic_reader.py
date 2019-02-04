import sys
import os
import time
import random
import inference
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 34
iterations = 1000

SAVER_DIR = "train-saver/digits/"

LETTERS_DIGITS = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
license_num = ""


def cul_file_sum(path):
    count = 0
    for i in range(0, NUM_CLASSES):
        dir = os.path.join(path, str(i))
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                count += 1
    images = np.zeros((count, HEIGHT, WIDTH))
    labels = np.zeros((count, NUM_CLASSES))
    return count, images, labels


def gener_img_lbls(path, images, labels):
    index = 0
    for i in range(0, NUM_CLASSES):
        dir = os.path.join(path, str(i))
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + '/' + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) > 230:
                            # images[index][w + h * width] = 0
                            images[index][h][w] = 0
                        else:
                            # images[index][w + h * width] = 1
                            images[index][h][w] = 1
                labels[index][i] = 1
                index += 1
    images=images.reshape((-1, HEIGHT, WIDTH, 1))
    return images, labels


time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 1])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])

input_dir = './tf_car_license_dataset/train_images/training-set/'
input_sum, input_images, input_labels = cul_file_sum(input_dir)
input_images, input_labels = gener_img_lbls(input_dir, input_images, input_labels)

valid_dir = './tf_car_license_dataset/train_images/validation-set/'
valid_sum, valid_images, valid_labels = cul_file_sum(valid_dir)
valid_images, valid_labels = gener_img_lbls(valid_dir, valid_images, valid_labels)

time_elapsed = time.time() - time_begin
print("读取图片文件耗费时间：%d秒" % time_elapsed)

logits = inference.inference(x_image, True, None)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    time_begin = time.time()

    print("一共读取了 %s 个训练图像， %s 个标签" % (input_sum, input_sum))

    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
    batch_size = 60
    batches_count = int(input_sum / batch_size)
    remainder = input_sum % batch_size
    print("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

    # 执行训练迭代
    for it in range(1000):
        for n in range(batches_count):
            train_step.run(feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                      y_: input_labels[n * batch_size:(n + 1) * batch_size]})
        if remainder > 0:
            start_index = batches_count * batch_size;
            train_step.run(
                feed_dict={x: input_images[start_index:input_sum - 1], y_: input_labels[start_index:input_sum - 1]})

        # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
        iterate_accuracy = 0
        if it % 5 == 0:
            iterate_accuracy = accuracy.eval(feed_dict={x: valid_images, y_: valid_labels})
            print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
            if iterate_accuracy >= 0.9999 and it >= iterations:
                break;

    print('完成训练!')
    time_elapsed = time.time() - time_begin
    print("训练耗费时间：%d秒" % time_elapsed)
    time_begin = time.time()

    # 保存训练结果
    if not os.path.exists(SAVER_DIR):
        print('不存在训练数据保存目录，现在创建保存目录')
        os.makedirs(SAVER_DIR)
    # 初始化saver
    saver = tf.train.Saver()
    saver_path = saver.save(sess, "%smodel.ckpt" % (SAVER_DIR))
