# 参考 https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-01-classifier/
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 读取训练，测试的图片，标签数据，训练：60,000 个手写数字图片， 测试10,000个
mnist = input_data.read_data_sets('./data', one_hot=True)
import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None,):
    """
    add one more layer and return the output of this layer
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


def compute_accuracy(v_xs, v_ys):
    """
    计算准确性
    """
    global prediction
    # 使用模型对测试的图片进行识别
    y_pre = sess.run(prediction, feed_dict={pictures: v_xs})
    # 使用识别后的数字 与 测试的正确标签对比
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    # 得到识别的正确概率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={pictures: v_xs, ys: v_ys})
    return result

# 测试的图片都是28x28大小， 所有定义了一个占位符，可以输入任意个的图片
pictures = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(pictures, 784, 10, activation_function=tf.nn.softmax)

# loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
reduction_indices=[1])) # loss

# 使用GradientDescent优化器，优化训练速度， 0.5 mean 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(3000)
    sess.run(train_step, feed_dict={pictures: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))




