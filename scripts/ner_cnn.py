import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import data_util

IS_TRAINING = False
INIT_LEARNING_RATE = 0.003


class NERCNN(object):
    def __init__(self):
        self.batch_size = 128
        self.keep_prob_value = 0.5
        self.init_learning_rate = INIT_LEARNING_RATE
        self.is_training = True
        self.net = None

        # build the net
        if self.is_training is True:
            name_prefix = "train"
        else:
            name_prefix = "test"
        with tf.variable_scope(name_prefix + "_inputs"):
            self.input = tf.placeholder(tf.float32, [None, 70, 500, 1], name="input_map")
            self.output = tf.placeholder(tf.float32, [None, 12, 500, 1], name="output_map")

        with tf.variable_scope(name_prefix + "config"):
            self.keep_prob = tf.Variable(float(self.keep_prob_value), trainable=False, name="keep_prob")
            self.learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, name="lr")

        with tf.name_scope("CNN"):
            self.net = self.add_c_layer(self.input, 32, [3, 64], "c1")
            self.net = self.add_c_layer(self.net, 32, [3, 64], "c2")

        with tf.name_scope("Flat"):
            self.net = tc.layers.flatten(self.net)

        with tf.name_scope("FC"):
            self.net = self.add_fc_layer(self.net, 512, "fc1")
            # self.net = self.add_fc_layer(self.net, 512, "fc2")

            # with tf.name_scope("CNN_TP"):
            #     self.net = tc.layers.conv2d_transpose(self.net, 1, [])

    def add_c_layer(self, net, size, kernel_size, name):
        with tf.name_scope(name):
            net = tc.layers.conv2d(net, size, kernel_size=kernel_size)
            # net = tc.layers.max_pool2d(net, kernel_size=[1, 3])
            net = tf.nn.dropout(net, self.keep_prob)
        return net

    def add_fc_layer(self, net, size, name):
        with tf.name_scope(name):
            net = tf.nn.dropout(tc.layers.fully_connected(net, size), keep_prob=self.keep_prob)
        return net


if __name__ == "__main__":
    model = NERCNN()
    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("log/", sess.graph)
    sess.run(init)
    sess.close()
