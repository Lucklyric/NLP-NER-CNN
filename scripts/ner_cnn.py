import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from data_util import DataManager

IS_TRAINING = False
INIT_LEARNING_RATE = 0.00001


class NERCNN(object):
    def __init__(self):
        self.batch_size = 128
        self.keep_prob_value = 1
        self.init_learning_rate = INIT_LEARNING_RATE
        self.is_training = True
        self.net = None

        # build the net
        if self.is_training is True:
            name_prefix = "train"
        else:
            name_prefix = "test"
        with tf.variable_scope(name_prefix + "_inputs"):
            self.input = tf.placeholder(tf.float32, [None, 70, 500], name="input_map")
            self.output = tf.placeholder(tf.float32, [None, 12, 500], name="output_map")

        with tf.variable_scope(name_prefix + "config"):
            self.keep_prob = tf.Variable(float(self.keep_prob_value), trainable=False, name="keep_prob")
            self.learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, name="lr")

        with tf.name_scope("CNN"):
            self.net = tf.reshape(self.input, [-1, 70, 500, 1], "input_reshape")
            self.net = self.add_c_layer(self.net, 32, [3, 27], "c1")
            self.net = self.add_c_layer(self.net, 32, [3, 27], "c2")

        with tf.name_scope("Flat"):
            self.net = tc.layers.flatten(self.net)

        with tf.name_scope("FC"):
            self.net = self.add_fc_layer(self.net, 1 * 12 * 50, "fc1")
            # self.net = self.add_fc_layer(self.net, 512, "fc2")

        with tf.name_scope("Expand"):
            self.net = self.add_fc_layer(self.net, 1 * 12 * 500, "projection")

        with tf.name_scope("Output"):
            self.net = tf.reshape(self.net, [-1, 12, 500])

        with tf.name_scope("Loss"):
            with tf.name_scope("SubLoss"):
                for i in range(5):
                    tf.add_to_collection("column loss",
                                         tf.nn.softmax_cross_entropy_with_logits(logits=self.net[:, :, i],
                                                                                 labels=self.output[:, :, i]))
                    # self.net = tf.reshape(self.net, [-1, 512, 1, 1], "expand")
                    # with tf.name_scope("DCNN"):
                    #     self.net = self.add_dc_layer(self.net, 32, [3, 1], [1, 20], "dc1")
                    # with tf.name_scope("CNN_TP"):
                    #     self.net = tc.layers.conv2d_transpose(self.net, 1, [])
            with tf.name_scope("TotalLoss"):
                self.loss = tf.div(tf.reduce_sum(tf.get_collection("column loss"), name="addLoss"), self.batch_size,
                                   name='average_loss')

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def add_c_layer(self, net, size, kernel_size, name):
        with tf.variable_scope(name):
            net = tc.layers.conv2d(net, size, kernel_size=kernel_size)
            net = tc.layers.max_pool2d(net, kernel_size=[1, 9])
            net = tf.nn.dropout(net, self.keep_prob)
        return net

    def add_fc_layer(self, net, size, name):
        with tf.variable_scope(name):
            net = tf.nn.dropout(tc.layers.fully_connected(net, size), keep_prob=self.keep_prob)
        return net

    def add_dc_layer(self, net, size, kernel_size, stride, name):
        with tf.variable_scope(name):
            net = tc.layers.conv2d_transpose(net, size, kernel_size=kernel_size, stride=stride)
        return net


def run_train(sess, model, data_instance):
    epoch = 0
    step = 0
    while epoch < 10000:
        batch_input, batch_output, new_epoch = data_instance.get_batch()
        if new_epoch:
            epoch += 1
            step = 0
        feed_dict = {
            model.input: batch_input,
            model.output: batch_output
        }
        _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
        step += 1
        print ("epoch %d, step %d, loss %f" % (epoch, step, loss))


if __name__ == "__main__":
    ner_model = NERCNN()
    session = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("log/", session.graph)
    session.run(init)
    data_manager = DataManager("../data/train", "../data/test", 50)
    run_train(session, ner_model, data_instance=data_manager)
    session.close()
