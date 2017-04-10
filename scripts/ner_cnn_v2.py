import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from data_util_v2 import DataManager
import data_util_v2

IS_TRAINING = False
INIT_LEARNING_RATE = 0.00001


class Config(object):
    batch_size = 64
    fc_keep_prob_value = 1
    cnn_keep_prob_value = 0.5
    is_training = True


class TrainConfig(Config):
    batch_size = 64
    cnn_keep_prob_value = 1


class TestConfig(Config):
    batch_size = 1
    cnn_keep_prob_value = 1
    fc_keep_prob_value = 1
    is_training = False


class NERCNN(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.fc_keep_prob_value = config.fc_keep_prob_value
        self.cnn_keep_prob_value = config.cnn_keep_prob_value
        self.init_learning_rate = INIT_LEARNING_RATE
        self.is_training = config.is_training
        self.net = None

        # build the net
        if self.is_training is True:
            name_prefix = "train"
        else:
            name_prefix = "test"
        with tf.variable_scope(name_prefix + "_inputs"):
            self.input = tf.placeholder(tf.float32, [None, 71, 400], name="input_map")
            self.output = tf.placeholder(tf.float32, [None, 400], name="output_map")

        with tf.variable_scope("config"):
            self.fc_keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")
            self.cnn_keep_prob = tf.placeholder(tf.float32, name="cnn_keep_prob")
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
            self.learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, name="lr")
            if self.is_training is True:
                tf.summary.scalar('learning rate', self.learning_rate)

        with tf.name_scope("CNN"):
            self.net = tf.reshape(self.input, [-1, 71, 400, 1], "input_reshape")  # 70*500 = 35000
            # self.net = tc.layers.conv2d(self.net, 32, [70, 5], stride=1, padding="VALID")  # 66*496*32
            # self.net = tc.layers.max_pool2d(self.net, [1, 2], stride=2)  # 33*246*32
            self.net = tc.layers.conv2d(self.net, 64, [71, 1], stride=1, padding="VALID")  # 1*400*64
            self.net = tc.layers.conv2d(self.net, 128, [1, 3], stride=1, padding="VALID")  # 1*398*128
            self.net = tc.layers.max_pool2d(self.net, [1, 2], stride=2)  #

        with tf.name_scope("Flat"):
            self.net = tc.layers.flatten(self.net)

        with tf.name_scope("FC"):
            self.net = self.add_fc_layer(self.net, 400 * 2, "fc1")

        with tf.name_scope("Output"):
            # self.net_output = self.net = self.add_fc_layer(self.net, 500, "out")
            self.net_output = self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 400, activation_fn=None),
                                                       keep_prob=self.fc_keep_prob)
        if self.is_training is False:
            return

        with tf.name_scope("Loss"):
            self.loss = tf.div(tf.nn.l2_loss(self.output - self.net_output), self.batch_size)
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.increase_step = self.global_step.assign_add(1)

    def add_c_layer(self, net, size, kernel_size, name):
        with tf.variable_scope(name):
            net = tc.layers.conv2d(net, size, kernel_size=kernel_size, padding='VALID', stride=[1, 2])
            # net = tc.layers.max_pool2d(net, kernel_size=[1, 3], stride=[1, 2])
            net = tf.nn.dropout(net, self.cnn_keep_prob)
        return net

    def add_fc_layer(self, net, size, name):
        with tf.variable_scope(name):
            net = tf.nn.dropout(tc.layers.fully_connected(net, size), keep_prob=self.fc_keep_prob)
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
            model.output: batch_output,
            model.cnn_keep_prob: model.cnn_keep_prob_value,
            model.fc_keep_prob: model.fc_keep_prob_value
        }
        _, loss, _, g_steps, merged_summary = sess.run(
            [model.train_op, model.loss, model.increase_step, model.global_step, merged],
            feed_dict=feed_dict)
        step += 1
        writer.add_summary(merged_summary, global_step=g_steps)
        if g_steps % 100 == 0:
            saver.save(sess, "model/v2/model.ckpt")
            print ("Save model")
        print ("epoch %d, step %d, loss %f" % (epoch, step, loss))


def run_evaluate(sess, model, data_instance):
    sample_input, sample_output = data_instance.get_one_sample(4, source="test")
    feed_dict = {
        model.input: np.reshape(sample_input, [1, 71, 400]),
        model.cnn_keep_prob: model.cnn_keep_prob_value,
        model.fc_keep_prob: model.fc_keep_prob_value
    }
    output = sess.run([model.net_output], feed_dict=feed_dict)
    class_output = []
    class_correct_output = []
    dic_list = range(0, 13)
    print (np.shape(output))
    for i in range(400):
        class_output.append(min(dic_list, key=lambda x: abs(x - np.asarray(output)[0, 0, i])))
        class_correct_output.append(min(dic_list, key=lambda x: abs(x - np.asarray(sample_output)[i])))
    print (class_output)
    print (class_correct_output)


if __name__ == "__main__":
    if IS_TRAINING is True:
        ner_model = NERCNN(TrainConfig)
    else:
        ner_model = NERCNN(TestConfig)
    session = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/v2", session.graph)
    session.run(init)
    ckpt = tf.train.get_checkpoint_state('model/v2')

    # Check Data file
    if tf.gfile.Exists("../data/train_np_v2.npy") is False:
        print ("not find npy data file, parse data ..")
        data_util_v2.save_to_disk("../data/train", "../data/test")
        print ("Done!")
    data_manager = DataManager("../data/train_np_v2.npy", "../data/test_np_v2.npy", ner_model.batch_size)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print('Checkpoint found')
    else:
        print('No checkpoint found')
    if IS_TRAINING is True:
        run_train(session, ner_model, data_instance=data_manager)
    else:
        run_evaluate(session, ner_model, data_instance=data_manager)

    session.close()
