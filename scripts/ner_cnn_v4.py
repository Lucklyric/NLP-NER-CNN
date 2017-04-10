import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from data_util_v4 import DataManager
import data_util_v4

IS_TRAINING = True
INIT_LEARNING_RATE = 0.0001


class Config(object):
    batch_size = 64
    fc_keep_prob_value = 0.5
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

        with tf.variable_scope("inputs"):
            self.input = tf.placeholder(tf.float32, [None, 50, 70, 20], name="input_map")
            self.output = tf.placeholder(tf.float32, [None, 13, 50], name="output_map")

        with tf.variable_scope("config"):
            self.fc_keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")
            self.cnn_keep_prob = tf.placeholder(tf.float32, name="cnn_keep_prob")
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
            self.learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, name="lr")
            if self.is_training is True:
                tf.summary.scalar('learning rate', self.learning_rate)
        with tf.name_scope("Word-Embedding"):
            with tf.name_scope("Flat_words"):
                self.net = tf.reshape(self.input, [-1, 70, 20, 1], "input_reshape")  # ==> [batch*50,71,20,1]
                with tf.name_scope("CNN"):
                    self.net = tc.layers.conv2d(self.net, 32, [70, 1], stride=1,
                                                padding="VALID")  # ==> [batch*50,1,20,32]
                    self.net = tc.layers.conv2d(self.net, 64, [1, 3], stride=1,
                                                padding="VALID")  # ==> [batch*50,1,18,64]
                    self.net = tc.layers.conv2d(self.net, 64, [1, 3], stride=1,
                                                padding="VALID")  # ==> [batch*50,1,16,64]
                    # self.net = tc.layers.conv2d(self.net, 128, [1, 3], stride=1, padding="VALID")  # 1 * 186 * 128
                    # self.net = tc.layers.max_pool2d(self.net, [1, 2], stride=2)  # 1 * 92 * 128

                with tf.name_scope("Flat"):
                    self.net = tc.layers.flatten(self.net)

                with tf.name_scope("FC"):
                    self.net = self.add_fc_layer(self.net, 128, "fc1")
                    # self.net = self.add_fc_layer(self.net, 1 * 12 * 500, "fc2")
                    # self.net = self.add_fc_layer(self.net, 512, "fc2")

                with tf.name_scope("WE-reshape"):
                    self.net = tf.reshape(self.net, [-1, 50, 128, 1], name="WE-reshape")
                    self.net = tf.transpose(self.net, [0, 2, 1, 3])

        with tf.name_scope("CNN"):
            self.net = tc.layers.conv2d(self.net, 32, [128, 1], stride=1,
                                        padding="VALID")  # ==> [batch*50,1,20,32]
        with tf.name_scope("Flat"):
            self.net = tc.layers.flatten(self.net)

        with tf.name_scope("FC"):
            self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 13 * 50),
                                     keep_prob=self.fc_keep_prob)

            self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 13 * 50, activation_fn=None),
                                     keep_prob=self.fc_keep_prob)

        with tf.name_scope("Output"):
            self.net_output = self.net = tf.reshape(self.net, [-1, 13, 50])

        if self.is_training is False:
            return

        with tf.name_scope("Loss"):
            with tf.name_scope("SubLoss"):
                for i in range(50):
                    tf.add_to_collection("column loss",
                                         tf.nn.softmax_cross_entropy_with_logits(logits=self.net[:, :, i],
                                                                                 labels=self.output[:, :, i]))
            with tf.name_scope("TotalLoss"):
                self.loss = tf.div(tf.reduce_sum(tf.get_collection("column loss"), name="addLoss"), self.batch_size,
                                   name='average_loss')
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
            saver.save(sess, "model/v4/model.ckpt")
            print ("Save model")
        print ("epoch %d, step %d, loss %f" % (epoch, step, loss))


def run_evaluate(sess, model, data_instance):
    sample_input, sample_output = data_instance.get_one_sample(1, source="test")
    feed_dict = {
        model.input: np.expand_dims(sample_input, 0),
        model.cnn_keep_prob: model.cnn_keep_prob_value,
        model.fc_keep_prob: model.fc_keep_prob_value
    }
    output = sess.run([model.net_output], feed_dict=feed_dict)
    class_output = []
    class_correct_output = []
    for i in range(50):
        class_output.append(np.argmax(np.asarray(output)[0, 0, :, i]))
        class_correct_output.append(np.argmax(np.asarray(sample_output)[:, i]))
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

    # Check folders
    if tf.gfile.Exists("log/v4") is False:
        tf.gfile.MkDir("log/v4")
    if tf.gfile.Exists("model/v4") is False:
        tf.gfile.MkDir("model/v4")

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/v4", session.graph)
    session.run(init)
    ckpt = tf.train.get_checkpoint_state('model/v4')
    if tf.gfile.Exists("../data/train_in_np.npy") is False:
        print ("not find npy data file, parse data ..")
        data_util_v4.save_to_disk("../data/train", "../data/test")
        print ("Done!")
    data_manager = data_util_v4.DataManager("../data/train_in_np.npy", "../data/train_out_np.npy",
                                            "../data/test_in_np.npy", "../data/test_out_np.npy", 64)
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
