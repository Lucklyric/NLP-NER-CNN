import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from data_util import DataManager

IS_TRAINING = True
INIT_LEARNING_RATE = 0.0001


class Config(object):
    batch_size = 64
    fc_keep_prob_value = 0.5
    cnn_keep_prob_value = 0.5
    is_training = True


class TrainConfig(Config):
    batch_size = 1
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
            self.input = tf.placeholder(tf.float32, [None, 70, 500], name="input_map")
            self.output = tf.placeholder(tf.float32, [None, 12, 500], name="output_map")

        with tf.variable_scope("config"):
            self.fc_keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")
            self.cnn_keep_prob = tf.placeholder(tf.float32, name="cnn_keep_prob")
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
            self.learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, name="lr")
            if self.is_training is True:
                tf.summary.scalar('learning rate', self.learning_rate)

        with tf.name_scope("CNN"):
            self.net = tf.reshape(self.input, [-1, 70, 500, 1], "input_reshape")  # 70*500 = 35000
            self.net = self.add_c_layer(self.net, 8, [3, 3], "c1")  # 35*250*32 = 280000
            self.net = self.add_c_layer(self.net, 16, [3, 3], "c2")  # 18*125*64 = 144000

        with tf.name_scope("Flat"):
            self.net = tc.layers.flatten(self.net)

        with tf.name_scope("FC"):
            self.net = self.add_fc_layer(self.net, 1 * 12 * 500, "fc1")
            # self.net = self.add_fc_layer(self.net, 1 * 12 * 500, "fc2")
            # self.net = self.add_fc_layer(self.net, 512, "fc2")

        with tf.name_scope("Expand"):
            self.net = self.add_fc_layer(self.net, 1 * 12 * 500, "projection")

        with tf.name_scope("Output"):
            self.output = self.net = tf.reshape(self.net, [-1, 12, 500])

        if self.is_training is False:
            return

        with tf.name_scope("Loss"):
            with tf.name_scope("SubLoss"):
                for i in range(500):
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
                tf.summary.scalar('loss', self.loss)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.increase_step = self.global_step.assign_add(1)

    def add_c_layer(self, net, size, kernel_size, name):
        with tf.variable_scope(name):
            net = tc.layers.conv2d(net, size, kernel_size=kernel_size)
            net = tc.layers.max_pool2d(net, kernel_size=[1, 2])
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
            saver.save(sess, "model/model.ckpt")
            print ("Save model")
        print ("epoch %d, step %d, loss %f" % (epoch, step, loss))


def run_evaluate(sess, model, data_instance):
    sample_input, sample_output = data_instance.get_one_sample(700, source="test")
    feed_dict = {
        model.input: np.reshape(sample_input, [1, 70, 500]),
        model.cnn_keep_prob: model.cnn_keep_prob_value,
        model.fc_keep_prob: model.fc_keep_prob_value
    }
    output = sess.run([model.output], feed_dict=feed_dict)
    class_output = []
    class_correct_output = []
    for i in range(500):
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
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/", session.graph)
    session.run(init)
    ckpt = tf.train.get_checkpoint_state('model')
    data_manager = DataManager("../data/train", "../data/test", ner_model.batch_size)

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
