import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import data_util_v5
from Seq2SeqModel import Seq2SeqModel
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell

IS_TRAINING = False
INIT_LEARNING_RATE = 0.0075


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

        with tf.variable_scope("inputs"):
            self.input = tf.placeholder(tf.float32, [None, 50, 70, 20], name="input_map")
            self.output = tf.placeholder(tf.float32, [None, 51], name="output_seq")

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
                    self.net = tc.layers.conv2d(self.net, 128, [70, 5], stride=1,
                                                padding="VALID")  # ==> [batch*50,1,16,64]
                    self.net = tf.transpose(self.net, [0, 3, 2, 1])  # ==> [batch*50,64,16,1]
                    self.net = tf.nn.dropout(self.net, keep_prob=self.cnn_keep_prob)
                    self.net = tc.layers.max_pool2d(self.net, [1, 16], stride=1)  # ==> [batch*50,64,1]

                with tf.name_scope("Flat"):
                    self.net = tc.layers.flatten(self.net)

                with tf.name_scope("WE-reshape"):
                    self.net = tf.reshape(self.net, [-1, 51, 128], name="WE-reshape")

        with tf.name_scope("seq_2_seq"):
            self.seq2seq_model = Seq2SeqModel(encoder_cell=LSTMCell(256),
                                              decoder_cell=LSTMCell(256),
                                              encoder_inputs=self.net,
                                              decoder_train_inputs=self.output,
                                              output_symbol_size=12,
                                              embedding_size=128,
                                              bidirectional=True,
                                              attention=True)

            self.net_output = self.seq2seq_model.decoder_prediction_inference

            self.loss = self.seq2seq_model.loss

            self.train_op = self.seq2seq_model.train_op
            self.increase_step = self.global_step.assign_add(1)


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
    in_data, target_data = data_instance.get_data(source="test")
    test_output = []
    for i in range(len(in_data)):
        sample_input = in_data[i]
        feed_dict = {
            model.input: np.expand_dims(sample_input, 0),
            model.cnn_keep_prob: model.cnn_keep_prob_value,
            model.fc_keep_prob: model.fc_keep_prob_value
        }
        output = sess.run([model.net_output], feed_dict=feed_dict)
        test_output.append(output[0][0])

    print (np.shape(target_data))
    print (np.shape(np.asarray(test_output)))

    data_util_v5.final_evaluate(np.asarray(test_output), target_data)


if __name__ == "__main__":
    if IS_TRAINING is True:
        ner_model = NERCNN(TrainConfig)
    else:
        ner_model = NERCNN(TestConfig)
    session = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Check folders
    if tf.gfile.Exists("log/v5") is False:
        tf.gfile.MkDir("log/v5")
    if tf.gfile.Exists("model/v5") is False:
        tf.gfile.MkDir("model/v5")

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/v5", session.graph)
    session.run(init)
    ckpt = tf.train.get_checkpoint_state('model/v5')
    if tf.gfile.Exists("../data/train_in_np_v5.npy") is False:
        print ("not find npy data file, parse data ..")
        data_util_v5.save_to_disk("../data/train", "../data/test")
        print ("Done!")
    data_manager = data_util_v5.DataManager("../data/train_in_np_v5.npy", "../data/train_out_np_v5.npy",
                                            "../data/test_in_np_v5.npy", "../data/test_out_np_v5.npy", 64)
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
