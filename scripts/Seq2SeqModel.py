import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


class Seq2SeqModel():
    """TF 1.0 Seq2Seq model"""

    def __init__(self, encoder_cell, decoder_cell, encoder_inputs, encoder_inputs_length, decoder_targets,
                 decoder_targets_length, bidirectional=True, attention=False):
        self.bidirectional = bidirectional
        self.attention = attention

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.encoder_inputs = encoder_inputs
        self.encoder_inputs_length = encoder_inputs
        self.decoder_targets = decoder_targets
        self.decoder_targets_length = decoder_targets_length

    def _build_graph(self):
        self._init_decoder_train_connectors()

    def _init_decoder_train_connectors(self):
        batch_size, feature_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets))

        self.decoder_train_inputs = self.decoder_targets
        self.decoder_train_length = self.decoder_targets_length + 1

        EOS_SLICE = tf.ones([batch_size, feature_size], dtype=tf.int32)

        self.loss_weight = tf.ones([batch_size, tf.reduce_max(self.decoder_train_length)])
