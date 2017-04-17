import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import math


class Seq2SeqModel():
    """TF 1.0 Seq2Seq model"""
    EOS = 1
    PAD = 0

    def __init__(self, encoder_cell, decoder_cell, encoder_inputs, decoder_train_inputs, output_symbol_size,
                 embedding_size,
                 bidirectional=True, attention=False, train=True):
        self.bidirectional = bidirectional
        self.attention = attention

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        self.is_training = train
        self.encoder_inputs = encoder_inputs
        if self.is_training:
            self.decoder_train_inputs = decoder_train_inputs
        self.batch_size, _, _ = tf.unstack(tf.shape(self.encoder_inputs))
        self.encoder_inputs_length = tf.ones([self.batch_size], dtype=tf.int32) * 50
        self.decoder_targets_length = tf.ones([self.batch_size], dtype=tf.int32) * 50

        self.output_symbol_size = output_symbol_size
        self.embedding_size = embedding_size

        self._build_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _build_graph(self):
        if self.is_training:
            self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_bidirectional_encoder()

        self._init_decoder()
        if self.is_training:
            self._init_optimizer()

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.output_symbol_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            if self.is_training:
                self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix,
                                                                            self.decoder_train_inputs)

    def _init_decoder_train_connectors(self):
        batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_train_inputs))

        self.decoder_train_length = self.decoder_targets_length + 1
        PAD_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.PAD
        self.decoder_targets = tf.concat([self.decoder_train_inputs[:, 1:], PAD_SLICE], axis=1)  # Remove EOS

        self.loss_weights = tf.ones([batch_size, tf.reduce_max(self.decoder_train_length)])

    def _init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder"):
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=False,
                                                dtype=tf.float32)
            )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tc.layers.fully_connected(outputs, self.output_symbol_size, activation_fn=None, scope=scope)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(
                        self.encoder_inputs_length),
                    num_decoder_symbols=self.output_symbol_size
                )
            else:
                (attention_keys,
                 attention_values,
                 attention_score_fn,
                 attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=self.encoder_outputs,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length),
                    num_decoder_symbols=self.output_symbol_size,
                )
            if self.is_training:
                (self.decoder_outputs_train,
                 self.decoder_state_train,
                 self.decoder_context_state_train) = (
                    seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell,
                        decoder_fn=decoder_fn_train,
                        inputs=self.decoder_train_inputs_embedded,
                        sequence_length=self.decoder_train_length,
                        time_major=False,
                        scope=scope,
                    )
                )

                self.decoder_logits_train = output_fn(self.decoder_outputs_train)
                self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                          name='decoder_prediction_train')

                scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=False,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1,
                                                          name='decoder_prediction_inference')

    def _init_optimizer(self):

        self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_inference, targets=self.decoder_targets,
                                          weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
