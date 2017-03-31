import tensorflow as tf
import numpy as np

a = tf.Variable(tf.zeros([12, 70, 500]))
sa = a[:, :, 0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t = sess.run(sa)
    print np.shape(t)
