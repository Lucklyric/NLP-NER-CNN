import tensorflow as tf
import numpy as np

a = np.asarray([[[1, 1, 1],
                 [2, 2, 2]],
                [[3, 3, 3],
                 [4, 4, 4]],
                [[5, 5, 5],
                 [6, 6, 6]]])

print a[1,:,:]
# t = np.reshape(a, [2, -1, 3])


# a = np.asarray([1, 1, 1, 1])
# b = a * 2
# c = a * 3
# a_tensor = tf.convert_to_tensor(a, dtype=np.float32, name="a")
# b_tensor = tf.convert_to_tensor(b, dtype=np.float32, name="b")
# c_tensor = tf.convert_to_tensor(c, dtype=np.float32, name="c")
# t_tensor = tf.transpose(tf.stack([a_tensor, b_tensor, c_tensor]))
#
# # print np.shape(t)
# # print t
# # a = tf.Variable(tf.zeros([12, 70, 500]))
# # sa = a[:, :, 0]
# #
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a_o = sess.run(a_tensor)
#     t_o = sess.run(t_tensor)
#     print (a_o)
#     print (t_o[:, 0])
