import tensorflow as tf
tf.enable_eager_execution()

label = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(tf.one_hot(label, 10, 1, 0))

# print:
# tf.Tensor(
# [[1 0 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 0 0]
#  [0 0 1 0 0 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0 0]
#  [0 0 0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 0 0 0 0 0 1]], shape=(10, 10), dtype=int32)
