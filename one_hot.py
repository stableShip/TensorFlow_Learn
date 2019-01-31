import tensorflow as tf
tf.enable_eager_execution()

indices = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(tf.one_hot(indices=indices, on_value=1, off_value=0, depth=10))

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
