import tensorflow as tf 
tf.enable_eager_execution()

tf.enable_eager_execution()
a = tf.Variable([1,2])
b = tf.Variable([3,3])
result = tf.multiply(a, b)
print(result)