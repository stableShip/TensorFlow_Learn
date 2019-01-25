# declare the value and operation to execute
import tensorflow as tf
a = tf.Variable([1, 2])
b = tf.Variable([1, 3])
multiply = tf.multiply(a, b)
add = tf.add(a, b)

x = tf.placeholder(tf.int32)
linear_model = a * x + b

initialize = tf.global_variables_initializer()

# create a session to execute the operation
session = tf.Session()
session.run(initialize)
print(session.run(a))
print(session.run(b))
print(session.run(multiply))
print(session.run(add))
print(session.run(linear_model, {x: [1, 7]}))
