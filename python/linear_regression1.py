import os
import numpy as np 
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# x: from 0 to 10, 100 items
x_input = np.linspace(0, 10, 100)
weight = 5
bias = 2.5
y_input = weight * x_input + bias

W = tf.Variable(tf.random_normal([1]), name = "weight")
B = tf.Variable(tf.random_normal([1]), name = "bias")

# Placeholders
with tf.name_scope("input"):
	X = tf.placeholder(tf.float32, name = "X")
	Y = tf.placeholder(tf.float32, name = "Y")

# Model
with tf.name_scope("model"):
	y_pred = tf.add(tf.multiply(X, W), B)

# Loss
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.square(y_pred - Y))

optimizer = tf.train.GradientDescentOptimizer(0.01)

# Training step
train = optimizer.minimize(loss)

# Initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost = tf.summary.scalar("loss", loss)

sess.run(init)
epoch = 2001

merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(r"C:\Users\drexa\git\machine-learning\python\linear", graph = tf.get_default_graph())
for step in range(epoch):
	[_, accuracy, summary] = sess.run([train, loss, merged_summary_op], feed_dict = {X: x_input, Y: y_input})
	summary_writer.add_summary(summary, step)
	if step % 50 == 0:
		print(accuracy)
		
train_writer.close()

print("\nModel parameters:")       
print("Weight: %f" % sess.run(W))
print("bias: %f" % sess.run(B))
