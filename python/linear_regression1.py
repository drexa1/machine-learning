import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: from 0 to 10, 100 items
x_input = np.linspace(0, 10, 100)

weight = 12
bias = 10
y_input = weight * x_input + bias

W = tf.Variable(tf.random_normal([1]), name = 'weight')
B = tf.Variable(tf.random_normal([1]), name = 'bias')

# Placeholders
with tf.name_scope('input'):
	X = tf.placeholder(tf.float32, name = 'X')
	Y = tf.placeholder(tf.float32, name = 'Y')

# Model
with tf.name_scope('model'):
	Y_pred = tf.add(tf.multiply(X, W), B)

# Loss
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(Y_pred - Y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	# Training step
	train = optimizer.minimize(loss)

# Initialize
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf.summary.scalar('SIMPLE-loss', loss)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(r'C:\Users\drexa\git\machine-learning\python\linear', graph = tf.get_default_graph())

epochs = 2001
for step in range(epochs):
	[_, curr_loss, curr_summary] = sess.run([train, loss, merged_summary], feed_dict = {X: x_input, Y: y_input})
	writer.add_summary(curr_summary, step)
	if step % 50 == 0:
		print(curr_loss)
		
writer.close()
    
print('\nWeight: %f' % sess.run(W))
print('Bias: %f\n' % sess.run(B))