import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# 250 points of 1 dimension
n = 250
d = 1
sample_x = np.random.rand(n, d)
print(sample_x)

slope = 12
bias = 10
sample_y = slope * sample_x + bias

noise = np.random.rand(n, d)
sample_x += noise

# Weight & bias
W = tf.Variable(np.array([[5.0]]), dtype = tf.float32, name = "weight")
B = tf.Variable(np.array([[5.0]]), dtype = tf.float32, name = "bias")

# Placeholders
X = tf.placeholder(tf.float32, shape = (None, 1), name = "X")
Y = tf.placeholder(tf.float32, shape = (None, 1), name = "Y")
lr_rate = tf.placeholder(tf.float32, shape = (), name = "LR")

# Model
y_labels = W * X + B

# Loss
loss = tf.div(tf.reduce_mean(tf.square(Y - y_labels)), 2*n)
# Optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr_rate)
# Training step
train = optimizer.minimize(loss)

# Summaries
def summaries(var, name):
	with tf.name_scope(name):
	 	mean = tf.reduce_mean(var)
	 	tf.summary.scalar("mean", mean)
	 	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	 	tf.summary.scalar('std-dev', stddev)
	 	maxx = tf.reduce_max(var)
	 	tf.summary.scalar('max', maxx)
	 	minn = tf.reduce_min(var)
	 	tf.summary.scalar('min', minn)
	 	tf.summary.histogram('histogram', var)

# Define summaries
summaries(W, "010-Weights")
summaries(B, "020-Biases")
summaries(loss, "030-Loss")

# Tensorboard directory
log_dir = r"C:\Users\drexa\git\machine-learning\python\linear"
lr = 0.1
epochs = 5001

# Train the model
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Merge all the summaries and write them
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(log_dir, sess.graph)
	for epoch in range(epochs):
		[y_pred, curr_w, curr_b, curr_loss, _, curr_summary] = sess.run([y_labels, W, B, loss, train, merged], feed_dict = {X:sample_x, Y: sample_y, lr_rate: lr})
		writer.add_summary(curr_summary, epoch)
		if epoch % 50 == 0:
			print(curr_loss)

writer.close()
print("\nSlope: {}, Bias: {}".format(slope, bias))
print("W: {}, B: {}, Loss: {}\n".format(curr_w[0][0], curr_b[0][0], curr_loss))

plt.scatter(sample_x, sample_y, marker = "x")
plt.scatter(sample_x, y_pred, c = "red", marker = "o")
plt.gcf().canvas.set_window_title("LR Tensorflow")
plt.show()