import os
import threading
import numpy as np
from random import shuffle

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# From 1 to 2^20 (1048575) in binary sequence of 20 digits
sequences = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(sequences)
sequence_map = [map(int,i) for i in sequences]
tensor = []
for i in sequence_map:
    temp_list = []
    for j in i:
        temp_list.append([j])
    tensor.append(np.array(temp_list))
train_input = tensor

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * 21)
    temp_list[count] = 1
    train_output.append(temp_list)

HOLDOUT = 10_000
test_input = train_input[HOLDOUT:]
test_output = train_output[HOLDOUT:]
train_input = train_input[:HOLDOUT]
train_output = train_output[:HOLDOUT]

# Placeholders [Batch Size, Sequence Length, Input Dimension]
X = tf.placeholder(tf.float32, [None, 20, 1])
Y = tf.placeholder(tf.float32, [None, 21])

# Tensorflow variables
NUM_HIDDEN = 24
W = tf.Variable(tf.truncated_normal([NUM_HIDDEN, int(Y.get_shape()[1])]))
B = tf.Variable(tf.constant(0.1, shape = [Y.get_shape()[1]]))

# Model
cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, state_is_tuple = True)
[val, state] = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

prediction = tf.nn.softmax(tf.matmul(last, W) + B)

# Loss
cross_entropy = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(prediction, 1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(Y, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

LOG_DIR = r'C:\Users\drexa\git\machine-learning\python\LSTM'
batch_size = 1000
epochs = 5000
no_of_batches = int(len(train_input) / batch_size)

saver = tf.train.Saver()

# Initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Restoring model from disk...')
    saver.restore(sess, os.path.join(LOG_DIR, "LSTM.ckpt"))

    tf.summary.scalar('LSTM-XE', cross_entropy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, graph = tf.get_default_graph())

    # Train
    print('Training...')
    for step in range(epochs):
        portion = 0
        for batch in range(no_of_batches):
            [training_x, training_y] = train_input[portion:portion + batch_size], train_output[portion:portion + batch_size]
            portion += batch_size
            [_, curr_xe, curr_summary] = sess.run([train_step, cross_entropy, merged_summary], feed_dict = {X: training_x, Y: training_y})
            writer.add_summary(curr_summary, step)
        # TODO: patience/tolerance early termination
        print("Epoch: {} Cross entropy: {}".format(step, curr_xe))
        saver.save(sess, os.path.join(LOG_DIR, "LSTM.ckpt"))

    incorrect = sess.run(error, feed_dict = {X: test_input, Y: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(step + 1, 100 * incorrect))
    writer.close()

    print("Prediction for: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]")
    testing_y = sess.run(prediction, feed_dict = {X: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
    print(testing_y)

sess.close()
print('Done')
