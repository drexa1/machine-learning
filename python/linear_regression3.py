import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import optimizers, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# 250 points of 1 dimension
n = 250
d = 1
sample_x = np.random.rand(n, d)
noise = np.random.rand(n, d)
sample_x += noise

slope = 12
bias = 10
sample_y = slope * sample_x + bias

noise = np.random.rand(n, d)
sample_x += noise

# Create a layer to take an input
input_layer = Input(shape=np.array([1]))

# Compute W * X + B
dense = Dense(np.array([1]), activation = 'linear')
output = dense(input_layer)

# Model
model = Model(inputs = [input_layer], outputs = [output])
model.summary()

# Tensorboard directory
log_dir = r"C:\Users\drexa\git\machine-learning\python\linear3"
lr = 0.1
epochs = 5001

# Optimizer to minimize loss
sgd = optimizers.sgd(lr = lr)
model.compile(optimizer = sgd, loss = "mean_squared_error")

tensorboard = TensorBoard(log_dir = log_dir)

# Train the model
verbose = 0
if verbose == 0: print("Training...")
fit = model.fit(x = sample_x, y = sample_y, epochs = epochs, verbose = verbose, callbacks = [tensorboard])

# Predict y
y_pred = model.predict(sample_x)
y_pred_loss = model.evaluate(sample_x, sample_y)

print("\nSlope: {}, Bias: {}".format(slope, bias))
print("W: {}, B: {}, Loss: {}\n".format(dense.get_weights()[0][0][0], dense.get_weights()[1][0], y_pred_loss))

plt.scatter(sample_x, sample_y, marker = "x")
plt.scatter(sample_x, y_pred, c = "red", marker = "o")
plt.show()