from keras.datasets import boston_housing

from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalize
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# This is called per fold
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1)) # don't force activation for continuous scalar
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model

# CV
k = 11
num_val_samples = len(train_data) // k
num_epochs = 500

all_mae_histories = []
for i in range(k):
	print('Fold #{}'.format(i))
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
	partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
	partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
	model = build_model()
	history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


print('\nRetraining fresh model:')
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('\nMean abs error: {}($)'.format(test_mae_score))
