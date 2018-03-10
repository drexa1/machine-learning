from keras.datasets import imdb

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10_000)

def vectorize_sequences(sequences, dimension=10_000):
	# Creates an all-zero matrix of shape (len(sequences), dimension)
	results = np.zeros((len(sequences), dimension))
	# Sets specific indices of results[i] to 1s
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results

def embToEnglish(str):
	word_index = imdb.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in str])
	return decoded_review

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10_000, ))) # zeroes out the negatives
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) # squashes arbitrary values into [0, 1]

print('\nCompiling model:')
model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
			  loss=losses.binary_crossentropy, 
			  metrics=[metrics.binary_accuracy])

x_val = x_train[ :10000]
partial_x_train = x_train[10000: ]

y_val = y_train[ :10000]
partial_y_train = y_train[10000: ]

print('\nTraining model for evaluation:')
m_hist = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = m_hist.history
print(history_dict.keys())
loss = history_dict['loss']
acc = history_dict['binary_accuracy']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_binary_accuracy']

epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

# Retraining from scratch
print('\nRetraining with full set:')
model.fit(x_train, y_train, epochs=5, batch_size=512)
results = model.evaluate(x_test, y_test)
print('\nLoss: {}, accuracy: {}'.format(results[0], results[1]))

# Predict new data
y_pred = model.predict(x_test)
print('Predictions confidence:')
print(y_pred)

print('\nTest data example:\n{}'.format(embToEnglish(test_data[0])))
print('\nSentiment: {}\n'.format(y_pred[0]))
