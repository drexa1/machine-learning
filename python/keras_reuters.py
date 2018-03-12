from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

import matplotlib.pyplot as plt
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10_000)

def embToEnglish(str):
	word_index = reuters.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in str])
	return decoded_review

def vectorize_sequences(sequences, dimension=10_000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
y_val = one_hot_train_labels[:1000]

partial_x_train = x_train[1000:]
partial_y_train = one_hot_train_labels[1000:]

model_hist  = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']
acc = model_hist.history['acc']
val_acc = model_hist.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.figure(1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

print('\nRetraining with full set:')
model.fit(x_train, one_hot_train_labels, epochs=10, batch_size=512)
results = model.evaluate(x_test, one_hot_test_labels)
print('\nLoss: {}, accuracy: {}'.format(results[0], results[1]))

y_pred = model.predict(x_test)
print

print('\nTest data example:{}\n'.format(embToEnglish(test_data[0])))
max_prob = max(y_pred[0])
topic = y_pred[0].tolist().index(max_prob)
print('Topic: {} ({} confidence)\n'.format(topic, max_prob))
