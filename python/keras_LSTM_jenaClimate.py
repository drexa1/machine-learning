import os
import logging
import yaml

import numpy as np

import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.models import model_from_yaml

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def main():

	float_data = get_data('jena_climate_2009_2016.csv')

	lookback = 1440
	step = 6
	delay = 144
	batch_size = 128

	data = normalize(float_data[:200000])

	train_gen = generator(data, lookback=lookback, delay=delay, min_index=None, max_index=200000, step=step, batch_size=batch_size)
	val_gen = generator(data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step, batch_size=batch_size)
	test_gen = generator(data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step, batch_size=batch_size)

	try:
		model = load_model("jenaClimate_model.yml")
		logger.info("Model found.")
		logger.info(model.summary)
	except FileNotFoundError:
		logger.info("Model not found.")
		logger.info("Creating model...")
		model = create_model(data)
		logger.info("Saving model...")
		save_model(model, "jenaClimate_model.yml")
	finally:
		logger.info("Compiling model...")
		model.compile(optimizer=RMSprop(), loss='mae')

	try:
		model.load_weights('jenaClimate_checkpoints.h5')
		logger.info("Weights found.")
	except OSError:
		logger.info("Weights not found.")

	val_steps = (300000 - 200001 - lookback)
	history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps, callbacks=model_callbacks())

	model.save_weights('jenaClimate_weights.h5')

	epochs = range(1, len(loss) + 1)
	loss = model_history.history['loss']
	val_loss = model_history.history['val_loss']

	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.show()

def load_model(model_file):
	with open(model_file, 'r') as f:
		yaml_string = yaml.load(f)
	model = model_from_yaml(yaml_string)
	return model

def save_model(model, model_file):
	yaml_string = model.to_yaml()
	with open(model_file, 'w') as outfile:
		yaml.dump(yaml_string, outfile, fileEncoding = 'UTF-8')

def create_model(data):
	model = Sequential()
	model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, data.shape[-1])))
	model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
	model.add(layers.Dense(1))
	return model

def model_callbacks():
	keras.callbacks.ModelCheckpoint(filepath='jenaClimate_checkpoints.h5', monitor='val_loss', save_best_only=True,)
	earlystop_callback = keras.callbacks.EarlyStopping(monitor='acc', patience=1)
	plateau_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensorboard_jenaClimate')
	return [earlystop_callback, plateau_callback, tensorboard_callback]

def get_data(csv_file):
	with open(csv_file, 'r') as f:
		data = f.read()
	f.close()

	header_and_lines = data.split('\n')
	header = header_and_lines[0].split(',')
	lines = header_and_lines[1:]

	float_data = np.zeros((len(lines), len(header) - 1))
	for i, line in enumerate(lines):
		values = [float(x) for x in line.split(',')[1:]]
		float_data[i, :] = values

	return float_data

def normalize(data):
	mean = data.mean(axis=0)
	data -= mean
	std = data.std(axis=0)
	data /= std
	return data

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
	if min_index is None:
		min_index = 0
	if max_index is None:
		max_index = len(data) - delay - 1
	index = min_index + lookback
	while True:
		if shuffle:
			rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
		else:
			if index + batch_size >= max_index:
				index = min_index + lookback
			rows = np.arange(index, min(index + batch_size, max_index))
			index += len(rows)

		x = np.zeros((len(rows), lookback//step, data.shape[-1]))
		y = np.zeros((len(rows),))
		for i, row in enumerate(rows):
			indices = range(rows[i] - lookback, rows[i], step)
			x[i] = data[indices]
			y[i] = data[rows[i] + delay][1]

		yield x, y

if __name__ == "__main__":
	# execute only if run as a script
	main()
