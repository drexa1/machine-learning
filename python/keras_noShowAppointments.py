import os
import pandas as pd
import numpy as np
import datetime as dt
import unicodedata

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data
dataset_x = pd.read_csv('../datasets/noshowappointments.csv', usecols=[4, 7, 8, 9, 10, 11, 12])
dataset_y = pd.read_csv('../datasets/noshowappointments.csv', usecols=[13])

def dayOfTheWeek(appointmentDay):
	return dt.datetime.strptime(appointmentDay, '%Y-%m-%dT%H:%M:%SZ').date().weekday()

def cleanText(text):
	return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore')

def noShowToInt(text):
	return [0, 1][text == 'Yes']

# Transform to numerical
dataset_x.AppointmentDay = dataset_x.AppointmentDay.apply(dayOfTheWeek)
dataset_y['No-show'] = dataset_y['No-show'].apply(noShowToInt)

# Model
model = Sequential()
model.add(Dense(dataset_x.shape[1] + 1, input_shape = (dataset_x.shape[1], ), activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dense(dataset_x.shape[1] + 1, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dense(dataset_x.shape[1] + 1, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dense(1, activation = 'relu', kernel_initializer = 'uniform'))
model.summary()

log_dir = r'C:\Users\drexa\git\machine-learning\python\noShowAppointments'
tbCallback = TensorBoard(log_dir = log_dir, write_graph = True)

# Train the model
print('Training...')
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(dataset_x.values, dataset_y.values, epochs = 1001, batch_size = 50,  verbose = 0, validation_split = 0.3, callbacks = [tbCallback])
