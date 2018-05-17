import requests
import io
import time
import pandas as pd
import numpy as np # keras takes numpy arrays, not dataframe :(
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import matplotlib.pyplot as plt

# fetch csv file using Alpha Vantage api
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'

print('collecting data...')
data = requests.get(url)
df = pd.read_csv(io.StringIO(data.text))
# df = pd.read_csv('data.csv')

# checking to see the data was collected
print(df.head())

# save csv
df.to_csv('data.csv')

# pre_process data; only need daily_adjusted
data_raw = df.loc[::-1,'adjusted_close'].values

# normalize data
data_norm = data_raw/data_raw[0] - 1

# convert raw series data into x and y dataset. y = x(t+1)
# representation with 1 input feature by default when using a stateful LSTM
# alternatively, we can use multiple days. see https://machinelearningmastery.com/use-features-lstm-networks-time-series-forecasting/


def create_dataset(data, feature_size = 1):
    X, Y = [],[]
    for i in range(data.size - feature_size - 1):
        a = data[i:(i+feature_size)]
        X.append(a)
        Y.append(data[i+feature_size])
    return np.array(X),np.array(Y)


# separate training (~2016) and test (2017~2018) data [0:4276]
feature_size = 1
train_size = round(data_norm.size* .8)
X_train, Y_train = create_dataset(data_norm[0:train_size],feature_size)
X_test, Y_test = create_dataset(data_norm[train_size+1::],feature_size)

# reshape data into 3D LSTM input [samples, timesteps, features]
# see https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))

# create LSTM network (modify/continue from here)
model = Sequential()
model.add(LSTM(
    input_shape = (1,feature_size),
    units = 1000,  # output space
    return_sequences=True))

# model.add(Dropout(0.2))

model.add(LSTM(
    100,  # output space
    return_sequences=False))


# model.add(Dropout(0.2))

model.add(Dense(units=1))
model.add(Activation('linear'))

start = time.time()
sgd = kr.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)
print('compilation time : ', time.time() - start)

# train model
history = model.fit(
    X_train,
    Y_train,
    batch_size=100,
    epochs=15,
    validation_split=0.33)

# predict and plot
score = 1 - model.evaluate(X_test, Y_test, batch_size=100)
print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
print("The score is " + str(score))

# get predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# plot normalized predictions
plt.plot(test_predictions)
plt.plot(Y_test)
plt.legend(['predictions','actual'],loc='upper right')

# de-normalize the predictions
train_predictions = (train_predictions+1) * data_raw[0]
test_predictions = (test_predictions+1) * data_raw[0]
train_actual = (Y_train+1) * data_raw[0]
test_actual = (Y_test+1) * data_raw[0]

# plot the denormalized predictions
plt.plot(test_predictions)
plt.plot(test_actual)
plt.legend(['predictions','actual'],loc='upper right')
