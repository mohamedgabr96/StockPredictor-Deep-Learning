import requests
import io
import time
import pandas as pd
import numpy as np
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import SimpleRNN, RNN, LeakyReLU
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random


epoch_range = [80 , 160]
decay_range = [.9, .99]
decay_range2 = [0.05 , 0.15]
lr_range = [0.0003, 0.0008]
momentum_range = [0.8, 0.95]
feature_size_range = [330, 380]
rho_range = [.8 , .9]

scale_range = [-1, 1]


def scrapdata():
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'
    print('collecting data...')
    data = requests.get(url)

    df = pd.read_csv(io.StringIO(data.text),index_col=None)
    # df = pd.read_csv('data.csv')

    # checking to see the data was collected
    print(df.head())

    # save csv
    df.to_csv('data.csv')

    # pre_process data; only need daily_adjusted
    data_raw = df.loc[::-1,'adjusted_close'].values

    #Scale and normalize data
    data_norm = data_raw/data_raw[0] - 1

    return data_norm


def difference_data(data_set):
    difference=[]
    for i in range(1, len(data_set)):
        difference.append(data_set[i] - data_set[i-1])
    return difference


def create_dataset(data, feature_size = 1):
    X, Y = [],[]
    for i in range(data.size - feature_size - 1):
        a = data[i:(i+feature_size)]
        X.append(a)
        Y.append(data[i+feature_size])
    return np.array(X),np.array(Y)


def rescale(data_set):
    scaling = MinMaxScaler(feature_range=(scale_range(0),scale_range(1)))
    rescaled = scaling.fit(data_set)
    return rescaled


def train(x_train, y_train, lr, decay, momentum, rho, feature_size, epoch):
    model = Sequential()
    model.add(LSTM(
        input_shape=(1, feature_size),
        units=500,  # output space #########
        activation='tanh',
        return_sequences=False))
    model.add(Dense(units=1))
    start = time.time()
    sgd = kr.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    rms = kr.optimizers.rmsprop(lr=lr, rho=rho, epsilon=None, decay=decay)
    model.compile(loss='mse', optimizer=sgd)
    print('compilation time : ', time.time() - start)
    history = model.fit(
        x_train,
        y_train,
        batch_size=240,
        epochs=epoch,
        validation_split=0.4)
    return model, history

def LSTM_1(lr, decay, momentum, rho , feature_size, epoch):
    # scrap the data
    raw_data = scrapdata()
    # difference the data to remove increasing trends
    # split data into training and test
    train_size = round(raw_data.size * .6)
    X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
    X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))
    return train(X_train, Y_train, lr, decay, momentum, rho , feature_size, epoch)



def no_epochs_find(trials):
    histories = []
    epochs = 1000
    feature_size = 200
    momentum = 0.9
    rho = .85
    lr = 0.0003
    decay = .9
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    for i in range(0, trials):
        model, history = LSTM_1(lr, decay, momentum, rho, feature_size, epochs)
        histories.append(history)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
    plt.savefig('loss.png')




no_epochs_find(10)