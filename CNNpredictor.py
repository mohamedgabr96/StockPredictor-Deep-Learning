import requests
import io
import time
import pandas as pd
import numpy as np
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, GlobalAveragePooling1D, Flatten
import matplotlib.pyplot as plt
import random
from keras import backend as K

K.tensorflow_backend._get_available_gpus()


# set range for hyperparameters
epoch_range = [40,200]
decay_range = [.9, .99]
lr_range = [0.0000001, 0.001]
momentum_range = [0.8, 0.95]
feature_size_range = [1, 500]

# additional hyperparameters for CNN
strides_range = [1,5]
filters_range = [5,50] # output space, number of output filters in the convolution
kernel_size_range = [3,20] # filter size

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'
print('collecting data...')
data = requests.get(url)

df = pd.read_csv(io.StringIO(data.text),index_col=None)
#df = pd.read_csv('data.csv')

# checking to see the data was collected
print(df.head())

# save csv
df.to_csv('data.csv')

# pre_process data; only need daily_adjusted
data_raw = df.loc[::-1,'adjusted_close'].values

# normalize data
data_norm = data_raw/data_raw[0] - 1


def scrapdata():
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'
    print('collecting data...')
    data = requests.get(url)

    df = pd.read_csv(io.StringIO(data.text), index_col=None)
    # df = pd.read_csv('data.csv')

    # checking to see the data was collected
    print(df.head())

    # save csv
    df.to_csv('data.csv')

    # pre_process data; only need daily_adjusted
    data_raw = df.loc[::-1, 'adjusted_close'].values

    # normalize data
    data_norm = data_raw / data_raw[0] - 1
    return data_norm



def create_dataset(data, feature_size = 1):
    X, Y = [],[]
    for i in range(data.size - feature_size - 1):
        a = data[i:(i+feature_size)]
        X.append(a)
        Y.append(data[i+feature_size])
    return np.array(X),np.array(Y)


def train_CNN(filters_, kernel_size_, strides_, feature_size, lr, decay, momentum, X_train, Y_train, epoch):
    # create 2-layer CNN
    model = Sequential()
    model.add(Conv1D(
        filters=filters_[0], kernel_size=kernel_size_[0], strides=strides_[0],
        input_shape=(feature_size, 1), kernel_initializer='uniform',
        activation='relu'))

    # model.add(Flatten())

    model.add(Conv1D(
        strides=strides_[1],
        filters=filters_[1],
        kernel_size=kernel_size_[1]))

    model.add(Flatten())

    model.add(Dense(1, activation='relu'))

    start = time.time()

    ##### try different optimizers and loss fxn ###### below
    sgd = kr.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    rms = kr.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='mse', optimizer=sgd)
    print('compilation time : ', time.time() - start)

    # train model ####try different batchsize, epoch, justify why validation split=0.4
    history = model.fit(
        X_train,
        Y_train,
        batch_size=240,
        epochs=epoch,
        validation_split=0.4,
        verbose=0)
    return model, history



def CNN( epoch, decay, lr, momentum, feature_size, strides_, filters_, kernel_size_):
    # separate training and test data
    # feature_size = feature_size ######
    data_norm = scrapdata()
    train_size = round(data_norm.size* .6)
    X_train, Y_train = create_dataset(data_norm[0:train_size], feature_size)
    X_test, Y_test = create_dataset(data_norm[train_size::], feature_size)

    # reshape data into 3D LSTM input [samples, timesteps, features]
    # see https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    X_train = np.reshape(X_train, (X_train.shape[0], feature_size, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], feature_size, 1))


    model, history = train_CNN(filters_, kernel_size_, strides_, feature_size, lr, decay, momentum, X_train, Y_test, epoch)
    # create 2-layer CNN

    score = model.evaluate(X_test, Y_test, batch_size=240)
    return score

# CNN randomization, requires a few additional hyperparameters
def try_random_CNN(trials):
    results = []
    for i in range(0, trials):
        print("Trial "+str(i)+"\n")
        epoch = int(random.uniform(epoch_range[0], epoch_range[1]))
        print("Number of Epochs: " + str(epoch) + "\n")
        decay = random.uniform(decay_range[0], decay_range[1])
        lr = random.uniform(lr_range[0], lr_range[1])
        momentum = random.uniform(momentum_range[0], momentum_range[1])
        feature_size = int(random.uniform(feature_size_range[0], feature_size_range[1]))
        print("Feature Size: " + str(feature_size) + "\n")
        strides_ = []
        strides_.append(int(random.uniform(strides_range[0],strides_range[1])))
        strides_.append(int(random.uniform(strides_range[0],strides_range[1])))
        filters_ = []
        filters_.append(int(random.uniform(filters_range[0],filters_range[1])))
        filters_.append(int(random.uniform(filters_range[0],filters_range[1])))
        kernel_size_ = []
        kernel_size_.append(int(random.uniform(kernel_size_range[0],kernel_size_range[1])))
        kernel_size_.append(int(random.uniform(kernel_size_range[0],kernel_size_range[1])))
        score = CNN(data_norm, epoch, decay, lr, momentum, feature_size, strides_,filters_,kernel_size_)
        print("The Score is: " + str(score) + "\n")
        results.append([epoch, decay, lr, momentum, feature_size, strides_, filters_, kernel_size_, score])
    return results



