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
    #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'
    print('collecting data...')
    #data = requests.get(url)

    #df = pd.read_csv(io.StringIO(data.text),index_col=None)
    df = pd.read_csv('data.csv')

    # checking to see the data was collected
    print(df.head())

    # save csv
   # df.to_csv('data.csv')

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


def train(x_train, y_train, lr, decay, momentum, rho, feature_size, epoch,neurons, batch_size):
    model = Sequential()
    model.add(LSTM(
        input_shape=(1, feature_size),
        units=neurons,  # output space #########
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
        batch_size=batch_size,
        epochs=epoch,
        validation_split=0.4)
    return model, history

def LSTM_1(lr, decay, momentum, rho , feature_size, epoch, neurons, batch_size, X_train, Y_train, X_test,Y_test):
    # scrap the data
    model, history = train(X_train, Y_train, lr, decay, momentum, rho, feature_size, epoch,neurons, batch_size)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    return model, history, score



def load_data(feature_size):
    raw_data = scrapdata()
    # difference the data to remove increasing trends
    # split data into training and test
    train_size = round(raw_data.size * .6)
    X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
    X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))
    return X_train, Y_train, X_test,Y_test


def no_epochs_find(trials):
    histories = []
    epochs = 1000
    feature_size = int(random.uniform(1, 500))
    momentum = 0.9
    rho = .85
    lr = 0.0003
    decay = .9
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    X_train, Y_train, X_test, Y_test = load_data(feature_size)
    for i in range(0, trials):
        model, history, score = LSTM_1(lr, decay, momentum, rho, feature_size, epochs, 240)
        histories.append(history)

        feature_size = int(random.uniform(1, 500))
    plt.savefig('loss.png')

#after the test with epochs, 700 was chosen
def no_feature_find(trials):
    histories = []
    epochs = 700
    feature_size = int(random.uniform(1, 500))
    momentum = 0.9
    rho = .85
    lr = 0.0003
    decay = .9
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    raw_data = scrapdata()
    for i in range(0, trials):
        print("ITERATION NUMBER " + str(i))
        train_size = round(raw_data.size * .6)
        X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
        X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))
        model, history, score = LSTM_1(lr, decay, momentum, rho, feature_size, epochs, 240, X_train, Y_train, X_test, Y_test)
        histories.append([feature_size, score])
        feature_size = int(random.uniform(1, 500))
    df = pd.DataFrame(data=histories, columns=['Feature Size', 'Score'])
    df.to_csv("Feature_Size_Loss.csv")



def no_neurons_find(trials):
    histories = []
    epochs = 700
    feature_size = 390
    batch_size = int(random.uniform(1, 300))
    neurons = int(random.uniform(1,1000))
    momentum = 0.9
    rho = .85
    lr = 0.0003
    decay = .9
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    raw_data = scrapdata()
    for i in range(0, trials):
        print("ITERATION NUMBER " + str(i))
        train_size = round(raw_data.size * .6)
        X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
        X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))
        model, history, score = LSTM_1(lr, decay, momentum, rho, feature_size, epochs, neurons, batch_size,  X_train, Y_train, X_test, Y_test)
        histories.append([batch_size, score])
        batch_size = int(random.uniform(1, 300))
    df = pd.DataFrame(data=histories, columns=['batch size', 'Score'])
    df.to_csv("Batch_Size_Loss.csv")


def hyperparameter_find(trials):
    histories = []
    epochs = 700
    feature_size = 390
    batch_size = 180
    neurons = 275
    momentum = 0.9
    rho = .85
    lr = 0.0003
    decay = .9
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    raw_data = scrapdata()
    for i in range(0, trials):
        momentum = int(random.uniform(momentum_range[0],momentum_range[1]))
        rho = int(random.uniform(rho_range[0],rho_range[1]))
        lr = int(random.uniform(lr_range[0],lr_range[1]))
        decay = int(random.uniform(decay_range[0],decay_range[1]))
        print("ITERATION NUMBER " + str(i))
        train_size = round(raw_data.size * .6)
        X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
        X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))
        model, history, score = LSTM_1(lr, decay, momentum, rho, feature_size, epochs, neurons, batch_size,  X_train, Y_train, X_test, Y_test)
        histories.append([momentum, rho, lr, decay, score])
    df = pd.DataFrame(data=histories, columns=['momentum', 'rho', 'lr', 'decay', 'score'])
    df.to_csv("SGD_LSTM_Loss.csv")



def no_batch_find(trials):
    histories = []
    epochs = 700
    feature_size = 390
    batch_size = 30
    neurons = 25
    momentum = 0.9
    rho = .85
    lr = 0.0003
    decay = .9
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    raw_data = scrapdata()
    for i in range(0, trials):
        print("ITERATION NUMBER " + str(i))
        train_size = round(raw_data.size * .6)
        X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
        X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))

        model, history, score = LSTM_1(lr, decay, momentum, rho, feature_size, epochs, neurons, batch_size,  X_train, Y_train, X_test, Y_test)
        histories.append([batch_size, score])
        batch_size += 30
    df = pd.DataFrame(data=histories, columns=['batch size', 'Score'])
    df.to_csv("Batch_Size_Loss.csv")


def predict():
    histories = []
    epochs = 700
    feature_size = 375
    batch_size = 180
    neurons = 1000
    momentum = 0.8418428759613501
    rho = 0.9194713153044991
    lr = 0.0007888826881850667
    decay = 0.9738939197576053
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    raw_data = scrapdata()
    train_size = round(raw_data.size * .6)
    X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
    X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, feature_size))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, feature_size))
    model, history, score = LSTM_1(lr, decay, momentum, rho, feature_size, epochs, neurons, batch_size, X_train,
                                   Y_train, X_test, Y_test)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'accuracy'], loc='upper right')
    plt.savefig("loss")
    print("The score is " + str(score))
    plt.clf()

    # get predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # plot normalized predictions
    plt.plot(test_predictions)
    plt.plot(Y_test)
    plt.title('Test Predictions vs Actual, Normalized')
    plt.ylabel('score')
    plt.xlabel('sequence(t)')
    plt.legend(['predictions','actual'],loc='upper right')
    plt.savefig("prediction_normalized.png")
    plt.clf()
    # de-normalize the predictions
    train_predictions = (train_predictions+1) * raw_data[0]
    test_predictions = (test_predictions+1) * raw_data[0]
    train_actual = (Y_train+1) * raw_data[0]
    test_actual = (Y_test+1) * raw_data[0]

    # plot the denormalized predictions
    plt.plot(test_predictions)
    plt.plot(test_actual)
    plt.title('Test Predictions vs Actual, Denormalized')
    plt.ylabel('score')
    plt.xlabel('sequence(t)')
    plt.legend(['predictions','actual'],loc='upper right')
    plt.savefig("prediction_denormalized.png")
    plt.clf()
    return 0

#no_epochs_find(10)
#no_feature_find(100)

predict()