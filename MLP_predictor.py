import requests
import io
import time
import pandas as pd
import numpy as np
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, GlobalAveragePooling1D, Flatten
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

# set range for hyperparameters
epoch_range = [1,400] #700
decay_range = [.9, .99]
lr_range = [0.0000001, 0.001]
momentum_range = [0.8, 0.95]
batch_size_range = [155,240]
feature_size_range = [135,321]
neuron_range = [5, 100]


def scrapdata():
    #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'
    print('collecting data...')
    #data = requests.get(url)

    #df = pd.read_csv(io.StringIO(data.text),index_col=None)
    df = pd.read_csv('data.csv')

    # checking to see the data was collected
    print(df.head())

    # save csv
    df.to_csv('data.csv')

    # pre_process data; only need daily_adjusted
    data_raw = df.loc[::-1,'adjusted_close'].values

    #Scale and normalize data
    data_norm = data_raw/data_raw[0] - 1

    return data_norm


def create_dataset(data, feature_size = 1):
    X, Y = [],[]
    for i in range(data.size - feature_size - 1):
        a = data[i:(i+feature_size)]
        X.append(a)
        Y.append(data[i+feature_size])
    return np.array(X),np.array(Y)


def train(X_train,Y_train,X_test,Y_test,lr,decay,momentum,feature_size,epoch,batch_size=240):
    # create Multilayered Perceptron network
    model = Sequential()
    model.add(Dense(25,input_dim=feature_size, activation='relu'))
    model.add(Dropout(0.1)) # dropout regularization
    model.add(Dense(25,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(1))

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
        batch_size=batch_size,
        epochs=epoch,
        validation_split=0.4)

    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    return model,history, score


def Multi_Perceptron(X_train,Y_train,X_test,Y_test,lr, decay, momentum, feature_size, epoch, batch_size):
    '''
    # scrap the data
    raw_data = scrapdata()
    # difference the data to remove increasing trends
    # split data into training and test
    train_size = round(raw_data.size * .6)
    X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
    X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
    X_train = np.reshape(X_train, (X_train.shape[0],  feature_size))
    X_test = np.reshape(X_test, (X_test.shape[0],  feature_size))
    '''
    return train(X_train, Y_train,X_test,Y_test, lr, decay, momentum, feature_size, epoch, batch_size)


def no_epochs_find(trials):
    histories = []
    epochs = 1000
    feature_size_n = [30,390]
    momentum_n = [.83,.89]
    lr_n = [.0002,.001]
    decay_n = [.91,.99]
    batch_size=240
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    for i in range(0, trials):
        lr = random.uniform(lr_n[0], lr_n[1])
        decay = random.uniform(decay_n[0], decay_n[1])
        momentum = random.uniform(momentum_n[0], momentum_n[1])
        feature_size = int(random.uniform(feature_size_n[0], feature_size_n[1]))

        model, history, score = Multi_Perceptron(lr, decay, momentum, feature_size, epochs, batch_size)
        histories.append(history)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
    plt.savefig('MLP_loss.png')


def no_features_find(trials):
    histories = []
    epochs = 700
    feature_size_n = [30,390]
    momentum_n = .083572
    lr = .000971
    decay = .988072
    batch_size=240
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    for i in range(0, trials):
        print('trial '+str(i)+'\n')
        feature_size = int(random.uniform(feature_size_n[0], feature_size_n[1]))
        #hard set to best trial results, constant values
        model, history, score = Multi_Perceptron(lr, decay, momentum, feature_size, epochs, batch_size)
        histories.append([feature_size, score])
    df = pd.DataFrame(data=histories, columns=['Feature Size', 'Score'])
    df.to_csv("MLP_Feature_Size_Loss.csv")


def no_batch_find(trials):
    histories = []
    batch_size = int(random.uniform(1, 500))
    epochs = 700
    feature_size = 390
    momentum = .083572
    lr = .000971
    decay = .988072
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    for i in range(0, trials):
        print("ITERATION NUMBER " + str(i))
        model, history, score = Multi_Perceptron(lr, decay, momentum, feature_size, epochs, batch_size)
        histories.append([batch_size, score])
        batch_size = int(random.uniform(1, 380))
    df = pd.DataFrame(data=histories, columns=['batch size', 'Score'])
    df.to_csv("MLP_Batch_Size_Loss.csv")


def final_find(trials):
    # scrap the data
    raw_data = scrapdata()
    # difference the data to remove increasing trends
    # split data into training and test
    train_size = round(raw_data.size * .6)

    histories = []
    epochs = 700 
    feature_size_n = [135,321]
    momentum_n = [.83,.89]
    lr_n = [.0002,.001]
    decay_n = [.91,.99]
    batch_size_n=[155,240]
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'black'])
    for i in range(0, trials):
        print("ITERATION NUMBER " + str(i))

        lr = random.uniform(lr_n[0], lr_n[1])
        decay = random.uniform(decay_n[0], decay_n[1])
        momentum = random.uniform(momentum_n[0], momentum_n[1])
        feature_size = int(random.uniform(feature_size_n[0], feature_size_n[1]))
        batch_size = int(random.uniform(batch_size_n[0],batch_size_n[1]))

        X_train, Y_train = create_dataset(raw_data[0:train_size], feature_size)
        X_test, Y_test = create_dataset(raw_data[train_size::], feature_size)
        X_train = np.reshape(X_train, (X_train.shape[0],  feature_size))
        X_test = np.reshape(X_test, (X_test.shape[0],  feature_size))

        model, history, score = Multi_Perceptron(X_train,Y_train,X_test,Y_test,lr, decay, momentum, feature_size, epochs, batch_size)
        histories.append([lr,decay,momentum,feature_size,epochs,batch_size,score])
    plt.savefig('MLP_loss.png')
    df = pd.DataFrame(data=histories, columns=['lr', 'decay','momentum','feature_size','epochs','batch_size','score'])
    df.to_csv("MLP_fine_search.csv")
    

final_find(50)
