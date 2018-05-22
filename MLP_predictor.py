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
epoch_range = [1,400]
decay_range = [.9, .99]
lr_range = [0.0000001, 0.001]
momentum_range = [0.8, 0.95]
feature_size_range = [1, 500]
neuron_range = [5, 100]

# fetch csv file using Alpha Vantage api
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'
print('collecting data...')
data = requests.get(url)

df = pd.read_csv(io.StringIO(data.text),index_col=None)
#df = pd.read_csv('data.csv')

# checking to see the data was collected
print(df.head())

# save csv
#df.to_csv('data.csv')

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

# random hyperparameter trial runs for a network
def try_random(trials):
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
        score = Multi_Perceptron(data_norm, epoch, decay, lr, momentum, feature_size)
        print("The Score is: " + str(score) + "\n")
        results.append([epoch, decay, lr, momentum, feature_size, score])
    return results


def Multi_Perceptron(data_norm, epoch, decay, lr, momentum, feature_size):
    # separate training and test data
    # feature_size = feature_size ######
    train_size = round(data_norm.size* .6)
    X_train, Y_train = create_dataset(data_norm[0:train_size], feature_size)
    X_test, Y_test = create_dataset(data_norm[train_size::], feature_size)

    # reshape data into 3D LSTM input [samples, timesteps, features]
    # see https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    X_train = np.reshape(X_train, (X_train.shape[0], feature_size))
    X_test = np.reshape(X_test, (X_test.shape[0], feature_size))


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
        batch_size=240,
        epochs=epoch,
        validation_split=0.4,
        verbose=0)

    score = model.evaluate(X_test, Y_test, batch_size=240)
    '''
    # predict and plot

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'accuracy'], loc='upper right')
    plt.savefig('MLP_loss.png',bbox_inches='tight')
    print("The score is " + str(score))

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
    plt.savefig('MLP_predictions_norm.png',bbox_inches='tight')
    
    # de-normalize the predictions
    train_predictions = (train_predictions+1) * data_raw[0]
    test_predictions = (test_predictions+1) * data_raw[0]
    train_actual = (Y_train+1) * data_raw[0]
    test_actual = (Y_test+1) * data_raw[0]
    
    # plot the denormalized predictions
    plt.plot(test_predictions)
    plt.plot(test_actual)
    plt.title('Test Predictions vs Actual, Denormalized')
    plt.ylabel('score')
    plt.xlabel('sequence(t)')
    plt.legend(['predictions','actual'],loc='upper right')
    plt.savefig('MLP_predictions_denorm.png',bbox_inches='tight')
    '''
    return score


results = try_random(200)
results_df = pd.DataFrame(data= results, columns=['Epoch','Decay','Learning Rate','Momentum','Feature Size','Score'])
results_df.to_csv("MLP_sgd_results.csv")
