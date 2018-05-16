import requests
import io
import pandas as pd

# fetch csv file using Alpha Vantage api
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'

print('collecting data...')
#data = requests.get(url)
#df = pd.read_csv(io.StringIO(data.text))
df = pd.read_csv('data.csv')

# checking to see the data was collected
print(df.head())

# save csv
df.to_csv('data.csv')

# preprocess data; only need daily_adjusted
x_raw = df.loc[::-1,'adjusted_close'].values
x_raw = pd.Series(x_raw)

# normalize data
x_norm = x_raw/x_raw[0] - 1

# convert raw series data into x and y dataset. y looks is 1 day ahead by default
# representation with 1 input feature by default when using a stateful LSTM
# alternatively, we can use multiple days. see https://machinelearningmastery.com/use-features-lstm-networks-time-series-forecasting/
# ctrl+F for "Experiments with Features"
def create_dataset(data, days_ahead = 1):
    data = data.values
    X, Y = [],[]
    for i in range(data.size - days_ahead):
        X.append(data[i])
        Y.append(data[i+days_ahead])
    return pd.Series(X),pd.Series(Y)


# separate training (~2016) and validation (2017~2018) data [0:4276]
X_train, Y_train = create_dataset(x_norm[0:4276])
X_val, Y_val = create_dataset(x_norm[4277::])

# create LSTM network (modify/continue from here)
model = Sequential()
model.add(LSTM(
    input_dim = 1,
    output_dim = 50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add
