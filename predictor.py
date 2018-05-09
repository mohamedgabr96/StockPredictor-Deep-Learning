import requests
import io
import pandas as pd

# fetch csv file using Alpha Vantage api
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=INX&outputsize=full&datatype=csv&apikey=9B9U2G2YHKS9ME8T'

print('collecting data...')
data = requests.get(url)
df = pd.read_csv(io.StringIO(data.text))

# checking to see the data was collected
print(df.head())

# normalize data


# save csv
df.to_csv('data.csv')


