# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:33:01 2020

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("BTC-USD_train.csv",date_parser="Date")
df_test = pd.read_csv("BTC-USD_test.csv",date_parser="Date")

df.dropna(inplace=True)
df_test.dropna(inplace=True)


train = df.iloc[:, 1:2].values
test = df_test.iloc[:, 1:2].values

inputs = test

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

train = sc.fit_transform(train)
inputs = sc.transform(inputs)

X_train = np.array(train[:-1]).reshape(-1,1,1)
y_train = train[1:]

inputs = np.reshape(inputs, (-1,1,1))


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

regressor.add(LSTM(4,input_shape=(None,1),activation="sigmoid"))


regressor.add(Dense(1))


regressor.compile(optimizer="adam",loss="mean_squared_error")

regressor.fit(X_train,y_train,batch_size=32,epochs=200)


predicted = regressor.predict(inputs)
predicted = sc.inverse_transform(predicted)

plt.plot(test, color = 'red', label = 'Real Bitcoin Price')
plt.plot(predicted, color = 'blue', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
