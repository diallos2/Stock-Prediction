"""
Programmer: Salimatou Diallo
Project: An Application that predicts the stock market based on 
company ticker received from user. 
Research: Researched on how to integrate API in Python on https://towardsdatascience.com/free-stock-data-for-python-using-yahoo-finance-api-9dafd96cad2e , 
Watched a similar Project Example on Youtube
Preparations: Set Aside 3hours-5hours of quiet work time

"""
#Import all the neeeded tools for our program

import pandas as pd
import numpy as np
import pandas_datareader as pdr #first pip install using "pip install pandas-datareader" if error occurs
import matplotlib.pyplot as plt
import seaborn
import yfinance as yf
import math
from sklearn.preprocessing import MinMaxScaler 
import streamlit as st

#import all necessary for building the model

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#First Step is to ask user for company ticker. 

company_ticker=input("Enter the Company Ticker you wish to predict stocks for: ")

#Second is to get the stocks information for that ticker and display it in the terminal

data=yf.Ticker(company_ticker)

#Third is to get the data for the past 11 years and display that in the terminal

historycal_data=data.history(period="11y")
print(historycal_data.info)

#Then we will drop the Date, Dividends, Stock Splits, and Adj Close

historycal_data.drop(['Dividends', 'Stock Splits'], axis = 1)
print(historycal_data.info) #check to make sure came out right

#Next, Create a chart that tracks trends occuring over periods of time for Close

plt.figure(figsize=(12, 6)) #creates the figure
plt.title("CLOSING PRICE HISTORY") #creates a title for out figure that will be at top
plt.plot(historycal_data['Close']) #Plots the Close dataset inside the historical data
plt.xlabel("Date", loc= "center") #labels our x axis with 'Date'
plt.ylabel("Closing Price $", loc = "center") #labels our y axis with 'Closing Price $'
plt.show() #displays the figure

#After this, we want to create a new data frame for just our close column

close_data=historycal_data.filter(["Close"]) #filters out only the closing data from the historical data
close_data_set=close_data.values #this creates a new array containing the close_data values
length_training_data = math.ceil( len(close_data_set) * 0.5) #this assigns 50% of the data to train the model 
length_training_data

#Here we want to standardize our data using the minmaxscaler technique 

scaler = MinMaxScaler(feature_range=(0,1)) #transforms our data to be between 0 and 1
scaler_data = scaler.fit_transform(close_data_set) #assigns scaler_data var with our scaled close_data_set
scaler_data

#Now we want to create our train model 

data_train = scaler_data[0:length_training_data, :] #extracts and indexes the first rows of length_training_data and all columns of length_training_data
train_x = [] #creates an empty x list
train_y = [] #creates an empty y list
prediction_days=210

for i in range(prediction_days, len(data_train)): #for every i value in range from 210 to the lenth of our data_train array
    train_x.append(data_train[i-prediction_days:i, 0]) #Starts at 0 and goes all the way to i while appending to our x
    train_y.append(data_train[i, 0]) #Starts from i while appending it to our y

#Next is to convert our train_x and train_y to np arrays

train_x, train_y = np.array(train_x), np.array(train_y)

#Here we should reshape the data

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

#Create our ML Model below

model = Sequential() #creates the Sequential model
model.add(LSTM(units=50, activation='relu', return_sequences= True, input_shape = (train_x.shape[1], 1))) #adds everything inside to the Sequential Model
model.add(Dropout(0.3)) #Drops out in 0.3 seconds

#Repeat 4 times while omitting and changing slight details

model.add(LSTM(units=60, activation='relu', return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units=80, activation='relu', return_sequences= True))
model.add(Dropout(0.5))

model.add(LSTM(units=120, activation='relu',))
model.add(Dropout(0.6))

model.add(Dense(units= 1))

#Next is to compile out model

model.compile(optimizer="adam", loss="mean_squared_error") #compiles model using the adam optimizer

#Next we fit the data

model.fit(train_x, train_y, epochs = 50) #fits the model to the train_x and train_y at a 50 epochs

model.save('keras_model.h4')

#Here we will Create a test data

data_test = scaler_data[length_training_data-prediction_days: , :] #creates new array data_set that contains the scaled values
test_x = [] #creates an empty set
test_y = close_data_set[length_training_data: ,:] #Starts from i while appending it to our y

for i in range(prediction_days, len(data_test)):
    test_x.append(data_test[i-prediction_days:i, 0]) #Starts at 0 and goes all the way to i while appending to our x

#Convert to np arrays

test_x = np.array(train_x) #converts the test_x data to an array
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1)) #reshapes the test_x array data

#Next we will be making the predictions
prediction = model.predict(test_x)
prediction = scaler.inverse_transform(prediction)

#Here we will be evaluating the model

root_square_error = np.sqrt( np.mean(prediction - test_y) **2)

#Lastly we will be plotting the prediction data in a Chart

train_predict= data[:length_training_data]
valid_predict = data[length_training_data:]
valid_predict['Prediction'] = prediction

plt.figure(figsize=(12, 6))
plt.title('Model')
plt.xlabel('Data', loc = "center")
plt.ylabel("Close Price $", loc = "center")
plt.plot(train_predict['Close'])
plt.plot(valid_predict[['Close', 'Prediction']])
plt.legend(['Train_Predict', 'Valied_predict', 'Prediction'], loc="lower right")
plt.show
valid_predict