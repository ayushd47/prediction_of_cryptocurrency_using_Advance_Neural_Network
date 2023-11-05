## Part1 Time Series Prediction(LSTM)

# In[]:


#importing the required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import ccxt

 #Set up the exchange
exchange = ccxt.binance()  # You can choose a different exchange here

st.title('CRYPTO PRICE PREDICTER')
st.subheader("Time series data analysis Prediction:")

user_input = st.text_input('Enter a currency for prediction', 'BTC')

# Loading the financial data (web scraping)

crypto = user_input
pair = f'{crypto}/USDT'  # You can choose a different trading pair if needed

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

# Fetch historical OHLCV data
ohlcv = exchange.fetch_ohlcv(pair, timeframe='1d', since=int(start.timestamp()) * 1000)

# Convert OHLCV data to a DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

st.subheader("Data From 2020 - Till today")
st.write(df.head(10))

# In[]:
    
# 2.Preparing the data(Data Pre-processing)

#removing unnecessary coloumns
df = df.reset_index()
df = df.drop(['timestamp'], axis = 1)

# In[]:

df.head()

# In[]:
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize = (10,4))
plt.plot(df.close)
st.pyplot(fig)

# In[]:

df1 = df.reset_index()['close']

# In[]:


# 3.Downscaling the data (MinMax scaler)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


print(df1)

# In[]:
    
# 4.Splitting the dataset into train and test split

training_size = int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

print(training_size,test_size)
# In[]:
#converting an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[ i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[]:

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[]:

print(X_train.shape)
print(y_train.shape)


# In[]:

print(X_test.shape)
print(y_test.shape)


# In[]:

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)



# In[]:
# 5.LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM 

model = Sequential()
model.add(LSTM(units = 50,return_sequences = True,input_shape = (100,1)))

model.add(LSTM(units = 50,return_sequences = True))

model.add(LSTM(units = 50))

model.add(Dense(units = 1))

model.compile(loss='mean_squared_error',optimizer='adam')


# In[]:

model.summary()

# In[]:

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=64,verbose=1)

# In[]:
#7. Testing our model on test dataset
#Prediction and Performance Matrix
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# In[]:
#Upscaling the data

#back to orirginal form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# In[]:

# RMSE performance metrics for training dataset
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# RMSE for test data
math.sqrt(mean_squared_error(y_test,test_predict))


# In[]:

# Plotting 
fig2=plt.figure(figsize=(12,6)) 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(scaler.inverse_transform(df1),'b',label ='Original Price')
plt.plot(trainPredictPlot,'orange',label ='Train Predicted Price')
plt.plot(testPredictPlot,'r',label='Test Predicted Price')
plt.legend()
st.subheader("Predicted price vs Original(Past)")
st.pyplot(fig2)


#comparing the orginal close prices to model predicted prices

# In[]:

# 7.Predicting future 30 days price
tlen=len(test_data)

x_input=test_data[tlen-100:].reshape(1,-1)
print(x_input.shape)
# In[]:

#storing data collected into list
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[]:

#logic for prediction

lst_output=[]
n_steps = 100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else :
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
# In[]:

day_new=np.arange(1,101)
day_pred = np.arange(101,131)

# In[]:

datalen=len(df1)

# In[]:
fig3=plt.figure(figsize=(12,6)) 
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(day_new,scaler.inverse_transform(df1[datalen-100:]),label='Past prices')
plt.plot(day_pred,scaler.inverse_transform(lst_output),label='Future Price')
plt.legend()
st.subheader("Predicted price for next 30 days")
st.pyplot(fig3)

