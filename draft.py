
#%% dependencies
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

# ML and AI
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers.activation import LeakyReLU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam

#%% 

# read in factor data as pd.dataframe
factors = pd.read_csv("./data/PredictorLSretWide.csv")

# format date column
factors["date"] = pd.to_datetime(factors["date"])

# filter factors for relevant period
factors = factors.loc[(factors['date'] >= pd.to_datetime('01-12-1990', format=r"%d-%m-%Y")) & (factors['date'] <= pd.to_datetime('31-03-2015', format=r"%d-%m-%Y"))]
  
# initialize parameters
start_date = datetime(1990, 12, 1)
end_date = datetime(2015, 3, 31)

# get the data
sp500 = yf.download('^SPX', start = start_date,
                   end = end_date)

# get last monthly closing prices (adjusted)
sp500_monthly = sp500["Adj Close"].resample("M").last()


## TEMPORARY ##
non_na_columns = factors.columns[factors.isna().sum() == 0]


factors = factors[non_na_columns]

# assign returns to factor data frame - merging is not suitable since dates vary slightly
factors["Return"] = sp500_monthly.pct_change().values

# drop na from returns
factors = factors.dropna()

# reset the index
factors = factors.reset_index()


#%% 

# create predictor variable matrix X and response y
y = np.array(factors["Return"].values)
X = np.array(factors.drop(columns=["Return", "date", "index"]))

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X)

# Apply standard scaling to the entire matrix
X_scaled = scaler.transform(X)

# Split your data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42, shuffle=False)

# define the number of time steps to be considered as input, i.e. how many lags: t-n, ..., t-1, t-0
window_length = 5

# number of factors (input features) to consider 
num_features = X_scaled.shape[1]

#%%
# generate a timeseries object to feed into the LSTM network
ts_train = TimeseriesGenerator(data=X_train, targets=y_train, length=window_length, sampling_rate=1, batch_size=1,
                               stride=1)
ts_test = TimeseriesGenerator(data=X_test, targets=y_test, length=window_length, sampling_rate=1, batch_size=1,
                               stride=1)

#%%

# Build the LSTM model
model = Sequential()
model.add(LSTM(200, input_shape=(window_length, num_features), return_sequences=False))
# model.add(Dropout(rate=0.1))
# model.add(LSTM(units=64, return_sequences=True))
# model.add(Dropout(rate=0.1))
# model.add(LSTM(units=32, return_sequences=False))
# model.add(Dropout(rate=0.1))
model.add(Dense(units=1))


# print the model summary
print(model.summary())

#%%

# define early stopping criteria
early_stopping = EarlyStopping(monitor="val_loss", patience=2)

# define optimiser
optimizer = Adam(lr=0.00001)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer,
              metrics="mse")

#%% Train the model

# number of samples to consider for each epoch
batch_size = 100

# epochs
epochs = 20


fit1 = model.fit(ts_train, validation_data=ts_test, batch_size=batch_size, 
          epochs=epochs,
          verbose=1,
          shuffle=False)#,
          #callbacks=[early_stopping])


#%% plot train hsitory


train_loss = fit1.history['loss']
val_loss = fit1.history['val_loss']

train_mse = fit1.history['mse']
val_mse = fit1.history['mse']

# Plot the training loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training accuracy
plt.plot(train_mse, label='Training MSE')
plt.plot(val_mse, label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()


#%%

# Evaluate the model
score = model.evaluate(ts_test, verbose=0)

# make predictions based on the test set
pred = model.predict(ts_test, verbose=0)



# %% plot the predictions against the target

plt.plot(pred, color="red", label="pred");
plt.plot(y_test);
plt.legend()

# %%
