import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit UI
st.title("📈 Apple vs. Google Stock Price Prediction")

# Select stock ticker
ticker = st.selectbox("Select Stock", ["AAPL", "GOOGL"])

# Load stock data
@st.cache_data
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    df.reset_index(inplace=True)
    return df

df = load_stock_data(ticker)

# Display data
st.subheader(f"{ticker} Stock Data")
st.write(df.tail())

# Plot stock prices
st.subheader("Stock Price Over Time")
fig, ax = plt.subplots()
ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Prepare training data
def create_sequences(data, time_step=50):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        Y.append(data[i+time_step])
    return np.array(X), np.array(Y)

time_step = 50
X, Y = create_sequences(df_scaled)

# Train-test split
split = int(len(X) * 0.8)
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, activation="relu", input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=1)
        st.success("Model training complete! ✅")

# Make predictions
if st.button("Predict"):
    Y_pred = model.predict(X_test)
    Y_pred = scaler.inverse_transform(Y_pred)
    Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Plot predictions
    st.subheader("📊 Prediction vs. Actual")
    fig, ax = plt.subplots()
    ax.plot(Y_test, label="Actual Price", color="green")
    ax.plot(Y_pred, label="Predicted Price", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

st.write("🚀 Built with Streamlit | Data from Yahoo Finance")