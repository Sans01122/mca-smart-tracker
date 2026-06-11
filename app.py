import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Page Title and Configuration
st.set_page_config(page_title="AI Portfolio Tracker", layout="wide")
st.title("Smart AI Investment Portfolio Tracker")
st.write("MCA 2nd Sem Practical Examination Project")

# 2. Hardcoded Top Assets (Simplifies user inputs)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# 3. Dynamic Data Pipeline Function with Format Fix
@st.cache_data
def load_data():
    # Downloads 1 year of data with traditional column layout
    df = yf.download(tickers, period="1y", group_by="ticker", auto_adjust=False)
    
    # Extract only the Close or Adj Close data safely for each ticker
    cleaned_df = pd.DataFrame()
    for ticker in tickers:
        if 'Adj Close' in df[ticker].columns:
            cleaned_df[ticker] = df[ticker]['Adj Close']
        else:
            cleaned_df[ticker] = df[ticker]['Close']
            
    return cleaned_df

try:
    data = load_data()
    latest_prices = data.iloc[-1]

    # 4. Interactive Data Visualization Grid
    st.subheader("Live Asset Status Overview")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("Latest Fetched Prices (USD):")
        st.dataframe(pd.DataFrame({"Current Price": latest_prices}))
    with col2:
        fig = px.pie(values=[25, 25, 25, 25], names=tickers, title="Portfolio Balance Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Machine Learning Core Engine
    st.markdown("---")
    st.subheader("Machine Learning Trend Forecasting (Next 30 Days)")
    selected_stock = st.selectbox("Select target asset to execute ML Model:", tickers)
    
    # Process sequential matrix arrays
    stock_prices = data[selected_stock].dropna()
    X = np.arange(len(stock_prices)).reshape(-1, 1) # Time sequence array
    y = stock_prices.values # Actual pricing vectors
    
    # Train Linear Regression AI model on the fly
    model = LinearRegression()
    model.fit(X, y)
    
    # Run predictions for future index parameters
    future_days = np.arange(len(stock_prices), len(stock_prices) + 30).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # Clean structures to feed graph rendering engine
    hist_df = pd.DataFrame({"Day": X.flatten(), "Price": y, "Dataset Type": "Historical Market Data"})
    pred_df = pd.DataFrame({"Day": future_days.flatten(), "Price": predictions, "Dataset Type": "AI Future Target Prediction"})
    final_df = pd.concat([hist_df, pred_df])
    
    # Draw real-time prediction graph
    fig_forecast = px.line(final_df, x="Day", y="Price", color="Dataset Type", title=f"{selected_stock} Predictive Trajectory Map")
    st.plotly_chart(fig_forecast, use_container_width=True)

except Exception as e:
    st.error(f"Data stream disconnect. Please check network. Error trace: {e}")
