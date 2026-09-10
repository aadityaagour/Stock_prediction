import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Stock Price Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .forecast-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        border: 1px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load Trained Model (Cached for fast rendering)
@st.cache_resource
def load_prediction_model():
    model_path = os.path.join(os.path.dirname(__file__), 'Stock Predictions Model.keras')
    return load_model(model_path)

try:
    model = load_prediction_model()
except Exception as e:
    st.error(f"⚠️ Error loading deep learning model ('Stock Predictions Model.keras'): {e}")
    st.stop()

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

popular_tickers = ["GOOG", "AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
selected_preset = st.sidebar.selectbox("Quick Select Ticker:", ["Custom"] + popular_tickers)

if selected_preset != "Custom":
    default_ticker = selected_preset
else:
    default_ticker = "GOOG"

stock = st.sidebar.text_input("Enter Stock Ticker Symbol:", value=default_ticker).strip().upper()

col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input("Start Date", value=datetime.date(2012, 1, 1))
with col_end:
    end_date = st.date_input("End Date", value=datetime.date(2023, 12, 31))

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Architecture")
st.sidebar.markdown("""
- **Model Type:** 4-Layer Stacked LSTM
- **Layer 1:** LSTM (50 units) + Dropout (0.2)
- **Layer 2:** LSTM (60 units) + Dropout (0.2)
- **Layer 3:** LSTM (80 units) + Dropout (0.2)
- **Layer 4:** LSTM (120 units) + Dropout (0.2)
- **Dense Output:** 1 Unit (Continuous Price)
- **Sequence Length:** 100 Trading Days
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Model uses 100 consecutive days of closing prices to predict future price trends.")

# Main Interface Header
st.markdown('<div class="main-header">📈 AI Stock Market Predictor & Forecaster</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Deep Learning LSTM analysis and real-time forecasting for <b>{stock}</b></div>', unsafe_allow_html=True)

if not stock:
    st.warning("Please enter a valid stock ticker symbol.")
    st.stop()

if start_date >= end_date:
    st.error("Error: Start Date must precede End Date.")
    st.stop()

# Download Market Data
with st.spinner(f"Fetching market data for {stock} from Yahoo Finance..."):
    data = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

if data.empty or len(data) == 0:
    st.error(f"❌ No historical data found for ticker '{stock}'. Please check the symbol and date range.")
    st.stop()

# Robust extraction of Close price series across various yfinance formats
if isinstance(data.columns, pd.MultiIndex):
    if 'Close' in data.columns.levels[0]:
        close_col = data['Close']
        if isinstance(close_col, pd.DataFrame):
            close_series = close_col.iloc[:, 0].dropna()
        else:
            close_series = close_col.dropna()
    else:
        close_series = data.iloc[:, 0].dropna()
else:
    if 'Close' in data.columns:
        close_series = data['Close'].dropna()
    else:
        close_series = data.iloc[:, 0].dropna()

total_days = len(close_series)

if total_days < 150:
    st.warning(f"⚠️ Only {total_days} trading days found. The LSTM model requires at least 150 trading days (100 for sequence window + testing period).")
    st.stop()

# Key Summary Metrics
latest_price = float(close_series.iloc[-1])
prev_price = float(close_series.iloc[-2]) if total_days > 1 else latest_price
change_val = latest_price - prev_price
change_pct = (change_val / prev_price) * 100 if prev_price != 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Close Price", f"${latest_price:,.2f}", f"{change_val:+,.2f} ({change_pct:+.2f}%)")
m2.metric("Period High", f"${float(close_series.max()):,.2f}")
m3.metric("Period Low", f"${float(close_series.min()):,.2f}")
m4.metric("Total Trading Days", f"{total_days:,}")

# Historical Data Table
with st.expander("📊 View Recent Historical Data Table"):
    st.dataframe(data.tail(15), use_container_width=True)

# Technical Analysis: Moving Averages
st.subheader("📉 Technical Analysis: Moving Averages")

tab1, tab2, tab3 = st.tabs(["50-Day Moving Average", "50-Day vs 100-Day MA", "100-Day vs 200-Day MA"])

with tab1:
    ma_50_days = close_series.rolling(50).mean()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(close_series.index, close_series, color='#22C55E', label='Close Price', linewidth=1.5)
    ax1.plot(close_series.index, ma_50_days, color='#EF4444', label='50-Day MA', linewidth=1.5, linestyle='--')
    ax1.set_title(f"{stock} Price vs 50-Day Moving Average", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig1)
    plt.close(fig1)

with tab2:
    ma_50_days = close_series.rolling(50).mean()
    ma_100_days = close_series.rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(close_series.index, close_series, color='#22C55E', label='Close Price', linewidth=1.5)
    ax2.plot(close_series.index, ma_50_days, color='#EF4444', label='50-Day MA', linewidth=1.5, linestyle='--')
    ax2.plot(close_series.index, ma_100_days, color='#3B82F6', label='100-Day MA', linewidth=1.5, linestyle='-.')
    ax2.set_title(f"{stock} Price vs 50-Day & 100-Day Moving Averages", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Price (USD)", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig2)
    plt.close(fig2)

with tab3:
    ma_100_days = close_series.rolling(100).mean()
    ma_200_days = close_series.rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(close_series.index, close_series, color='#22C55E', label='Close Price', linewidth=1.5)
    ax3.plot(close_series.index, ma_100_days, color='#EF4444', label='100-Day MA', linewidth=1.5, linestyle='--')
    ax3.plot(close_series.index, ma_200_days, color='#3B82F6', label='200-Day MA', linewidth=1.5, linestyle='-.')
    ax3.set_title(f"{stock} Price vs 100-Day & 200-Day Moving Averages", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Date", fontsize=11)
    ax3.set_ylabel("Price (USD)", fontsize=11)
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig3)
    plt.close(fig3)

# Machine Learning: LSTM Price Prediction
st.markdown("---")
st.subheader("🤖 LSTM Deep Learning Price Prediction")

# Data preparation for test sequence
train_size = int(total_days * 0.80)
data_train = pd.DataFrame(close_series[0:train_size])
data_test = pd.DataFrame(close_series[train_size:])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test_combined = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test_combined.values.reshape(-1, 1))

x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

with st.spinner("Running inference through 4-layer LSTM neural network..."):
    predict = model.predict(x, verbose=0)

scale = 1.0 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# Accuracy Metrics
rmse = np.sqrt(np.mean((predict.flatten() - y) ** 2))
mae = np.mean(np.abs(predict.flatten() - y))

col_metric1, col_metric2 = st.columns(2)
col_metric1.metric("Test Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
col_metric2.metric("Test Mean Absolute Error (MAE)", f"${mae:,.2f}")

# Plot Original vs Predicted Prices
fig4, ax4 = plt.subplots(figsize=(11, 6))
ax4.plot(y, color='#22C55E', label='Original / Actual Price', linewidth=2)
ax4.plot(predict, color='#EF4444', label='LSTM Predicted Price', linewidth=2, linestyle='--')
ax4.set_title(f"{stock} - Actual vs LSTM Predicted Price (Test Dataset)", fontsize=14, fontweight='bold')
ax4.set_xlabel("Time (Trading Days)", fontsize=12)
ax4.set_ylabel("Price (USD)", fontsize=12)
ax4.legend(loc="upper left", frameon=True)
ax4.grid(True, linestyle=':', alpha=0.6)
st.pyplot(fig4)
plt.close(fig4)

# Next-Day Stock Price Forecast
st.markdown("---")
st.subheader("🔮 Next Trading Day Price Forecast")

# Using the most recent 100 trading days
latest_100_days = close_series.tail(100).values.reshape(-1, 1)
latest_100_scaled = scaler.transform(latest_100_days)
x_next = np.array([latest_100_scaled])

next_pred_scaled = model.predict(x_next, verbose=0)
next_pred_price = float(next_pred_scaled[0, 0] * scale)

expected_delta = next_pred_price - latest_price
expected_delta_pct = (expected_delta / latest_price) * 100

st.markdown(f"""
<div class="forecast-card">
    <h3 style="margin-top:0; color:#93C5FD;">Forecast for Next Trading Day ({stock})</h3>
    <h1 style="font-size: 2.8rem; margin: 10px 0; color: {'#4ADE80' if expected_delta >= 0 else '#F87171'};">
        ${next_pred_price:,.2f}
    </h1>
    <p style="font-size: 1.15rem; color: #E2E8F0; margin-bottom: 0;">
        Expected Movement: <b>{expected_delta:+,.2f} ({expected_delta_pct:+.2f}%)</b> 
        {'🟢 Upward Trend' if expected_delta >= 0 else '🔴 Downward Trend'}
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Disclaimer: Stock market predictions are generated by an experimental LSTM deep learning model for educational and research purposes only. This tool should not be used as financial or investment advice.")