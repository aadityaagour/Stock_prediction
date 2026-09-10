# 📈 AI Stock Market Predictor & Forecaster

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

An end-to-end Machine Learning web application that analyzes historical financial data and predicts stock prices using a **4-Layer Stacked Long Short-Term Memory (LSTM)** deep learning model. Powered by **Streamlit**, **Yahoo Finance (`yfinance`)**, and **TensorFlow / Keras**, ready for 1-click cloud deployment on **Render**.

---

## 🚀 Key Features

- **🔴 Real-Time Market Data:** Automatically downloads historical daily stock prices directly from Yahoo Finance for any ticker symbol (e.g., `GOOG`, `AAPL`, `MSFT`, `TSLA`, `NVDA`, `AMZN`).
- **🧠 4-Layer Stacked LSTM Model:** Uses a trained deep neural network that captures temporal dependencies across 100-day rolling sequence windows.
- **🔮 Next-Day Price Forecasting:** Analyzes the most recent 100 trading days to forecast the **next trading day's closing price**, expected price movement ($), percentage change (%), and market trend direction (Bullish 🔼 / Bearish 🔽).
- **📉 Technical Moving Average Analysis:**
  - **50-Day Moving Average (MA50):** Evaluates short-to-medium-term price momentum.
  - **50-Day vs. 100-Day Moving Averages:** Identifies trend momentum and crossovers.
  - **100-Day vs. 200-Day Moving Averages:** Highlights long-term institutional trend direction and Golden/Death Cross patterns.
- **📊 Actual vs. Predicted Visualizations:** Interactive Matplotlib visualizations comparing original market prices against LSTM model predictions over the test period.
- **🎯 Performance Evaluation Metrics:** Computes **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** to quantify forecast accuracy.
- **☁️ Production-Ready for Render:** Lightweight `tensorflow-cpu` dependency configuration designed to run smoothly within Render's free tier (512MB RAM) without memory overflow or timeouts.

---

## 🏗️ Deep Learning Architecture

The prediction engine utilizes a stacked LSTM recurrent neural network designed for sequential time-series forecasting:

```
Input: 100-Day Sequential Window [batch_size, 100, 1]
   │
   ▼
LSTM Layer 1 (50 Units, return_sequences=True)
   │
Dropout (0.2 Regularization)
   │
   ▼
LSTM Layer 2 (60 Units, return_sequences=True)
   │
Dropout (0.2 Regularization)
   │
   ▼
LSTM Layer 3 (80 Units, return_sequences=True)
   │
Dropout (0.2 Regularization)
   │
   ▼
LSTM Layer 4 (120 Units, return_sequences=False)
   │
Dropout (0.2 Regularization)
   │
   ▼
Dense Output Layer (1 Unit - Continuous Price)
```

- **Feature Scaling:** `MinMaxScaler(feature_range=(0, 1))` scales historical prices to optimize gradient descent convergence.
- **Loss Function:** Mean Squared Error (`mean_squared_error`).
- **Optimizer:** Adam.
- **Weights File:** `Stock Predictions Model.keras` (2.2 MB).

---

## 📂 Project Structure

```
Stock_prediction/
├── .streamlit/
│   └── config.toml                  # Streamlit headless & theme settings
├── Stock Predictions Model.keras    # Trained 4-layer LSTM neural network
├── Untitled.ipynb                   # Jupyter notebook with model training & EDA
├── app.py                           # Streamlit web application & inference pipeline
├── render.yaml                      # Render Blueprint deployment configuration
├── requirements.txt                 # Optimized production dependencies
├── .python-version                  # Python 3.11 version pin for Render
├── .gitignore                       # Clean Git ignore rules
└── README.md                        # Documentation
```

---

## 💻 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aadityaagour/Stock_prediction.git
cd Stock_prediction
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Deploying to Render (Step-by-Step)

This repository is pre-configured with `render.yaml`, `.python-version`, and `.streamlit/config.toml` for seamless deployment.

### Method 1: Blueprint Deployment (Recommended & Fastest)

1. Sign in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Blueprint**.
3. Connect your GitHub account and select the **`aadityaagour/Stock_prediction`** repository.
4. Render will automatically detect `render.yaml` and configure:
   - **Service Type:** Web Service
   - **Runtime:** Python 3.11.8
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Apply**. Render will build and deploy your app in under 2 minutes!

---

### Method 2: Manual Web Service Setup

If you prefer setting up the web service manually on Render:

1. In Render Dashboard, click **New +** > **Web Service**.
2. Select your repository: `aadityaagour/Stock_prediction`.
3. Fill in the following settings:
   - **Name:** `stock-price-predictor` (or your preferred name)
   - **Language / Runtime:** `Python 3`
   - **Region:** Any region closest to your users (e.g., Oregon, Frankfurt, Singapore)
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - **Instance Type:** `Free`
4. Under **Advanced** > **Add Environment Variable**:
   - `PYTHON_VERSION`: `3.11.8`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: `false`
5. Click **Create Web Service**.

Once the build completes, Render will provide your public live URL (e.g., `https://stock-price-predictor.onrender.com`).

---

## 📊 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **[Python 3.11](https://python.org)** | Core programming runtime |
| **[Streamlit](https://streamlit.io)** | Interactive web UI and visualization dashboard |
| **[TensorFlow / Keras](https://tensorflow.org)** | Deep learning LSTM model loading and inference |
| **[yfinance](https://pypi.org/project/yfinance/)** | Live historical market financial data |
| **[pandas](https://pandas.pydata.org/)** & **[numpy](https://numpy.org/)** | Data wrangling and array manipulation |
| **[scikit-learn](https://scikit-learn.org/)** | Data normalization (`MinMaxScaler`) |
| **[Matplotlib](https://matplotlib.org/)** | Technical analysis charts and price plots |
| **[Render](https://render.com/)** | Cloud platform hosting and continuous deployment |

---

## ⚠️ Financial Disclaimer

This application is created for **educational and research purposes only**. Financial markets are volatile and inherently unpredictable. The forecasts provided by this LSTM model should not be considered financial, investment, or trading advice. Always conduct your own research or consult with a certified financial advisor before making investment decisions.

---

## 👤 Author

- **GitHub:** [@aadityaagour](https://github.com/aadityaagour)
- **Repository:** [Stock_prediction](https://github.com/aadityaagour/Stock_prediction)
