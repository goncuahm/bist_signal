import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Custom Stock Technical & Fundamental Strategy with Machine Forecast", layout="wide")

# ------------------------------
# Title
# ------------------------------
st.title("📊 Custom Stock Technical Strategy — Backtest & LSTM Forecast")

# ------------------------------
# User Input for Tickers
# ------------------------------
st.subheader("📝 Enter Stock Tickers")
ticker_input = st.text_area(
    "Enter stock tickers (one per line or comma-separated):",
    value="BASGZ.IS, AFYON.IS, ENJSA.IS, EREGL.IS, AYEN.IS, PAGYO.IS, YGGYO.IS, TUPRS.IS, KRDMD.IS, SISE.IS",
    height=100
)

# Parse ticker input
if ticker_input:
    # Split by commas or newlines and clean up
    tickers = [t.strip().upper() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
else:
    tickers = []

if not tickers:
    st.warning("⚠️ Please enter at least one ticker symbol.")
    st.stop()

st.success(f"✅ Analyzing {len(tickers)} stock(s): {', '.join(tickers)}")

# ------------------------------
# Fixed Parameters (Default Values)
# ------------------------------
period = "1y"
rsi_period = 9
buy_threshold = 40
sell_threshold = 65
tcost = 0.002

# Display parameters
st.info(f"**Strategy Parameters:** Period = {period} | RSI Period = {rsi_period} | Buy Threshold (RSI < {buy_threshold}) | Sell Threshold (RSI > {sell_threshold}) | Transaction Cost = {tcost*100}%")

# ------------------------------
# EPS Function
# ------------------------------
def get_eps(ticker):
    """Fetch EPS (Earnings Per Share) for a given ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        eps = info.get('trailingEps', None)
        return eps if eps is not None else np.nan
    except:
        return np.nan

# ------------------------------
# RSI Function
# ------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ------------------------------
# Backtest Function
# ------------------------------
def backtest_strategy(df, x1, x2, tcost):
    open_positions = []
    closed_trades = []
    buy_signals_idx = []
    sell_signals_idx = []

    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i]
        price = df["Close"].iloc[i]
        date = df.index[i]

        if rsi < x1:
            open_positions.append({"entry_price": price, "entry_date": date, "entry_idx": i})
            buy_signals_idx.append(i)
        elif rsi > x2 and open_positions:
            entry = open_positions.pop(0)
            sell_signals_idx.append(i)
            closed_trades.append({
                "buy_date": entry["entry_date"],
                "buy_price": entry["entry_price"],
                "sell_date": date,
                "sell_price": price,
                "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
            })

    total_return = np.sum([t["return"] for t in closed_trades])
    avg_return = np.mean([t["return"] for t in closed_trades]) if closed_trades else 0
    return total_return, avg_return, closed_trades, buy_signals_idx, sell_signals_idx

# ------------------------------
# Analysis Loop
# ------------------------------
st.subheader("🔍 Scanning Stocks...")

results = []
buy_signals = []
sell_signals = []
first_stock_data = None

for idx, ticker in enumerate(tickers):
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            st.warning(f"No data found for {ticker}")
            continue
        data["RSI"] = compute_rsi(data["Close"], rsi_period)
        data = data.dropna()

        total_return, avg_return, trades, buy_idx, sell_idx = backtest_strategy(data, buy_threshold, sell_threshold, tcost)
        latest_rsi = float(data["RSI"].iloc[-1])
        latest_close = float(data["Close"].iloc[-1])

        # Fetch EPS
        eps = get_eps(ticker)

        if latest_rsi < buy_threshold:
            # Only add to buy signals if EPS is positive
            if not np.isnan(eps) and eps > 0:
                signal = "BUY"
                buy_signals.append((ticker, latest_close, latest_rsi, eps))
            else:
                signal = "HOLD"
        elif latest_rsi > sell_threshold:
            signal = "SELL"
            sell_signals.append((ticker, latest_close, latest_rsi, eps))
        else:
            signal = "HOLD"

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Latest RSI": round(latest_rsi, 2),
            "Latest Close": round(latest_close, 2),
            "EPS": round(eps, 4) if not np.isnan(eps) else "N/A",
            "Cumulative Return (%)": round(total_return * 100, 2),
            "Return per Trade (%)": round(avg_return * 100, 2),
            "Number of Trades": len(trades)
        })

        # Store first stock data for plotting
        if idx == 0:
            first_stock_data = {
                "data": data,
                "ticker": ticker,
                "buy_idx": buy_idx,
                "sell_idx": sell_idx
            }

    except Exception as e:
        st.error(f"Error with {ticker}: {e}")

# ------------------------------
# Convert to DataFrame
# ------------------------------
results_df = pd.DataFrame(results).sort_values(by="Return per Trade (%)", ascending=False)

# ------------------------------
# Calculate Position Sizing
# ------------------------------
TOTAL_CAPITAL = 1000000  # Total capital in Liras
total_trades = results_df["Number of Trades"].sum()

if total_trades > 0:
    capital_per_trade = TOTAL_CAPITAL / (total_trades/2)
else:
    capital_per_trade = 0

# Format buy and sell DataFrames with proper rounding and order size
if buy_signals:
    buy_df = pd.DataFrame(buy_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
    buy_df["Close Price"] = buy_df["Close Price"].round(2)
    buy_df["RSI"] = buy_df["RSI"].round(2)
    buy_df["EPS"] = buy_df["EPS"].round(4)
    # Calculate order size (number of shares)
    buy_df["Order Size"] = (capital_per_trade / buy_df["Close Price"]).apply(lambda x: int(round(x)))
else:
    buy_df = pd.DataFrame()

if sell_signals:
    sell_df = pd.DataFrame(sell_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
    sell_df["Close Price"] = sell_df["Close Price"].round(2)
    sell_df["RSI"] = sell_df["RSI"].round(2)
    sell_df["EPS"] = sell_df["EPS"].apply(lambda x: round(x, 4) if not np.isnan(x) else "N/A")
    # Calculate order size (number of shares)
    sell_df["Order Size"] = (capital_per_trade / sell_df["Close Price"]).apply(lambda x: int(round(x)))
else:
    sell_df = pd.DataFrame()

# ------------------------------
# Display Results
# ------------------------------
st.subheader("📈 Technical Strategy Results")
st.dataframe(results_df, use_container_width=True)

# Display capital allocation info
st.info(f"💰 **Capital Allocation:** Total Capital = ₺{TOTAL_CAPITAL:,.0f} | Total Trades = {total_trades} | Capital per Trade = ₺{capital_per_trade:,.2f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🟢 Current BUY Signals (EPS > 0)")
    if not buy_df.empty:
        st.dataframe(buy_df, use_container_width=True)
    else:
        st.info("No buy signals with positive EPS found.")

with col2:
    st.subheader("🔴 Current SELL Signals")
    if not sell_df.empty:
        st.dataframe(sell_df, use_container_width=True)
    else:
        st.info("No sell signals found.")

# ================================================================
# PART 1.5: Plot Close Price with Buy/Sell Signals for First Stock
# ================================================================
if first_stock_data:
    st.subheader(f"📉 Close Price with Buy/Sell Signals — {first_stock_data['ticker']}")

    fig, ax = plt.subplots(figsize=(12, 5))
    data = first_stock_data["data"]

    # Plot close price
    ax.plot(data.index, data["Close"], label="Close Price", color="steelblue", linewidth=1.5)

    # Mark buy signals (green)
    if first_stock_data["buy_idx"]:
        buy_dates = data.index[first_stock_data["buy_idx"]]
        buy_prices = data["Close"].iloc[first_stock_data["buy_idx"]]
        ax.scatter(buy_dates, buy_prices, color="green", marker="^", s=100, label="Buy Signal", zorder=5)

    # Mark sell signals (red)
    if first_stock_data["sell_idx"]:
        sell_dates = data.index[first_stock_data["sell_idx"]]
        sell_prices = data["Close"].iloc[first_stock_data["sell_idx"]]
        ax.scatter(sell_dates, sell_prices, color="red", marker="v", s=100, label="Sell Signal", zorder=5)

    ax.set_title(f"{first_stock_data['ticker']} — Close Price with RSI-Based Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ================================================================
# PART 2: Select Stock for LSTM Forecast
# ================================================================
st.subheader("🤖 LSTM RSI Forecast (User-Selected Stock)")

selected_ticker = st.selectbox("Select a stock for RSI forecast:", ["None"] + tickers)

# ------------------------------
# LSTM Function
# ------------------------------
def lstm_forecast_rsi(rsi_series, n_past=9, n_future=4):
    if len(rsi_series) < n_past + 5:
        return [np.nan] * n_future

    scaler = MinMaxScaler(feature_range=(0, 1))
    rsi_scaled = scaler.fit_transform(rsi_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_past, len(rsi_scaled) - n_future):
        X.append(rsi_scaled[i - n_past:i, 0])
        y.append(rsi_scaled[i:i + n_future, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_past, 1)),
        Dense(25, activation='relu'),
        Dense(n_future)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=8, verbose=0)

    last_window = rsi_scaled[-n_past:].reshape((1, n_past, 1))
    forecast_scaled = model.predict(last_window)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    return forecast

# ------------------------------
# Run forecast only if user selected a stock
# ------------------------------
if selected_ticker != "None":
    st.write(f"### 🔮 Forecasting RSI for: **{selected_ticker}**")
    data = yf.download(selected_ticker, period=period, auto_adjust=True, progress=False)
    data["RSI"] = compute_rsi(data["Close"], rsi_period)
    data = data.dropna()

    forecast = lstm_forecast_rsi(data["RSI"], n_past=9, n_future=4)

    # Show forecast table
    forecast_df = pd.DataFrame({
        "Ticker": [selected_ticker],
        "Day+1 RSI": [round(forecast[0], 2)],
        "Day+2 RSI": [round(forecast[1], 2)],
        "Day+3 RSI": [round(forecast[2], 2)],
        "Day+4 RSI": [round(forecast[3], 2)],
    })
    st.dataframe(forecast_df, use_container_width=True)

    # Plot RSI with forecast
    st.write("📊 RSI Trend with Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot historical RSI
    ax.plot(data.index[-100:], data["RSI"].iloc[-100:], label="Historical RSI", color="steelblue", linewidth=2)

    # Create future dates for forecast (assuming daily data)
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=5, freq='D')[1:]  # Next 4 days

    # Plot forecast RSI
    ax.plot(forecast_dates, forecast, label="Forecasted RSI", color="orange", linewidth=2, linestyle='--', marker='o')

    # Add horizontal lines for buy/sell thresholds
    ax.axhline(y=buy_threshold, color='green', linestyle=':', alpha=0.7, label=f'Buy Threshold ({buy_threshold})')
    ax.axhline(y=sell_threshold, color='red', linestyle=':', alpha=0.7, label=f'Sell Threshold ({sell_threshold})')

    ax.set_title(f"{selected_ticker} — RSI with 4-Day Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

else:
    st.info("Select a stock above to generate RSI LSTM forecast.")

st.caption("Developed for educational and research purposes — RSI Strategy + LSTM Forecast on Custom Stocks.")
