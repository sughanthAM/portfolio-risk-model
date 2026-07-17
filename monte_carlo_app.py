import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")
st.title("Monte Carlo Stock Simulator")
st.markdown("Simulate 10,000 future price scenarios for any NSE stock")

st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter NSE Tickers (comma separated)", value="HAL.NS, BEL.NS, ICICIBANK.NS, NTPC.NS")
simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, 5000, step=1000)
days = st.sidebar.slider("Days to Simulate", 30, 365, 252)
period = st.sidebar.selectbox("Historical Data Period", ["6mo", "1y", "2y"], index=1)
run = st.sidebar.button("Run Simulation")

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    st.info(f"Downloading data for {tickers}...")
    
    data = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
    elif len(tickers) == 1:
        data = data.to_frame(name=tickers[0])
    data = data.dropna(axis=1, how="all")
    returns = data.pct_change().dropna()

    results = {}
    
    for stock in data.columns:
        daily_return = returns[stock].mean()
        daily_volatility = returns[stock].std()
        last_price = data[stock].iloc[-1]
        matrix = np.zeros((days, simulations))
        for sim in range(simulations):
            price = last_price
            for day in range(days):
                shock = np.random.normal(daily_return, daily_volatility)
                price = price * (1 + shock)
                matrix[day][sim] = price
        results[stock] = matrix

    st.subheader("Simulation Summary")
    cols = st.columns(len(results))
    for idx, (stock, matrix) in enumerate(results.items()):
        final_prices = matrix[-1]
        cols = st.columns(max(1, len(results)))
        prob_profit = (final_prices > last_price).mean() * 100
        prob_loss20 = (final_prices < last_price * 0.8).mean() * 100
        median = np.percentile(final_prices, 50)
        best = np.percentile(final_prices, 95)
        worst = np.percentile(final_prices, 5)
        with cols[idx]:
            st.metric(stock.replace(".NS", ""), f"₹{last_price:.0f}")
            st.write(f"Median: ₹{median:.0f}")
            st.write(f"Best (95th): ₹{best:.0f}")
            st.write(f"Worst (5th): ₹{worst:.0f}")
            if prob_profit > 55:
                st.success(f"Profit Prob: {prob_profit:.1f}%")
            elif prob_profit > 45:
                st.warning(f"Profit Prob: {prob_profit:.1f}%")
            else:
                st.error(f"Profit Prob: {prob_profit:.1f}%")
            st.write(f"-20% Risk: {prob_loss20:.1f}%")

    st.subheader("Simulation Charts")
    stock_colors = ["#00C48C", "#7B61FF", "#00A3FF", "#FF6B6B", "#FFB800", "#FF8C00"]
    n = len(results)
    cols_chart = 2
    rows_chart = (n + 1) // 2
    fig, axes = plt.subplots(rows_chart, cols_chart, figsize=(14, 5 * rows_chart))
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = np.array([[axes]])
    elif rows_chart == 1:
        axes = np.array([axes])

    for idx, (stock, matrix) in enumerate(results.items()):
        row = idx // cols_chart
        col = idx % cols_chart
        ax = axes[row][col]
        last_price = data[stock].iloc[-1]
        color = stock_colors[idx % len(stock_colors)]
        for i in range(min(200, simulations)):
            ax.plot(matrix[:, i], color=color, alpha=0.05, linewidth=0.5)
        ax.plot(np.percentile(matrix, 95, axis=1), color="green", linewidth=1.5, linestyle="--", label="95th (Best)")
        ax.plot(np.percentile(matrix, 50, axis=1), color="white", linewidth=2, label="50th (Median)")
        ax.plot(np.percentile(matrix, 5, axis=1), color="red", linewidth=1.5, linestyle="--", label="5th (Worst)")
        ax.axhline(last_price, color="yellow", linewidth=1, linestyle=":", label=f"Current ₹{last_price:.0f}")
        prob = (matrix[-1] > last_price).mean() * 100
        ax.set_title(f"{stock.replace('.NS','')} — Profit Prob: {prob:.1f}%", fontweight="bold", color="white")
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.set_xlabel("Trading Days", color="white")
        ax.set_ylabel("Price (₹)", color="white")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for idx in range(n, rows_chart * cols_chart):
        axes[idx // cols_chart][idx % cols_chart].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    st.success("Simulation complete!")