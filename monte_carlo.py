import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = {
    "HAL.NS": "Defence",
    "BEL.NS": "Defence",
    "NTPC.NS": "Defence",
    "ICICIBANK.NS": "Banking"
}

print("Downloading data...")
data = yf.download(list(stocks.keys()), period="1y", auto_adjust=True)["Close"]

returns = data.pct_change().dropna()

print("\n--- Daily Returns Sample ---")
print(returns.tail())
print("\nData ready for Monte Carlo!")
# Monte Carlo Simulation
simulations = 10000
days = 252  # 1 trading year ahead

results = {}

for stock in data.columns:
    daily_return = returns[stock].mean()
    daily_volatility = returns[stock].std()
    
    last_price = data[stock].iloc[-1]
    
    simulation_matrix = np.zeros((days, simulations))
    
    for sim in range(simulations):
        price = last_price
        for day in range(days):
            shock = np.random.normal(daily_return, daily_volatility)
            price = price * (1 + shock)
            simulation_matrix[day][sim] = price
    
    results[stock] = simulation_matrix

print("Monte Carlo done — 10,000 scenarios per stock!")

# Summary stats
print("\n===== MONTE CARLO RESULTS =====")
for stock, matrix in results.items():
    final_prices = matrix[-1]
    last_price = data[stock].iloc[-1]
    
    print(f"\n{stock}")
    print(f"  Current Price:     ₹{last_price:.2f}")
    print(f"  Median (50th %):   ₹{np.percentile(final_prices, 50):.2f}")
    print(f"  Best Case (95th):  ₹{np.percentile(final_prices, 95):.2f}")
    print(f"  Worst Case (5th):  ₹{np.percentile(final_prices, 5):.2f}")
    print(f"  Prob of Profit:    {(final_prices > last_price).mean()*100:.1f}%")
    print(f"  Prob of -20% loss: {(final_prices < last_price*0.8).mean()*100:.1f}%")
    # ---- VISUALS ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Monte Carlo Simulation — Defence & Banking Stocks", fontsize=16, fontweight="bold")

stock_colors = {
    "BEL.NS": "#00C48C",
    "HAL.NS": "#7B61FF", 
    "ICICIBANK.NS": "#00A3FF",
    "NTPC.NS": "#FF6B6B"
}

for idx, (stock, matrix) in enumerate(results.items()):
    ax = axes[idx // 2][idx % 2]
    last_price = data[stock].iloc[-1]
    color = stock_colors[stock]
    
    # Plot 200 random simulation paths
    for i in range(200):
        ax.plot(matrix[:, i], color=color, alpha=0.05, linewidth=0.5)
    
    # Plot percentile lines
    ax.plot(np.percentile(matrix, 95, axis=1), color="green", linewidth=1.5, linestyle="--", label="95th (Best)")
    ax.plot(np.percentile(matrix, 50, axis=1), color="white", linewidth=2, label="50th (Median)")
    ax.plot(np.percentile(matrix, 5, axis=1), color="red", linewidth=1.5, linestyle="--", label="5th (Worst)")
    
    # Horizontal line at current price
    ax.axhline(last_price, color="yellow", linewidth=1, linestyle=":", label=f"Current ₹{last_price:.0f}")
    
    final_prices = matrix[-1]
    prob_profit = (final_prices > last_price).mean() * 100
    
    ax.set_title(f"{stock.replace('.NS','')} — Profit Probability: {prob_profit:.1f}%", fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (₹)")
    ax.legend(fontsize=7)
    ax.set_facecolor("#0d1117")
    ax.grid(True, alpha=0.2)

fig.patch.set_facecolor("#0d1117")
plt.tight_layout()
plt.savefig("d:/monte_carlo_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved!")