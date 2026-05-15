import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]

print("Downloading data...")
data = yf.download(stocks, period="1y", auto_adjust=True)["Close"]
returns = data.pct_change().dropna()

trading_days = 252
annualized_return = returns.mean() * trading_days
annualized_volatility = returns.std() * np.sqrt(trading_days)
sharpe_ratio = annualized_return / annualized_volatility

def max_drawdown(prices):
    peak = prices.cummax()
    return ((prices - peak) / peak).min()

drawdowns = data.apply(max_drawdown)
correlation = returns.corr()

summary = pd.DataFrame({
    "Annual Return %": (annualized_return * 100).round(2),
    "Volatility %": (annualized_volatility * 100).round(2),
    "Sharpe Ratio": sharpe_ratio.round(2),
    "Max Drawdown %": (drawdowns * 100).round(2)
})

# ---- CHARTS ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Portfolio Risk Dashboard", fontsize=18, fontweight="bold", y=1.01)

colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6"]

# Chart 1 - Price history normalized
normalized = (data / data.iloc[0]) * 100
ax1 = axes[0, 0]
for i, col in enumerate(normalized.columns):
    ax1.plot(normalized.index, normalized[col], label=col.replace(".NS", ""), color=colors[i], linewidth=1.8)
ax1.set_title("Price Performance (Normalized to 100)", fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylabel("Value")
ax1.grid(True, alpha=0.3)

# Chart 2 - Annual return vs volatility
ax2 = axes[0, 1]
bars = ax2.bar(
    summary.index.str.replace(".NS", ""),
    summary["Annual Return %"],
    color=["#2ecc71" if x > 0 else "#e74c3c" for x in summary["Annual Return %"]]
)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Annual Return % per Stock", fontweight="bold")
ax2.set_ylabel("Return %")
ax2.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, summary["Annual Return %"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Chart 3 - Correlation heatmap
ax3 = axes[1, 0]
mask = np.zeros_like(correlation, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    ax=ax3,
    linewidths=0.5,
    vmin=-1, vmax=1,
    xticklabels=[s.replace(".NS","") for s in correlation.columns],
    yticklabels=[s.replace(".NS","") for s in correlation.index]
)
ax3.set_title("Correlation Matrix", fontweight="bold")

# Chart 4 - Sharpe ratio
ax4 = axes[1, 1]
sharpe_colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in summary["Sharpe Ratio"]]
bars4 = ax4.bar(
    summary.index.str.replace(".NS", ""),
    summary["Sharpe Ratio"],
    color=sharpe_colors
)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("Sharpe Ratio (Risk-Adjusted Return)", fontweight="bold")
ax4.set_ylabel("Sharpe Ratio")
ax4.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars4, summary["Sharpe Ratio"]):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() - 0.08,
             f"{val}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

plt.tight_layout()
plt.savefig("portfolio_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as portfolio_dashboard.png")