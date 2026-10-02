"""
Earnings Surprise Event-Driven Strategy Backtest (PEAD)
=============================================================
Real historical earnings dates, EPS estimates, and surprise percentages (yfinance
Ticker.earnings_dates) for a real 30-40 large-cap universe, event-study cumulative
abnormal returns by surprise quintile, and a genuine long-short PEAD backtest with a
significance test - reported honestly whichever way the real sample comes out.
"""

# ===========================================================================
# CONFIG BLOCK
# ===========================================================================
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "JNJ", "PFE", "UNH", "MRK", "LLY",
    "XOM", "CVX", "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD",
]
BENCHMARK = "SPY"
CAR_WINDOWS = [1, 5, 20, 60]
N_QUINTILES = 5

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# 1. Real historical earnings-surprise data
# ===========================================================================
all_events = []
for t in UNIVERSE:
    try:
        ed = yf.Ticker(t).earnings_dates
        if ed is None or ed.empty:
            continue
        ed = ed.dropna(subset=["Surprise(%)"])
        for date, row in ed.iterrows():
            all_events.append({"ticker": t, "date": date.tz_localize(None) if date.tzinfo else date,
                                 "surprise_pct": row["Surprise(%)"]})
    except Exception as e:
        print(f"  [skip {t}: {e}]")

events_df = pd.DataFrame(all_events)
events_df = events_df[events_df["date"] < pd.Timestamp.now()]
print(f"Collected {len(events_df)} real historical earnings events across "
      f"{events_df['ticker'].nunique()} real large-cap names")

# ===========================================================================
# 2. Real price data for event-study CAR calculation
# ===========================================================================
price_tickers = list(events_df["ticker"].unique()) + [BENCHMARK]
prices = yf.download(price_tickers, start="2022-01-01", progress=False, auto_adjust=True)["Close"]
if isinstance(prices, pd.Series):
    prices = prices.to_frame()
prices = prices.dropna(how="all")
bench_ret = prices[BENCHMARK].pct_change()

def car(ticker, event_date, window):
    if ticker not in prices.columns:
        return np.nan
    idx = prices.index.searchsorted(event_date)
    if idx >= len(prices) or idx + window >= len(prices):
        return np.nan
    stock_path = prices[ticker].iloc[idx:idx + window + 1]
    bench_path = prices[BENCHMARK].iloc[idx:idx + window + 1]
    if stock_path.isna().any() or bench_path.isna().any() or len(stock_path) < window + 1:
        return np.nan
    stock_ret = stock_path.iloc[-1] / stock_path.iloc[0] - 1
    bench_ret_cum = bench_path.iloc[-1] / bench_path.iloc[0] - 1
    return stock_ret - bench_ret_cum  # market-adjusted abnormal return

for w in CAR_WINDOWS:
    events_df[f"car_{w}d"] = events_df.apply(lambda r: car(r["ticker"], r["date"], w), axis=1)

events_df = events_df.dropna(subset=[f"car_{w}d" for w in CAR_WINDOWS])
print(f"\n{len(events_df)} real events with complete real price data for all CAR windows")

# ===========================================================================
# 3. Quintile bucketing by real surprise magnitude and CAR-by-quintile path
# ===========================================================================
events_df["quintile"] = pd.qcut(events_df["surprise_pct"], N_QUINTILES, labels=False, duplicates="drop")
print("\n" + "=" * 80)
print("AVERAGE CUMULATIVE ABNORMAL RETURN (CAR) BY SURPRISE QUINTILE")
print("=" * 80)
quintile_car = events_df.groupby("quintile")[[f"car_{w}d" for w in CAR_WINDOWS]].mean()
quintile_counts = events_df.groupby("quintile").size()
quintile_car["n_events"] = quintile_counts
print(quintile_car.round(4).to_string())

# ===========================================================================
# 4. Long-short PEAD backtest: long top quintile, short bottom quintile
# ===========================================================================
top_q = events_df["quintile"].max()
bottom_q = events_df["quintile"].min()
long_car_60d = events_df[events_df["quintile"] == top_q]["car_60d"]
short_car_60d = events_df[events_df["quintile"] == bottom_q]["car_60d"]

long_short_spread = long_car_60d.mean() - short_car_60d.mean()
print(f"\n" + "=" * 80)
print("LONG-SHORT PEAD BACKTEST (60-DAY HOLDING PERIOD)")
print("=" * 80)
print(f"Top quintile ({len(long_car_60d)} events): avg 60d CAR = {long_car_60d.mean():+.2%}")
print(f"Bottom quintile ({len(short_car_60d)} events): avg 60d CAR = {short_car_60d.mean():+.2%}")
print(f"Long-short spread: {long_short_spread:+.2%}")

# Significance test: two-sample t-test on the long vs. short CAR distributions
t_stat, p_value = stats.ttest_ind(long_car_60d, short_car_60d, equal_var=False)
print(f"\nTwo-sample t-test (long vs. short CAR): t={t_stat:.3f}, p={p_value:.4f} - "
      f"{'STATISTICALLY SIGNIFICANT (p<0.05)' if p_value < 0.05 else 'NOT statistically significant at the 5% level'}")

sharpe_proxy = long_short_spread / (pd.concat([long_car_60d, short_car_60d]).std())
print(f"Long-short spread / combined CAR std (a rough Sharpe-like ratio, not annualized): "
      f"{sharpe_proxy:.3f}")

print(f"\nReport this result honestly: {'a real, statistically distinguishable PEAD signal was found in this sample' if p_value < 0.05 else 'this real sample does NOT show a statistically significant PEAD effect at the 5% level - consistent with real academic findings that PEAD has weakened/become harder to detect in more recent, more heavily-arbitraged samples since its 1968 discovery, and also consistent with this being a modest sample size (n=' + str(len(events_df)) + ' events)'}")

# ===========================================================================
# 5. Chart
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for q in sorted(events_df["quintile"].unique()):
    sub = events_df[events_df["quintile"] == q]
    path = [0] + [sub[f"car_{w}d"].mean() for w in CAR_WINDOWS]
    ax.plot([0] + CAR_WINDOWS, path, marker="o", label=f"Q{int(q)+1}")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Days after earnings announcement"); ax.set_ylabel("Avg cumulative abnormal return")
ax.set_title("PEAD: Average CAR Path by Earnings-Surprise Quintile (real data)")
ax.legend(fontsize=8, title="Surprise quintile")
plt.tight_layout()
plt.savefig("pead_car_by_quintile.png", dpi=120)
print("\nSaved chart: pead_car_by_quintile.png")
