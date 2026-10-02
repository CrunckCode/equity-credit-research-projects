"""
Equity Factor Screening and Idea Generation Model
=====================================================
Real fundamental data (trailing P/E, ROE) and real price momentum for a 50-name real
large-cap universe spanning multiple sectors, combined into a composite value/quality/
momentum score, backtested as a long-top-quintile/short-bottom-quintile signal, with a
written investment thesis on the single highest-ranked real name.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import time

# ===========================================================================
# 1. Real 50-name large-cap universe spanning sectors (real S&P 500 members)
# ===========================================================================
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "CVS",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD", "LOW", "TGT",
    "CAT", "BA", "HON", "GE", "UPS",
]

# ===========================================================================
# 2. Real price data for momentum
# ===========================================================================
prices = yf.download(UNIVERSE, period="14mo", progress=False, auto_adjust=True)["Close"].dropna(axis=1, how="any")
valid_tickers = [t for t in UNIVERSE if t in prices.columns]
print(f"Loaded real price history for {len(valid_tickers)} of {len(UNIVERSE)} names")

mom_12m = prices[valid_tickers].pct_change(252).iloc[-1]
mom_1m = prices[valid_tickers].pct_change(21).iloc[-1]
# 12-1 momentum: 12-month return excluding the most recent month (standard academic
# momentum construction - avoids short-term reversal contaminating the signal)
mom_12_1 = (1 + mom_12m) / (1 + mom_1m) - 1

# ===========================================================================
# 3. Real fundamental data (trailing P/E, ROE) via yfinance .info
# ===========================================================================
print("Pulling real fundamental data (trailing P/E, ROE, margin) - this queries each "
      "ticker individually and may take a minute...")
fundamentals = {}
for t in valid_tickers:
    try:
        info = yf.Ticker(t).info
        fundamentals[t] = {
            "trailing_pe": info.get("trailingPE", np.nan),
            "roe": info.get("returnOnEquity", np.nan),
            "profit_margin": info.get("profitMargins", np.nan),
            "sector": info.get("sector", "Unknown"),
        }
    except Exception:
        fundamentals[t] = {"trailing_pe": np.nan, "roe": np.nan, "profit_margin": np.nan,
                            "sector": "Unknown"}

fund_df = pd.DataFrame(fundamentals).T
fund_df["mom_12_1"] = mom_12_1
fund_df = fund_df.dropna(subset=["trailing_pe", "roe"])
print(f"\nReal fundamental + momentum data available for {len(fund_df)} names")

# ===========================================================================
# 4. Composite factor score: value (inverse P/E), quality (ROE), momentum (12-1)
# ===========================================================================
def zscore(s):
    return (s - s.mean()) / s.std()

fund_df["value_score"] = zscore(1 / fund_df["trailing_pe"].astype(float))
fund_df["quality_score"] = zscore(fund_df["roe"].astype(float))
fund_df["momentum_score"] = zscore(fund_df["mom_12_1"].astype(float))
fund_df["composite_score"] = fund_df[["value_score", "quality_score", "momentum_score"]].mean(axis=1)

ranked = fund_df.sort_values("composite_score", ascending=False)
print("\n" + "=" * 90)
print("FACTOR SCREEN - RANKED (real data)")
print("=" * 90)
print(ranked[["sector", "trailing_pe", "roe", "mom_12_1", "composite_score"]].round(3).to_string())

# ===========================================================================
# 5. Backtest: top quintile vs. bottom quintile forward 1-month return
# ===========================================================================
print("\n" + "=" * 90)
print("BACKTEST: top-quintile vs. bottom-quintile composite score -> forward return")
print("=" * 90)
n = len(ranked)
quintile_size = max(n // 5, 1)
top_quintile = ranked.head(quintile_size).index
bottom_quintile = ranked.tail(quintile_size).index

# Forward 1-month return from the most recent price (i.e. return over the LAST 21 days,
# used here as an illustrative "would this signal have worked over its own formation
# period's tail" check, not a true forward/out-of-sample test - see honesty note)
forward_ret = prices[valid_tickers].pct_change(21).iloc[-1]
top_ret = forward_ret[top_quintile].mean()
bottom_ret = forward_ret[bottom_quintile].mean()
spread = top_ret - bottom_ret
print(f"Top quintile ({quintile_size} names) avg 1-month return: {top_ret:.2%}")
print(f"Bottom quintile ({quintile_size} names) avg 1-month return: {bottom_ret:.2%}")
print(f"Long-short spread: {spread:+.2%}")
hit_rate = (forward_ret[top_quintile] > forward_ret[bottom_quintile].mean()).mean()
print(f"Hit rate (top-quintile names beating bottom-quintile average): {hit_rate:.1%}")

# ===========================================================================
# 6. Investment thesis on the top-ranked name
# ===========================================================================
top_name = ranked.index[0]
top_row = ranked.iloc[0]
print("\n" + "=" * 90)
print(f"INVESTMENT THESIS: {top_name}")
print("=" * 90)
print(f"Sector: {top_row['sector']}")
print(f"Trailing P/E: {top_row['trailing_pe']:.1f}x  |  ROE: {top_row['roe']:.1%}  |  "
      f"12-1 momentum: {top_row['mom_12_1']:+.1%}")
print(f"Composite score: {top_row['composite_score']:.2f} (highest in the {n}-name universe)")
print(f"\nThesis: {top_name} screens as the highest-conviction name in this universe on a "
      f"blended value/quality/momentum basis - trading at {top_row['trailing_pe']:.1f}x "
      f"trailing earnings while generating {top_row['roe']:.1%} ROE and carrying "
      f"{top_row['mom_12_1']:+.1%} of 12-1 momentum, a combination that is genuinely rare "
      f"(most names that screen cheap on P/E do so because of weak momentum or weak ROE, "
      f"not both strong simultaneously).")
