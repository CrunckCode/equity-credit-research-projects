"""
Macro Nowcasting Dashboard
==============================
Real leading economic indicators (FRED): initial jobless claims, retail sales, industrial
production, and the 10Y-2Y yield curve spread, combined into a diffusion index and a
simplified GDP nowcast regression, then correlated against real S&P 500 returns to build
a "what typically happens next" market-implication table.
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TODAY = datetime.date.today()
START = datetime.datetime(2005, 1, 1)

# ===========================================================================
# 1. Real macro indicators (FRED)
# ===========================================================================
series = {
    "GDP": "GDPC1",              # Real GDP, quarterly
    "IndProd": "INDPRO",         # Industrial Production Index, monthly
    "RetailSales": "RSAFS",      # Retail Sales, monthly
    "InitialClaims": "ICSA",     # Initial jobless claims, weekly
    "YieldCurve": "T10Y2Y",      # 10Y-2Y Treasury spread, daily
}
data = {}
for name, code in series.items():
    df = web.DataReader(code, "fred", start=START).dropna()
    data[name] = df.iloc[:, 0]
    print(f"Real {name} (FRED {code}): {len(df)} observations, "
          f"latest = {df.iloc[-1, 0]:.2f} as of {df.index[-1].date()}")

# ===========================================================================
# 2. Resample everything to quarterly for the GDP nowcast regression
# ===========================================================================
gdp_growth = data["GDP"].pct_change(1) * 100  # quarterly annualized-ish % change
gdp_growth.index = gdp_growth.index.to_period("Q").to_timestamp("Q")  # align to quarter-END, matching the resample("QE") indices below
ind_prod_q = data["IndProd"].resample("QE").mean().pct_change(1) * 100
retail_q = data["RetailSales"].resample("QE").mean().pct_change(1) * 100
claims_q = data["InitialClaims"].resample("QE").mean().pct_change(1) * -100  # invert: fewer claims = positive signal
yield_curve_q = data["YieldCurve"].resample("QE").mean()

nowcast_df = pd.DataFrame({
    "gdp_growth": gdp_growth, "ind_prod_growth": ind_prod_q, "retail_growth": retail_q,
    "claims_signal": claims_q, "yield_curve": yield_curve_q,
}).dropna()
print(f"\nBuilt quarterly nowcast panel: {len(nowcast_df)} real quarters "
      f"({nowcast_df.index[0].date()} to {nowcast_df.index[-1].date()})")

# ===========================================================================
# 3. Simplified GDP nowcast: regress GDP growth on LAGGED indicators
#    (using last quarter's indicator levels to nowcast this quarter's growth,
#    consistent with how a real nowcast uses higher-frequency data available
#    before GDP itself is reported)
# ===========================================================================
X = nowcast_df[["ind_prod_growth", "retail_growth", "claims_signal", "yield_curve"]].shift(1).dropna()
y = nowcast_df["gdp_growth"].reindex(X.index)
valid = y.notna()
X, y = X[valid], y[valid]

from numpy.linalg import lstsq
X_with_const = np.column_stack([np.ones(len(X)), X.values])
coefs, _, _, _ = lstsq(X_with_const, y.values, rcond=None)
y_pred = X_with_const @ coefs
r2 = 1 - np.sum((y.values - y_pred)**2) / np.sum((y.values - y.values.mean())**2)

print("\n" + "=" * 70)
print("SIMPLIFIED GDP NOWCAST REGRESSION (lagged indicators -> current-quarter GDP growth)")
print("=" * 70)
print(f"R-squared: {r2:.3f}")
print(f"Coefficients: const={coefs[0]:.3f}, ind_prod={coefs[1]:.3f}, "
      f"retail={coefs[2]:.3f}, claims_signal={coefs[3]:.3f}, yield_curve={coefs[4]:.3f}")

# Current nowcast using latest available lagged indicators
latest_inputs = nowcast_df[["ind_prod_growth", "retail_growth", "claims_signal", "yield_curve"]].iloc[-1]
current_nowcast = coefs[0] + coefs[1:] @ latest_inputs.values
print(f"\nCURRENT GDP GROWTH NOWCAST (using latest real indicators): {current_nowcast:.2f}%")
print(f"Most recent actual reported GDP growth: {nowcast_df['gdp_growth'].iloc[-1]:.2f}%")

# ===========================================================================
# 4. Diffusion index: share of indicators improving month-over-month (monthly frequency)
# ===========================================================================
def to_month_end(s):
    s = s.copy()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s

monthly_ind_prod = to_month_end(data["IndProd"].pct_change(1))
monthly_retail = to_month_end(data["RetailSales"].pct_change(1))
monthly_claims = to_month_end(-data["InitialClaims"].resample("ME").mean().pct_change(1))
diffusion_df = pd.DataFrame({"ind_prod": monthly_ind_prod, "retail": monthly_retail,
                               "claims": monthly_claims}).dropna()
diffusion_index = (diffusion_df > 0).mean(axis=1)
print(f"\nDIFFUSION INDEX (share of 3 real indicators improving, latest 6 months):")
print(diffusion_index.tail(6).round(2).to_string())

# ===========================================================================
# 5. Real yield curve as a recession signal overlay
# ===========================================================================
yc_daily = data["YieldCurve"]
inverted_days = (yc_daily < 0).sum()
print(f"\nReal 10Y-2Y yield curve: currently {yc_daily.iloc[-1]:.2f}%, "
      f"inverted on {inverted_days} of {len(yc_daily)} real observed days since {START.year}")
last_inversion = yc_daily[yc_daily < 0].index[-1] if (yc_daily < 0).any() else None
print(f"Most recent real inversion date: {last_inversion.date() if last_inversion is not None else 'None in sample'}")

# ===========================================================================
# 6. Correlate diffusion index moves with REAL subsequent SPY returns
# ===========================================================================
spy = yf.download("SPY", start=START, progress=False, auto_adjust=True)["Close"]
spy_series = spy["SPY"] if hasattr(spy, "columns") else spy
spy_monthly = spy_series.resample("ME").last().pct_change(1)

merged = pd.DataFrame({"diffusion": diffusion_index}).resample("ME").last()
merged["spy_fwd_1m"] = spy_monthly.shift(-1).reindex(merged.index)
merged = merged.dropna()

high_diffusion = merged[merged["diffusion"] >= 0.67]["spy_fwd_1m"]
low_diffusion = merged[merged["diffusion"] <= 0.33]["spy_fwd_1m"]

print("\n" + "=" * 70)
print("MACRO-TO-MARKET HISTORICAL CORRELATION (real SPY returns)")
print("=" * 70)
print(f"Months with diffusion index >= 0.67 (broad improvement): {len(high_diffusion)} "
      f"months, average next-month real SPY return: {high_diffusion.mean():.2%}")
print(f"Months with diffusion index <= 0.33 (broad deterioration): {len(low_diffusion)} "
      f"months, average next-month real SPY return: {low_diffusion.mean():.2%}")
corr = merged["diffusion"].corr(merged["spy_fwd_1m"])
print(f"Correlation (diffusion index level vs. next-month SPY return): {corr:.3f}")

# ===========================================================================
# 7. Chart
# ===========================================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].plot(diffusion_index.index, diffusion_index.values, color="steelblue")
axes[0].axhline(0.5, linestyle="--", color="gray")
axes[0].set_title("Real Macro Diffusion Index (3 real indicators)")
axes[1].plot(yc_daily.index, yc_daily.values, color="firebrick")
axes[1].axhline(0, linestyle="--", color="black")
axes[1].set_title("Real 10Y-2Y Treasury Yield Curve Spread")
plt.tight_layout()
plt.savefig("macro_dashboard.png", dpi=120)
print("\nSaved chart: macro_dashboard.png")
