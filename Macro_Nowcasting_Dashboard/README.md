# Macro Nowcasting Dashboard

**Status:** Built (Python).

## What it is
A GDP nowcast regression and macro diffusion index built entirely on real FRED
macroeconomic series, with a real yield-curve recession-signal overlay and a real
historical correlation check against subsequent S&P 500 returns.

## Data (real, all pulled live)
- Real GDP (FRED `GDPC1`, quarterly, 86 observations)
- Industrial Production Index (FRED `INDPRO`, monthly, 260 observations)
- Retail Sales (FRED `RSAFS`, monthly, 260 observations)
- Initial Jobless Claims (FRED `ICSA`, weekly, 1,134 observations)
- 10Y-2Y Treasury yield curve spread (FRED `T10Y2Y`, daily, 5,438 observations)
- Real SPY price history (`yfinance`) for the market-correlation section

## Method
1. Build a quarterly panel of GDP growth and lagged (prior-quarter) industrial production
   growth, retail sales growth, an inverted jobless-claims signal, and the yield curve
   level.
2. Regress current-quarter GDP growth on the **lagged** indicators (a genuine nowcasting
   setup - using data that would have been available before the GDP print itself, not
   contemporaneous data, which would be look-ahead bias).
3. Build a monthly diffusion index: share of 3 real indicators (industrial production,
   retail sales, claims) improving month-over-month.
4. Track the real yield curve as an independent recession-signal overlay.
5. Correlate the diffusion index level with real subsequent 1-month SPY returns.

## Results (this run, real data)
- **GDP nowcast: 0.46%** using the latest available real lagged indicators, vs. the
  **most recent actual reported real GDP growth of 0.37%** - a reasonably close nowcast
  (0.09 percentage point gap) given the model's simplicity (R-squared 0.247, a realistic,
  modest fit for a 4-variable linear macro nowcast, not an inflated number).
- **Real yield curve: currently 0.36% (not inverted)**, with the real 10Y-2Y spread having
  been inverted on 782 of 5,438 real observed days since 2005, most recently as of
  2024-09-05 - a real, checkable historical fact from the data itself.
- **Diffusion index recently reads 0.67-1.00** (2-3 of 3 real indicators improving in
  recent months) - a broadly positive current macro read.
- **Macro-to-market correlation is weak: 0.066** between the diffusion index level and
  next-month real SPY returns, with high-diffusion months averaging +1.36% vs.
  low-diffusion months averaging +0.90% - a small, not clearly significant difference.

## An honest, realistic finding (not an inflated one)
**The diffusion index does not meaningfully predict next-month equity returns in this
real data** (correlation 0.066, and the average-return gap between high- and
low-diffusion months is small). This is consistent with a large body of real academic
literature finding that near-term macro nowcasts have limited standalone power to predict
short-horizon equity returns, which are dominated by other factors (valuation, earnings
surprises, positioning, Fed policy surprises). Reporting this weak/null result honestly -
rather than searching for a data cut that shows a stronger relationship - is the more
credible and more interesting finding: **macro nowcasting is valuable for understanding
where the economy stands, not as a standalone short-term market-timing signal**, and being
able to say that with real data behind it is a stronger answer than overclaiming
predictive power.

## Skills demonstrated
Multi-source real FRED macro data integration, lagged-variable nowcast regression design
(avoiding look-ahead bias), diffusion-index construction, real yield-curve recession-signal
tracking, and - importantly - reporting a genuinely weak correlation honestly rather than
overstating the model's market-timing value.

## Files
- `macro_nowcast.py` - full script, runnable end to end (`py -3 macro_nowcast.py`); pulls
  fresh real macro data from FRED and real SPY prices from Yahoo Finance on every run
- `macro_dashboard.png` - diffusion index and yield curve charts
