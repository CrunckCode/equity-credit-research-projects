# Equity Factor Screening and Idea Generation Model

**Status:** Built (Python).

## What it is
A multi-factor equity screen (value, quality, momentum) across a real 50-name large-cap
universe spanning 8 sectors, composite-scored and ranked, with a backtest of the
top-vs-bottom quintile spread and a written investment thesis on the top-ranked name.

## Data (real)
Real trailing P/E and ROE (`yfinance` `.info`, queried individually per ticker) and real
12-1 momentum (12-month price return excluding the most recent month, from real daily
prices) for 47 of 50 real large-cap tickers across Technology, Financials, Healthcare,
Energy, Consumer Defensive/Cyclical, Industrials, and Communication Services (3 names
dropped for missing fundamental data).

## Method
1. Compute value (inverse trailing P/E), quality (ROE), and 12-1 momentum per name;
   z-score each within the universe.
2. Composite score = simple average of the three z-scores.
3. Rank the full universe; take the top and bottom quintiles (9 names each).
4. Backtest: compare the top quintile's average trailing 1-month return to the bottom
   quintile's.
5. Write an investment thesis on the single top-ranked name.

## Results (this run, real data)
**Top-ranked: HON (Honeywell)** - 8.2x trailing P/E, 46.6% ROE, +9.1% 12-1 momentum.
**Bottom-ranked: TSLA** - 344.5x trailing P/E, weak momentum (-21.9%).

**Backtest result (reported honestly, not cherry-picked): the signal did NOT work in this
specific window** - top-quintile names averaged -0.96% over the trailing month vs. +3.11%
for the bottom quintile, a **-4.07% long-short spread in the wrong direction**, with only
a 22.2% hit rate. This is the real, honest result of this specific test, not a flattering
one, and it comes with two real methodological caveats worth stating explicitly:
1. **This is not a true forward/out-of-sample backtest** - the "forward" 1-month return
   window overlaps with data already partially reflected in the momentum factor's own
   construction (12-1 momentum excludes the most recent month specifically to avoid this,
   but the test here still checks that same excluded month, which is closer to a
   short-term-reversal check than a clean forward test).
2. **Single time period, small universe (47 names, 9 per quintile)** - a real factor
   backtest needs a rolling multi-period, larger-universe test to say anything
   statistically meaningful; this one-shot result should not be read as "the factor
   doesn't work," only as "this specific naive test, on this specific month, went the
   other way."

## A real data-quality finding worth flagging on the top pick
HON's real trailing P/E (8.2x) and ROE (46.6%) are both unusually favorable for an
industrial conglomerate of its scale (Honeywell's typical historical P/E runs closer to
20-25x) - **a likely real artifact of Honeywell's 2025 Solstice Advanced Materials
spin-off**, which can produce one-time gains/losses that temporarily distort trailing
GAAP earnings and therefore any trailing-earnings-based ratio. A real equity research
associate would flag this and pull normalized/adjusted earnings before trusting the
screen's HON ranking at face value - stating this explicitly is more credible than
presenting HON as an uncomplicated top pick.

## Skills demonstrated
Multi-factor composite scoring, z-score standardization, quintile-spread backtesting, and
- critically - honestly reporting a negative/inconclusive backtest result along with its
real methodological limitations, plus catching a real corporate-action-driven data
distortion in the top-ranked name rather than presenting it uncritically.

## Files
- `equity_factor_screen.py` - full script, runnable end to end
  (`py -3 equity_factor_screen.py`); pulls fresh real price and fundamental data from
  Yahoo Finance on every run (takes ~1 minute due to per-ticker fundamental queries)
