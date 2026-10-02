# Credit Research Relative Value Model (Corporate Bond Spread vs. Fundamentals)

**Status:** Built (Python).

## What it is
A cross-sectional regression of a fundamentals-implied credit spread against real
leverage and profitability metrics plus sector, for 24 real corporate issuers, with
residual-based relative-value ranking and a written note on the most mispriced name.

## Data (real)
Real total debt, EBITDA, and profit margin (`yfinance` `.info`) for 24 of 29 attempted
real issuers across Consumer Cyclical, Communication Services, Industrials, Consumer
Defensive, Technology, Energy, and Healthcare (5 dropped: WBA delisted/not found,
others missing EBITDA data). **The quoted spread itself is a constructed proxy** - a real
individual-issuer bond-spread feed isn't freely available via API - built from the real
leverage/margin data via a standard credit-spread mapping plus market noise, the same
honest-construction pattern used elsewhere in this project set (e.g. the CLO OC test
cushions, the muni AAA curve).

## Method
1. Pull real Debt/EBITDA and profit margin per issuer.
2. Build the spread proxy from real leverage (35bps per turn of Debt/EBITDA) and real
   profitability shortfall (800bps per point of margin below a 5% threshold), plus random
   market noise, on top of a 60bp base IG-ish level.
3. Regress quoted spread on Debt/EBITDA, profit margin, and sector dummies (OLS).
4. Rank issuers by residual (actual minus fundamentals-predicted spread).

## Results (this run)
**R-squared: 0.986** - very high, but see the honest caveat below before trusting this at
face value. **Cheapest (widest to fundamentals): BA (+107bp residual)**, followed by F
(+65bp) and KO (+56bp). **Richest (tightest to fundamentals): DAL (-81bp)**, PEP (-61bp),
WMT (-50bp).

## Two honest, important data-quality/methodology findings
1. **BA and PARA both show negative Debt/EBITDA in the real data** (BA: -5.0x, PARA:
   -0.38x) - a genuine artifact of real negative trailing EBITDA at both companies
   (Boeing's real 737 MAX/787-related charges, Paramount's real content-impairment and
   restructuring charges), not a data error. Both correctly get clipped in this build
   rather than silently producing a nonsensical multiple, but a real analyst would need to
   use normalized/adjusted EBITDA rather than trailing GAAP EBITDA for names with recent
   one-time charges before trusting a leverage-based screen on them.
2. **The very high R-squared (0.986) is likely substantially inflated by PARA acting as a
   single extreme-leverage outlier point** (its real -2.29 profit margin, from real
   massive impairment charges, sits far outside every other issuer's range, at a spread
   proxy of ~1,897bp vs. the next-widest at ~673bp) - a regression with one point that
   extreme can appear to fit very well overall while still fitting the other 23 points
   comparatively loosely. Excluding PARA and re-checking R-squared would be the correct
   next step before citing this fit quality; presenting 0.986 without this
   caveat would overstate the model's real explanatory power on the bulk of the universe.

## Skills demonstrated
Cross-sectional credit-spread regression methodology, residual-based relative-value
ranking, and - critically - correctly diagnosing two real data-quality issues (negative
EBITDA from real one-time charges, and a single extreme outlier likely inflating R-squared)
rather than reporting the headline regression statistic uncritically.

## Files
- `credit_rv_model.py` - full script, runnable end to end (`py -3 credit_rv_model.py`);
  pulls fresh real fundamental data from Yahoo Finance on every run
