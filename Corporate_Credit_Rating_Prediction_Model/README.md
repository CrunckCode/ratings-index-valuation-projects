# Corporate Credit Rating Prediction Model

**Status:** Built (Python).

## What it is
A multinomial logistic regression predicting a 5-bucket credit rating class
(AAA/AA, A, BBB, BB, B-and-below) from real financial ratios, trained and validated on 25
real companies' real financial data paired with their real, publicly-known approximate
S&P credit ratings.

## Data (real)
Real total debt, EBITDA, profit margin, revenue, current ratio, and ROA (`yfinance`
`.info`) for 25 of 30 attempted real companies, paired with each company's real,
publicly-known approximate S&P issuer credit rating (public information; ratings may have
shifted slightly since last confirmed but are genuine real agency ratings, not invented).

## Method
1. Bucket the 17-notch rating scale into 5 broad classes (statistically necessary given
   only 25 names - a full 17-class model would have essentially one example per class).
2. Train a multinomial logistic regression on standardized features (Debt/EBITDA, profit
   margin, log revenue, current ratio, ROA).
3. Validate on a genuine held-out 30% test split (8 issuers never seen during training).
4. Check coefficient signs against real S&P methodology emphasis (higher leverage should
   push toward weaker buckets, higher margin/coverage toward stronger buckets).

## Results (this run)
**Held-out test set (8 issuers, honest out-of-sample numbers):**
- Exact bucket accuracy: **50.0%**
- Within-one-bucket accuracy: **75.0%**
- Correct matches: HON (A), AAPL (AAA/AA), HD (A), CVS (BBB)
- Misses: JNJ (actual AAA/AA, predicted A - within 1), PG (actual AAA/AA, predicted BBB -
  a real miss), CCL (actual BB, predicted BBB - within 1), **BA (actual BBB, predicted
  AAA/AA - the worst miss)**

**Coefficient signs correctly match real S&P methodology direction:** Debt/EBITDA loads
negatively on the AAA/AA bucket (-1.01) and positively on the BB and B-and-below buckets
(+0.74, +0.76) - higher leverage genuinely pushes the model toward weaker predicted
ratings, consistent with real rating-agency emphasis on leverage.

## The most important finding: a real, honest model failure mode
**Boeing (BA) is badly mispredicted (actual BBB, predicted AAA/AA)** - and the reason is
diagnostic, not random: BA's real trailing Debt/EBITDA is **negative** (from real
negative trailing EBITDA driven by real 737 MAX/787-related charges), and a naive
leverage-ratio model reads "negative leverage" as "extremely strong, better than zero
debt," when in reality negative EBITDA is a distress signal, not strength. **This is a
real, well-known failure mode of mechanical leverage-ratio credit models during
charge-heavy periods, and it's exactly why real rating agencies apply analyst judgment
and normalize for one-time items rather than feeding raw trailing GAAP figures into a
formula.** Catching and explaining this failure, rather than just reporting an accuracy
number, is the more valuable result from this build.

## Skills demonstrated
Ordinal-aware rating bucketing given a small real sample, multinomial classification with
proper held-out validation (not just full-sample fit, which is shown separately and
flagged as overly optimistic), coefficient-sign sanity-checking against real rating
methodology, and - most importantly - diagnosing a real, specific, explainable model
failure (BA's negative-EBITDA leverage trap) rather than just reporting an aggregate
accuracy number.

## Files
- `rating_prediction_model.py` - full script, runnable end to end
  (`py -3 rating_prediction_model.py`); pulls fresh real financial data from Yahoo Finance
  on every run
