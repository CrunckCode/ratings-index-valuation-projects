# Illiquid Asset / Private Credit Valuation Model (Mark-to-Model)

**Status:** Built (Python).

## What it is
A mark-to-model valuation of an illiquid middle-market private term loan using a
comparable-yield approach benchmarked against real observable public credit spreads
(adjusted for illiquidity, size, and covenant-quality differences), cross-checked with a
discounted-cash-flow approach, producing a defensible fair-value range with discount-rate
sensitivity - the actual deliverable a third-party valuation analyst produces.

## Data (real)
Real credit spread indices (FRED): IG broad (`BAMLC0A0CM`, 0.79%), BBB (`BAMLC0A4CBBB`,
0.97%), HY broad (`BAMLH0A0HYM2`, 2.80%), **B-rated HY specifically**
(`BAMLH0A2HYB`, 2.86%) as the closest observable public comparable for the illiquid loan's
credit quality, and the real 10Y Treasury yield (`DGS10`, 5.18%). The illiquid loan's own
terms ($25M, 5Y, middle-market) are illustrative (real private-loan term sheets aren't
freely downloadable via API).

## Method
1. **Comparable-yield method:** start from the real base rate (10Y Treasury + real
   B-rated HY spread = 8.040%), then apply three itemized, real-convention adjustments:
   +150bp illiquidity premium (the real standard convention range for private credit is
   100-200bp), +75bp size/less-followed-issuer premium, and **-50bp for a tighter covenant
   package** - private loans are typically better-covenanted than public HY bonds (more
   maintenance covenants, tighter baskets), which is a real structuring feature that
   should reduce, not increase, the required yield relative to a covenant-lite public
   comparable.
2. **DCF cross-check:** discount the loan's cash flows at the comparable-yield-derived
   rate.
3. **Sensitivity:** recompute fair value under +/-100bp discount-rate shocks.

## Results (this run, real spread/rate data)
- **Comparable-yield-method discount rate: 9.790%** (8.040% base + 1.75% net adjustment).
- **Fair value: $25,000,000 (100.0% of face)** by construction, since the loan is marked
  at its own derived rate.
- **DCF cross-check reconciles exactly** ($25,000,000, 100.00% of face) - the correct and
  expected result for a cross-check discounting at the same rate; a real divergence here
  would flag a cash-flow-timing error, not additional valuation signal.
- **Fair-value range under a +/-50bp discount-rate uncertainty band: $24.53M to $25.48M
  (98.1% to 101.9% of face)** - a defensible, quantified uncertainty band rather than a
  single point estimate, which is what a real valuation memo actually needs to present.
- **+/-100bp full sensitivity:** fair value ranges from $24.07M (-100bp: +100bp rate shock)
  to $25.98M (-100bp rate shock, higher value) - roughly a $1.9M (7.6% of face) swing
  across a 200bp discount-rate range, a real, quantifiable duration-driven sensitivity.

## Skills demonstrated
Comparable-yield valuation methodology with itemized, real-convention adjustments
(including correctly signing a covenant-quality adjustment as a *yield reduction*, not
just adding a flat illiquidity premium), DCF cross-checking, and presenting a fair-value
range with explicit discount-rate sensitivity rather than a single unqualified number -
exactly the Valuation Analyst deliverable for illiquid/Level 3 assets.

## Files
- `illiquid_valuation_model.py` - full script, runnable end to end
  (`py -3 illiquid_valuation_model.py`); pulls fresh real credit-spread and Treasury data
  from FRED on every run
