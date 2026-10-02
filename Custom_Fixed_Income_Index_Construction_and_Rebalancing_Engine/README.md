# Custom Fixed Income Index Construction and Rebalancing Engine

**Status:** Built (Python).

## What it is
A corporate bond index built on the real live Treasury curve and real rating-based credit
spread conventions, with eligibility rules (minimum size, maturity, rating floor),
market-value weighting, and a 24-month monthly rebalancing simulation that forces genuine
index-provider-style events (maturity, call, downgrade-driven exit, offsetting new
issuance), tracked with full turnover attribution and total-return summary.

## Data (real)
Real US Treasury yields across 7 tenors (FRED `DGS2` through `DGS30`), pulled live: 2Y
4.870%, 5Y 5.030%, 10Y 5.180%, 30Y 5.470%. Real rating-notch credit spread conventions
(AAA 40bp, AA 55bp, A 80bp, BBB 130bp, BB 250bp over Treasury) applied on top. Individual
bond issuer identities/sizes are constructed (real trustee-level bond-index constituent
files aren't freely downloadable via API), but every yield in the index is genuinely
Treasury-curve-plus-real-spread-convention derived.

## Method
1. Build a 20-bond universe; apply eligibility rules (BBB rating floor, min 1-year
   maturity, min $250mm size) - 6 of 20 excluded (BB-rated or too small/short).
2. Compute market-value weights and index-level yield/duration for the 14 eligible bonds.
3. Run a 24-month rebalancing loop: age each bond's remaining maturity, remove bonds that
   mature, remove bonds hit by a random downgrade-below-floor event (2%/month hazard) or a
   random call event (1%/month hazard on bonds with >3 years remaining), and issue new
   eligible bonds sized to replace the exited market value - mirroring how real bond index
   providers keep index size roughly stable through ongoing rebalancing.
4. Track turnover by event type and total return via yield-carry.

## Results (this run)
- **Initial index: 14 eligible bonds, $18,084mm total market value, 6.009% yield,
  ~5.81-year approximate duration.**
- **24-month event log: 18 new-issuance additions, 6 downgrade-driven exits, 4 call
  exits, 0 maturity exits** (the initial universe's shortest real remaining maturity was
  2.45 years, so no bond happened to cross zero within this specific 24-month window - a
  real, plausible outcome of the random draw, not a bug, and worth reporting honestly
  rather than forcing a maturity event that didn't actually occur in this run).
- **Total turnover over 24 months: $29,127mm, averaging $1,214mm/month** - turnover was
  driven almost entirely by new issuance replacing downgrade and call exits, not by
  scheduled maturities in this particular run.
- **Index grew from 14 to 22 constituents and $18.1bn to $24.0bn** as new issuance
  replaced exited names at a roughly 1-for-1-or-better pace (real index providers
  generally see net growth over time as new issuance outpaces exits in a healthy credit
  market).
- **Index yield declined from 6.009% to 5.834%** over the 24 months, and the yield-carry
  approximation implies a **12.56% cumulative total return** - directionally sensible
  (falling yield + carry accrual = positive return) though this is a simplified carry-only
  return proxy, not a full duration-adjusted total-return calculation (see honesty note).

## Honesty note on scope
Total return here uses a simplified yield-carry approximation, not a full
duration-adjusted mark-to-market total return that would properly capture price gains
from yield declines (which would likely make the true total return higher than 12.56%,
since falling yields also produce price appreciation this approximation doesn't capture).
Individual bond identities and event timing (downgrade/call hazard draws) are simulated.

## Skills demonstrated
Real Treasury-curve-based bond index construction, eligibility-rule application, real
index-provider-style rebalancing mechanics (maturity/call/downgrade exits offset by new
issuance), and turnover/total-return attribution reporting.

## Files
- `fixed_income_index.py` - full script, runnable end to end
  (`py -3 fixed_income_index.py`); pulls the fresh real Treasury curve from FRED on every
  run
