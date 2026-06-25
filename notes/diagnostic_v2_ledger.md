# Diagnostic v2 Ledger

## Data source migration

**2026-06-13**: Odds source migrated from VegasInsider HTML scraping to 
The Odds API median consensus across US bookmakers.
- Filter calibration (MIN_EDGE_DECIMAL, MAX_EDGE_DECIMAL, MIN_FAVORITE_EDGE_DECIMAL, 
  MAX_FAVORITE_ODDS, MIN_MODEL_PROB, etc.) was tuned on VegasInsider prices.
- Post-migration price source is slightly different (median across all US books 
  vs single-source consensus). Edge distributions may shift modestly.
- Diagnostic v2 should compare pre-migration (May 3 - June 6) vs 
  post-migration (June 13+) edge distributions and filter pass rates 
  before drawing any threshold-tuning conclusions.

## Filtered pick outcomes ledger (running)

### MAX_EDGE_DECIMAL=0.17 (FLAGGED_HIGH_EDGE)
- 6/1 CHW +144 (24% edge): FILTER CORRECT (CHW lost)

### MAX_FAVORITE_ODDS=-160 (SKIP_HEAVY_CHALK)
- 5/30 NYY -163: FILTER CORRECT (NYY lost)
- 6/2 NYY -199: FILTER CORRECT (NYY lost)
- 6/3 NYY -168: FILTER CORRECT (NYY lost)
- Net validation: 3-for-3, $30 protected

### MIN_FAVORITE_EDGE_DECIMAL=0.10 (SKIP_FAVORITE_LOW_EDGE)
- 5/31 NYY -156 (7.3% edge): FILTER WRONG (NYY won, would have profited $6.41)
- 6/4 NYY -156 (8.7% edge): FILTER WRONG (NYY won, would have profited $6.41)
- Net validation: 0-for-2 on NYY only; small sample, single team

### MIN_MODEL_PROB=0.42 (SKIP_LOW_PROB)
- 6/6 COL +209 (model prob 0.323): TBD
- 6/6 LAA +258 (model prob 0.317): TBD

## Tier subset records (post-May 18 filter, pre-migration)

### 5-10% UNDERDOG SUBSET
4-6 across 10 picks. Primary candidate for diagnostic investigation.

### 10-15% EDGE TIER  
Consistent positive performer (workhorse tier).

### 15%+ EDGE TIER
4-4 across 8 picks. Stalled in middle ground, small sample.

## Diagnostic v2 priorities (to investigate late June)

1. Does the 5-10% underdog subset justify its own filter? Need 50+ picks total.
2. Is MIN_FAVORITE_EDGE_DECIMAL=0.10 set too aggressively? Need non-NYY data.
3. Are MAX_FAVORITE_ODDS=-160 and MAX_EDGE_DECIMAL=0.17 holding under the new price source?
4. Has BLEND_2026_WEIGHT progression time arrived? Pitcher IPs now 60-95 range.
5. Run parameter sweep on LOGISTIC_K, PROB_SHRINK_TOWARD_MARKET against archived data.
