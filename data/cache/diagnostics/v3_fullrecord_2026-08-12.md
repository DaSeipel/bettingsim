# MLB Diagnostic v3 — FULL RECORD (2026-08-12)

Read-only extension of v3 over the **dashboard MLB Record** source (not the mid-season decision_log-only sample). Flat stake **$10**. No model params/thresholds/math changed.

## 0. Source inspection (before analysis)

- **Source:** `data/espn.db → play_history (Streamlit MLB Record tab)`
- **Dashboard filter:** sport=MLB, spread_or_total≈-999 (moneyline), result in {W,L}, my_edge_pct≥5, date_generated≥2026-05-03, deduped on natural key keep=last
- **Raw MLB rows in DB:** 278 (deduped: 278)
- **Dashboard view n:** **215** (2026-05-03 → 2026-08-11)
- **Per-play fields present:** edge=`my_edge_pct` (yes, 0 nulls), model_prob=`my_probability` (yes, 0 nulls), odds=`market_odds_at_time` (yes), result=`result` (yes)
- **Edge recompute:** not needed — `my_edge_pct` stored on every dashboard-view row (unreconstructable: **0**)
- **IP coverage:** min-starter-IP from decision_log join or current `pitcher_stats.csv` via reasoning_summary SP parse; **missing IP on 2/215 rows** (flagged in §6)
- **Reconciliation:** current full view = **215 picks, 103-112, P/L $+71.75, ROI +3.3%**. Checkpoint through 2026-07-31 = **186 picks, P/L $+114.45, ROI +6.2%** (matches the remembered ~186 / +$114.45 / +6.2%). Post-July results moved the live total to the current figure.

## 1. Edge-tier breakdown (dashboard played picks)

### (A) Full record (dashboard filter, from 2026-05-03)

| Tier | n | W-L | Win% | Wagered | P/L | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| 5-10% | n=105 | 44-61 | 41.9% | $1050 | $-87.87 | -8.4% |
| 10-15% | n=83 | 46-37 | 55.4% | $830 | $+143.87 | +17.3% |
| 15%+ | n=27 | 13-14 | 48.1% | $270 | $+15.76 | +5.8% |

**Overall:** n=215 | 103-112 | win% 47.9% | wagered $2150 | P/L $+71.75 | ROI +3.3%

**Finding:** Full-record tiers on stored `my_edge_pct` (n=215). 5-10% ROI -8.4% (n=105); 10-15% ROI +17.3% (n=83); 15%+ ROI +5.8% (n=27). Adequate-n tiers can inform monitoring; do not change params from post_odds_fix slices with n<20.


### (B) post_odds_fix only (date ≥ 2026-08-04)

| Tier | n | W-L | Win% | Wagered | P/L | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| 5-10% | n=7 ⚠️ insufficient — do not conclude | 3-4 | 42.9% | $70 | $-0.30 | -0.4% |
| 10-15% | n=5 ⚠️ insufficient — do not conclude | 2-3 | 40.0% | $50 | $-6.30 | -12.6% |
| 15%+ | n=3 ⚠️ insufficient — do not conclude | 0-3 | 0.0% | $30 | $-30.00 | -100.0% |

**Overall:** n=15 ⚠️ insufficient — do not conclude | 5-10 | win% 33.3% | wagered $150 | P/L $-36.60 | ROI -24.4%

**Finding:** Sample n=15 ⚠️ insufficient — do not conclude. Too thin for tier conclusions — hold thresholds.



## 2. MIN_MODEL_PROB analysis

### (A) Full record (dashboard filter, from 2026-05-03)

| model_prob | n | W-L | Win% | ROI |
| --- | --- | --- | --- | --- |
| 0.42–0.44 | n=13 ⚠️ insufficient — do not conclude | 8-5 | 61.5% | +59.2% |
| 0.44–0.46 | n=20 | 8-12 | 40.0% | -4.3% |
| 0.46–0.50 | n=60 | 26-34 | 43.3% | -2.5% |
| 0.50+ | n=122 | 61-61 | 50.0% | +1.5% |

**Baseline:** n=215 | 103-112 | win% 47.9% | wagered $2150 | P/L $+71.75 | ROI +3.3%
- Floor ≥0.44: keep n=202 | 95-107 | win% 47.0% | wagered $2020 | P/L $-5.25 | ROI -0.3%; remove 13 picks (8-5, ROI +59.2% on removed)
- Floor ≥0.45: keep n=195 | 93-102 | win% 47.7% | wagered $1950 | P/L $+17.05 | ROI +0.9%; remove 20 picks (10-10, ROI +27.4% on removed)

**Finding:** 0.42–0.44 bucket n=13 ⚠️ insufficient — do not conclude; cannot justify raising MIN_MODEL_PROB. Hold at 0.42.


### (B) post_odds_fix only (date ≥ 2026-08-04)

| model_prob | n | W-L | Win% | ROI |
| --- | --- | --- | --- | --- |
| 0.42–0.44 | n=1 ⚠️ insufficient — do not conclude | 1-0 | 100.0% | +152.0% |
| 0.44–0.46 | n=2 ⚠️ insufficient — do not conclude | 1-1 | 50.0% | +19.0% |
| 0.46–0.50 | n=5 ⚠️ insufficient — do not conclude | 0-5 | 0.0% | -100.0% |
| 0.50+ | n=7 ⚠️ insufficient — do not conclude | 3-4 | 42.9% | -8.0% |

**Baseline:** n=15 ⚠️ insufficient — do not conclude | 5-10 | win% 33.3% | wagered $150 | P/L $-36.60 | ROI -24.4%
- Floor ≥0.44: keep n=14 ⚠️ insufficient — do not conclude | 4-10 | win% 28.6% | wagered $140 | P/L $-51.80 | ROI -37.0%; remove 1 picks (1-0, ROI +152.0% on removed)
- Floor ≥0.45: keep n=13 ⚠️ insufficient — do not conclude | 3-10 | win% 23.1% | wagered $130 | P/L $-65.60 | ROI -50.5%; remove 2 picks (2-0, ROI +145.0% on removed)

**Finding:** n=15 ⚠️ insufficient — do not conclude. Cannot conclude on raising MIN_MODEL_PROB — need more data.



## 3. Favorite vs dog split

### (A) Full record (dashboard filter, from 2026-05-03)

| Side | n | W-L | Win% | P/L | ROI |
| --- | --- | --- | --- | --- | --- |
| Favorites (odds<0) | n=42 | 23-19 | 54.8% | $+4.05 | +1.0% |
| Dogs (odds>0) | n=173 | 80-93 | 46.2% | $+67.70 | +3.9% |
#### Favorites by MIN_FAVORITE_EDGE bucket

| Fav edge | n | W-L | Win% | ROI |
| --- | --- | --- | --- | --- |
| 0.10–0.12 | n=14 ⚠️ insufficient — do not conclude | 12-2 | 85.7% | +57.0% |
| 0.12–0.15 | n=8 ⚠️ insufficient — do not conclude | 4-4 | 50.0% | -7.4% |
| 0.15+ | n=7 ⚠️ insufficient — do not conclude | 4-3 | 57.1% | +9.9% |
#### MAX_FAVORITE_ODDS boundary cases

| Boundary | n | W-L | Win% | ROI | Notes |
| --- | --- | --- | --- | --- | --- |
| Played fav −155 to −160 | n=1 ⚠️ insufficient — do not conclude | 1-0 | 100.0% | +64.1% | inside MAX_FAVORITE_ODDS |
| Played fav exactly −160 | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — | at boundary |
| SKIP_HEAVY_CHALK −161…−170 (hypo) | n=3 ⚠️ insufficient — do not conclude | 2-1 | 66.7% | +6.3% | from decision_log only (mid-season+) |
**Finding:** favorites n=42 | 23-19 | win% 54.8% | wagered $420 | P/L $+4.05 | ROI +1.0%; dogs n=173 | 80-93 | win% 46.2% | wagered $1730 | P/L $+67.70 | ROI +3.9%. Boundary −155…−160 n=1 ⚠️ insufficient — do not conclude; chalk skips n=3 ⚠️ insufficient — do not conclude. Do not move MAX_FAVORITE_ODDS unless boundary n≥20.


### (B) post_odds_fix only (date ≥ 2026-08-04)

| Side | n | W-L | Win% | P/L | ROI |
| --- | --- | --- | --- | --- | --- |
| Favorites (odds<0) | n=1 ⚠️ insufficient — do not conclude | 0-1 | 0.0% | $-10.00 | -100.0% |
| Dogs (odds>0) | n=14 ⚠️ insufficient — do not conclude | 5-9 | 35.7% | $-26.60 | -19.0% |
#### Favorites by MIN_FAVORITE_EDGE bucket

| Fav edge | n | W-L | Win% | ROI |
| --- | --- | --- | --- | --- |
| 0.10–0.12 | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — |
| 0.12–0.15 | n=1 ⚠️ insufficient — do not conclude | 0-1 | 0.0% | -100.0% |
| 0.15+ | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — |
#### MAX_FAVORITE_ODDS boundary cases

| Boundary | n | W-L | Win% | ROI | Notes |
| --- | --- | --- | --- | --- | --- |
| Played fav −155 to −160 | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — | inside MAX_FAVORITE_ODDS |
| Played fav exactly −160 | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — | at boundary |
| SKIP_HEAVY_CHALK −161…−170 (hypo) | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — | from decision_log only (mid-season+) |
**Finding:** Favorites n=1 ⚠️ insufficient — do not conclude, dogs n=14 ⚠️ insufficient — do not conclude — too thin. Hold favorite filters.



## 4. MAX_EDGE analysis

### (A) Full record (dashboard filter, from 2026-05-03)

| Bucket | n | W-L | Win% | P/L | ROI |
| --- | --- | --- | --- | --- | --- |
| PLAYED 0.13–0.15 | n=22 | 12-10 | 54.5% | $+37.31 | +17.0% |
| PLAYED 0.15–0.17 | n=27 | 13-14 | 48.1% | $+15.76 | +5.8% |
| FLAGGED_HIGH_EDGE (>0.17 hypo) | n=21 | 7-14 | 33.3% | $-41.50 | -19.8% |

**Tier-3 (15%+ played) reconciliation:** n=27 | 13-14 | win% 48.1% | wagered $270 | P/L $+15.76 | ROI +5.8%

**Ledger vs tier disagreement:** YES — signs differ

FLAGGED drivers:

| date | pick | odds | edge | result |
| --- | --- | --- | --- | --- |
| 2026-07-27 | Seattle Mariners | 2121 | 9.190 | L |
| 2026-08-10 | Atlanta Braves | 375 | 1.196 | L |
| 2026-08-04 | Washington Nationals | 353 | 0.976 | L |
| 2026-08-04 | Cleveland Guardians | 285 | 0.739 | L |
| 2026-08-03 | Washington Nationals | 189 | 0.462 | L |
| 2026-06-29 | Pittsburgh Pirates | 189 | 0.427 | W |
| 2026-08-10 | St. Louis Cardinals | 189 | 0.347 | L |
| 2026-07-26 | New York Yankees | 159 | 0.279 | L |
| 2026-07-07 | Arizona Diamondbacks | 131 | 0.267 | L |
| 2026-08-04 | San Francisco Giants | 170 | 0.263 | L |
| 2026-06-26 | Texas Rangers | 150 | 0.257 | W |
| 2026-07-17 | Los Angeles Angels | 162 | 0.234 | L |

PLAYED 15%+ sample:

| date | pick | odds | edge | result |
| --- | --- | --- | --- | --- |
| 2026-05-28 | Chicago Cubs | 144 | 0.170 | W |
| 2026-07-10 | Kansas City Royals | 131 | 0.169 | L |
| 2026-08-05 | Detroit Tigers | 130 | 0.169 | L |
| 2026-06-14 | Chicago Cubs | 113 | 0.168 | L |
| 2026-07-11 | Boston Red Sox | 125 | 0.168 | W |
| 2026-07-10 | Boston Red Sox | 125 | 0.168 | W |
| 2026-06-25 | Chicago Cubs | -105 | 0.165 | W |
| 2026-08-04 | Los Angeles Dodgers | 107 | 0.164 | L |
| 2026-05-05 | Atlanta Braves | 123 | 0.163 | W |
| 2026-07-09 | New York Yankees | 130 | 0.163 | W |
| 2026-08-01 | San Francisco Giants | 131 | 0.162 | L |
| 2026-06-22 | New York Yankees | -110 | 0.162 | L |

**Finding:** PLAYED 15%+ n=27 | 13-14 | win% 48.1% | wagered $270 | P/L $+15.76 | ROI +5.8%; FLAGGED hypo n=21 | 7-14 | win% 33.3% | wagered $210 | P/L $-41.50 | ROI -19.8%. Sign conflict — inspect drivers before changing MAX_EDGE. Weigh whether flagged hypo ROI supports keeping the 0.17 cap.


### (B) post_odds_fix only (date ≥ 2026-08-04)

| Bucket | n | W-L | Win% | P/L | ROI |
| --- | --- | --- | --- | --- | --- |
| PLAYED 0.13–0.15 | n=1 ⚠️ insufficient — do not conclude | 1-0 | 100.0% | $+12.30 | +123.0% |
| PLAYED 0.15–0.17 | n=3 ⚠️ insufficient — do not conclude | 0-3 | 0.0% | $-30.00 | -100.0% |
| FLAGGED_HIGH_EDGE (>0.17 hypo) | n=7 ⚠️ insufficient — do not conclude | 1-6 | 14.3% | $-45.10 | -64.4% |

**Tier-3 (15%+ played) reconciliation:** n=3 ⚠️ insufficient — do not conclude | 0-3 | win% 0.0% | wagered $30 | P/L $-30.00 | ROI -100.0%

**Ledger vs tier disagreement:** No clear sign conflict (or thin samples)

FLAGGED drivers:

| date | pick | odds | edge | result |
| --- | --- | --- | --- | --- |
| 2026-08-10 | Atlanta Braves | 375 | 1.196 | L |
| 2026-08-04 | Washington Nationals | 353 | 0.976 | L |
| 2026-08-04 | Cleveland Guardians | 285 | 0.739 | L |
| 2026-08-10 | St. Louis Cardinals | 189 | 0.347 | L |
| 2026-08-04 | San Francisco Giants | 170 | 0.263 | L |
| 2026-08-05 | Washington Nationals | 149 | 0.228 | W |
| 2026-08-11 | Boston Red Sox | 120 | 0.185 | L |

PLAYED 15%+ sample:

| date | pick | odds | edge | result |
| --- | --- | --- | --- | --- |
| 2026-08-05 | Detroit Tigers | 130 | 0.169 | L |
| 2026-08-04 | Los Angeles Dodgers | 107 | 0.164 | L |
| 2026-08-10 | Baltimore Orioles | 117 | 0.152 | L |

**Finding:** PLAYED 15%+ n=3 ⚠️ insufficient — do not conclude; FLAGGED n=7 ⚠️ insufficient — do not conclude. Hold MAX_EDGE_DECIMAL=0.17.



## 5. Team subsets

### (A) Full record (dashboard filter, from 2026-05-03)

| Subset | n | W-L | Win% | ROI | Outlier? |
| --- | --- | --- | --- | --- | --- |
| Overall | n=215 | 103-112 | 47.9% | +3.3% | — |
| LAD / Dodgers | n=4 ⚠️ insufficient — do not conclude | 2-2 | 50.0% | -16.5% | insufficient — do not conclude |
| NYY / Yankees | n=13 ⚠️ insufficient — do not conclude | 5-8 | 38.5% | -24.2% | insufficient — do not conclude |
| KC / Royals | n=16 ⚠️ insufficient — do not conclude | 8-8 | 50.0% | +16.5% | insufficient — do not conclude |
| CHC / Cubs | n=16 ⚠️ insufficient — do not conclude | 8-8 | 50.0% | +6.8% | insufficient — do not conclude |
| WSH / Nationals | n=9 ⚠️ insufficient — do not conclude | 4-5 | 44.4% | -3.9% | insufficient — do not conclude |
**Finding:** No subset both reaches n≥20 and diverges ≥25pp ROI from overall — no team-based param change.


### (B) post_odds_fix only (date ≥ 2026-08-04)

| Subset | n | W-L | Win% | ROI | Outlier? |
| --- | --- | --- | --- | --- | --- |
| Overall | n=15 ⚠️ insufficient — do not conclude | 5-10 | 33.3% | -24.4% | — |
| LAD / Dodgers | n=1 ⚠️ insufficient — do not conclude | 0-1 | 0.0% | -100.0% | insufficient — do not conclude |
| NYY / Yankees | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — | — |
| KC / Royals | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — | — |
| CHC / Cubs | n=1 ⚠️ insufficient — do not conclude | 1-0 | 100.0% | +107.0% | insufficient — do not conclude |
| WSH / Nationals | n=1 ⚠️ insufficient — do not conclude | 0-1 | 0.0% | -100.0% | insufficient — do not conclude |
**Finding:** Overall n=15 ⚠️ insufficient — do not conclude; no team-based param changes.



## 6. Pitcher IP guardrail

### (A) Full record (dashboard filter, from 2026-05-03)

| min(away_ip,home_ip) | n | W-L | Win% | ROI |
| --- | --- | --- | --- | --- |
| 10–15 IP | n=8 ⚠️ insufficient — do not conclude | 3-5 | 37.5% | -17.5% |
| 15–20 IP | n=6 ⚠️ insufficient — do not conclude | 3-3 | 50.0% | +9.6% |
| 20+ IP | n=193 | 93-100 | 48.2% | +4.0% |

IP missing (excluded from buckets): **2** / 215 rows.


**Baseline (IP-known only):** n=213 | 103-110 | win% 48.4% | wagered $2130 | P/L $+91.75 | ROI +4.3%
- Raise MIN_PITCHER_IP_FOR_PICK to 15: keep n=199 | 96-103 | win% 48.2% | wagered $1990 | P/L $+82.55 | ROI +4.1%; remove 14 (7-7, ROI +6.6% on removed)
- Raise MIN_PITCHER_IP_FOR_PICK to 20: keep n=193 | 93-100 | win% 48.2% | wagered $1930 | P/L $+76.81 | ROI +4.0%; remove 20 (10-10, ROI +7.5% on removed)

**Finding:** IP cut thin (10–15 n=8 ⚠️ insufficient — do not conclude; IP-known n=213). Missing IP on 2 rows. Do not raise MIN_PITCHER_IP_FOR_PICK from 10 — needs more data.


### (B) post_odds_fix only (date ≥ 2026-08-04)

| min(away_ip,home_ip) | n | W-L | Win% | ROI |
| --- | --- | --- | --- | --- |
| 10–15 IP | n=0 ⚠️ insufficient — do not conclude | 0-0 | — | — |
| 15–20 IP | n=1 ⚠️ insufficient — do not conclude | 0-1 | 0.0% | -100.0% |
| 20+ IP | n=14 ⚠️ insufficient — do not conclude | 5-9 | 35.7% | -19.0% |

IP missing (excluded from buckets): **0** / 15 rows.


**Baseline (IP-known only):** n=15 ⚠️ insufficient — do not conclude | 5-10 | win% 33.3% | wagered $150 | P/L $-36.60 | ROI -24.4%
- Raise to 15: keep n=15 ⚠️ insufficient — do not conclude | 5-10 | win% 33.3% | wagered $150 | P/L $-36.60 | ROI -24.4%; remove n=0
- Raise MIN_PITCHER_IP_FOR_PICK to 20: keep n=14 ⚠️ insufficient — do not conclude | 5-9 | win% 35.7% | wagered $140 | P/L $-26.60 | ROI -19.0%; remove 1 (0-1, ROI -100.0% on removed)

**Finding:** IP cut thin (10–15 n=0 ⚠️ insufficient — do not conclude; IP-known n=15 ⚠️ insufficient — do not conclude). Missing IP on 0 rows. Do not raise MIN_PITCHER_IP_FOR_PICK from 10 — needs more data.



## Cross-cutting notes


- Authoritative universe = dashboard MLB Record filter on `play_history` (n=215 now; July-31 checkpoint n=186 matched the ~186 / +$114.45 / +6.2% sanity target).
- Edge and model_prob are **stored** on every row (`my_edge_pct`, `my_probability`) — no recompute required; 0 unreconstructable.
- FLAGGED / SKIP_HEAVY_CHALK hypo results still come from mid-season `decision_log` + Stats API (those verdicts are not archived as plays).
- post_odds_fix n=15: do **not** change params from (B) cuts with n&lt;20.
