# Diagnostic V2 Results — MLB Moneyline Filter Counterfactuals

Generated read-only pass. Universe floor: `2026-05-03`. Flat $10 stakes. Excluded PPDs: 6/18 SF/ATL, 6/22 CHC (+129). Replay used frozen params (LOGISTIC_K=0.36, PROB_SHRINK=0.25); live `predict_mlb.py` currently has LOGISTIC_K=0.45 and PROB_SHRINK=0.15 — not changed in this pass.

## Step 0: Data Discovery

### `data/espn.db` schema

| Table | Rows | Date span | Columns (abbrev) |
|-------|------|-----------|------------------|
| `clv_tracker` | 881 | recorded_at: 2026-03-01T20:47:43.915514+00:00 → 2026-03-14T14:21:36.724195+00:00 | id, recorded_at, league, event_name, home_team, away_team, commence_time, market_type… |
| `game_situational_features` | 6819 | — | league, game_id, home_days_rest, away_days_rest, home_is_b2b, away_is_b2b, home_travel_miles, away_travel_miles… |
| `games` | 6819 | game_date: 2023-10-24T23:30Z → 2026-03-07T00:00Z | league, game_id, game_date, game_name, home_team_id, home_team_name, away_team_id, away_team_name… |
| `games_with_team_stats` | 6819 | game_date: 2023-10-24T23:30Z → 2026-03-07T00:00Z | league, game_id, game_date, game_name, home_team_id, home_team_name, away_team_id, away_team_name… |
| `ncaab_team_season_stats` | 1812 | — | season, TEAM, ADJOE, ADJDE, BARTHAG, EFG_O, EFG_D, TOR… |
| `play_history` | 275 | date_generated: 2026-03-09 → 2026-06-24 | play_id, date_generated, sport, home_team, away_team, bet_type, recommended_side, spread_or_total… |
| `schedules` | 5438 | game_date: 2023-10-24T23:30Z → 2025-04-13T19:30Z | league, game_id, game_date, game_name, home_team_id, home_team_name, away_team_id, away_team_name… |
| `sqlite_sequence` | 2 | — | name, seq |
| `team_advanced_stats` | 1216 | — | league, season, team_name, offensive_rating, defensive_rating, pace, turnover_rate, offensive_rebound_rate… |
| `team_season_stats` | 160 | — | league, season, team_id, team_name, wins, losses, record_summary |

MLB moneyline rows in `play_history` since 2026-05-03: **124** (resolved bets only; no skip ledger in DB).

### Archive JSON structure

**Pre–June 12 odds** (`data/odds/mlb_archive/2026-06-03.json`): top-level keys `['fetched_at_utc', 'sport_key', 'source', 'bookmaker_title', 'games_date_et', 'games']`; source=`statsapi_schedule+web_scrape`; game keys `['away_pitcher', 'away_team', 'bookmaker_title', 'commence_time', 'event_id', 'home_pitcher', 'home_team', 'moneyline', 'total']`.

**Post–June 12 odds** (`data/odds/mlb_archive/2026-06-13.json`): source=`statsapi_schedule+the_odds_api`; bookmaker=`TheOddsAPI consensus (median, US books)`.

**Cache archive pick entry keys** (both eras identical): `['away_pitcher', 'away_team', 'card_date', 'commence_time', 'edge', 'event_id', 'home_pitcher', 'home_team', 'market', 'model_prob', 'odds_american', 'park_mult', 'selection']`. No `filtered`, `skip_reason`, or component-prob fields.

### (a) Filtered / SKIPPED picks + W/L

Skipped picks are **not persisted** in SQLite or cache JSON. Cache archives store **passing plays only** (`data/cache/mlb_archive/` copies `mlb_value_plays.json` when ≥1 play clears filters). Counterfactual skip outcomes are reconstructed by **replaying** `predict_mlb.py` moneyline logic against `data/odds/mlb_archive/` daily snapshots, resolving results via MLB Stats API (`backfill_mlb_results.py`). This pass resolved **76** filter-relevant skip candidates (edge ≥ MIN_EDGE, failed a calibration filter) plus **121** play_history bets. Filter-EV on calibration skips at frozen params: **$-94.77**.

### (b) Component signals (pitcher-only / team-only pre-blend)

**Not archived.** Pick entries contain final `model_prob`, `edge`, `park_mult` only — no `pitcher_prob`, `team_prob`, or blend weights. Cumulative `pitcher_stats.csv` cannot reconstruct point-in-time 2026 IP/FIP. **PITCHER_BLEND question → INSUFFICIENT DATA** (re-prediction harness required for v3).

### Universe summary

- Total resolved universe: **197** (121 actual bets + 76 calibration skips)
- Actual bet P/L (play_history): **$+95.89** (60-59-2, 50.4% win)
- Filter-reapplied at frozen params: **108** would-bet, P/L=$+172.66
- Calibration skip filter-EV: **$-94.77** (n=76)

---

## Q1: MAX_FAVORITE_ODDS (current -160)

**Current value:** `-160`

**Empirical at current (actual play_history bets):** n=121, 60-59-2, win%=50.4%, P/L=$+95.89

**Filter-reapplied at -160 (counterfactual bet set):** n=108, 57-49-2, win%=53.8%, P/L=$+172.66, skip filter-EV=$-171.54

> **Replay caveat:** Skip candidates are rebuilt from current `team_stats.csv` / `pitcher_stats.csv`, not point-in-time snapshots. On replay, all 18 `SKIP_HEAVY_CHALK` candidates show edge &lt; 5% (mostly negative), so relaxing the chalk cap to -180 adds **zero** picks to the bet set — the flat sweep below is an artifact of replay, not proof that no marginal chalk ever existed. The manual ledger (`notes/diagnostic_v2_ledger.md`) records 3 pre-migration NYY chalk skips (5/30 -163, 6/2 -199, 6/3 -168) that were **filter-correct** (0-3, ~$30 protected). Treat ledger + replay skip P/L together for this question.

### Full sample sweep

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
| **-160** | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |
|-165 | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |
|-170 | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |
|-175 | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |
|-180 | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |

*All values identical: no replayed skip clears MIN_EDGE after chalk relaxation (see caveat above).*

### VegasInsider era (< 2026-06-12)

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
| **-160** | 76 | 40-34-2 | 54.1% | $+127.45 | $-211.93 | 61 |
|-165 | 76 | 40-34-2 | 54.1% | $+127.45 | $-211.93 | 61 |
|-170 | 76 | 40-34-2 | 54.1% | $+127.45 | $-211.93 | 61 |
|-175 | 76 | 40-34-2 | 54.1% | $+127.45 | $-211.93 | 61 |
|-180 | 76 | 40-34-2 | 54.1% | $+127.45 | $-211.93 | 61 |

*Flat sweep in VI era for same replay reason.*

### Odds API era (≥ 2026-06-12)

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
| **-160** | 32 | 17-15 | 53.1% | $+45.22 | $+40.39 | 28 |
|-165 | 32 | 17-15 | 53.1% | $+45.22 | $+40.39 | 28 |
|-170 | 32 | 17-15 | 53.1% | $+45.22 | $+40.39 | 28 |
|-175 | 32 | 17-15 | 53.1% | $+45.22 | $+40.39 | 28 |
|-180 | 32 | 17-15 | 53.1% | $+45.22 | $+40.39 | 28 |

### NYY subset (selection = NYY)

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
| **-160** | 4 | 2-2 | 50.0% | $-6.58 | $-35.39 | 12 |
|-165 | 4 | 2-2 | 50.0% | $-6.58 | $-35.39 | 12 |
|-170 | 4 | 2-2 | 50.0% | $-6.58 | $-35.39 | 12 |
|-175 | 4 | 2-2 | 50.0% | $-6.58 | $-35.39 | 12 |
|-180 | 4 | 2-2 | 50.0% | $-6.58 | $-35.39 | 12 |

### LAD subset

LAD appears in **9 chalk-filter skips** (replay). Filter-EV on those skips: **$-13.82** (would-be bets net lost money — filter working). LAD was filtered ~5× in 10 days per ledger concern; replay shows repeated heavy-chalk LAD passes blocked with negative re-computed edge, consistent with filter protecting against overconfident chalk rather than a team-specific leak in the bet set (LAD bet subset: 5-1, +$25.70).

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
| **-160** | 5 | 4-1 | 80.0% | $+25.70 | $-13.82 | 9 |
|-165 | 5 | 4-1 | 80.0% | $+25.70 | $-13.82 | 9 |
|-170 | 5 | 4-1 | 80.0% | $+25.70 | $-13.82 | 9 |
|-175 | 5 | 4-1 | 80.0% | $+25.70 | $-13.82 | 9 |
|-180 | 5 | 4-1 | 80.0% | $+25.70 | $-13.82 | 9 |

### BOS subset

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
| **-160** | 4 | 4-0 | 100.0% | $+50.00 | $-15.08 | 5 |
|-165 | 4 | 4-0 | 100.0% | $+50.00 | $-15.08 | 5 |
|-170 | 4 | 4-0 | 100.0% | $+50.00 | $-15.08 | 5 |
|-175 | 4 | 4-0 | 100.0% | $+50.00 | $-15.08 | 5 |
|-180 | 4 | 4-0 | 100.0% | $+50.00 | $-15.08 | 5 |

**Recommended action:** hold at -160. Ledger validates chalk filter on NYY; replay cannot re-open marginal chalk counterfactuals without point-in-time edge. LAD/BOS bet subsets are profitable; LAD skip filter-EV negative (filter correct).

**Confidence:** medium (high on hold; medium on sweep interpretability due to replay gap)

---

## Q2: 5–10% Underdog Tier (readout only)

**Current edge band:** 5–10% decimal edge on underdog (+) odds, among picks passing frozen filters.

**Empirical:** n=31, 16-15, win%=51.6%, P/L=$+46.40 (actual play_history underdog bets since 2026-05-18 with 5–10% edge)

Prior was ~11–13 over ~24 at 46%; verified **31 picks at 51.6%**, +$46.40 — recovered from early 35% low, no urgency to tighten MIN_EDGE.

**Counterfactual sweep:** N/A (readout tier; no parameter sweep requested).

**Recommended action:** hold edge band (5% MIN_EDGE unchanged); tier performance supports no tightening

**Confidence:** medium (n=31)

---

## Q3: MIN_FAVORITE_EDGE_DECIMAL (current 0.10)

**Current value:** `0.1`

**Empirical at current (actual play_history bets):** n=121, 60-59-2, win%=50.4%, P/L=$+95.89

### Sweep (full universe, entire filter stack re-applied)

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
|0.06 | 123 | 62-59-2 | 51.2% | $+113.67 | $-112.55 | 74 |
|0.08 | 112 | 57-53-2 | 51.8% | $+132.66 | $-131.54 | 85 |
| **0.1** | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |
|0.12 | 100 | 51-48-1 | 51.5% | $+135.02 | $-133.90 | 97 |
|0.14 | 96 | 48-47-1 | 50.5% | $+120.72 | $-119.60 | 101 |

*Note: 0.10 is optimal in replay sweep (+$172.66 vs $+113.67 at 0.06), but this uses replay bet-set not actual 121 play_history bets (+$95.89). Actual bets include 13 pre-5/18 favorites with 5–10% edge placed before this filter existed.*

**Recommended action:** hold at 0.10. Sweep peak at 0.10 on replay; relaxing to 0.06–0.08 adds mostly replay favorites with lower P/L. NYY split shows filter-EV -$10 on 1 low-edge skip vs +$7.30 on 5 non-NYY skips (tiny n).

**Confidence:** medium

---

## Q4: PITCHER_BLEND_2026_WEIGHT (current 0.60)

**Current value:** `0.6` (in `fetch_mlb_pitchers.py`; affects live stats, not archived picks).

**Empirical at current:** Cannot isolate — component pitcher/team probabilities not archived.

**Counterfactual sweep:** INSUFFICIENT DATA — re-blend at 0.60/0.70/0.75/0.80 requires point-in-time pitcher rows or archived component signals. Recommend v3 re-prediction harness.

**Recommended action:** insufficient data
**Confidence:** high (data gap confirmed)

---

## Q5: MIN_FAVORITE_EDGE — NYY vs non-NYY

**Current value:** `0.1`

### Non-NYY favorites

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
|0.06 | 24 | 13-10-1 | 56.5% | $+8.15 | $-2.96 | 13 |
|0.08 | 14 | 8-5-1 | 61.5% | $+17.14 | $-11.95 | 23 |
| **0.1** | 11 | 8-2-1 | 80.0% | $+47.14 | $-41.95 | 26 |
|0.12 | 4 | 3-1 | 75.0% | $+16.21 | $-11.03 | 33 |
|0.14 | 1 | 1-0 | 100.0% | $+8.62 | $-3.43 | 36 |

### NYY favorites only

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
|0.06 | 6 | 2-4 | 33.3% | $-26.58 | $-27.89 | 9 |
|0.08 | 5 | 2-3 | 40.0% | $-16.58 | $-37.89 | 10 |
| **0.1** | 4 | 2-2 | 50.0% | $-6.58 | $-47.89 | 11 |
|0.12 | 3 | 1-2 | 33.3% | $-13.29 | $-41.17 | 12 |
|0.14 | 2 | 0-2 | 0.0% | $-20.00 | $-34.46 | 13 |

NYY low-edge skips at 0.10: n=1, filter-EV=$-10.00. Non-NYY favorite low-edge skips: n=5, filter-EV=$+7.30.

**Recommended action:** hold 0.10 globally; NYY subset filter-EV $-10.00 vs non-NYY $+7.30

**Confidence:** low (NYY n small)

---

## Q6: MIN_MODEL_PROB (current 0.42)

**Current value:** `0.42`

**Empirical at current (actual play_history):** n=121, 60-59-2, win%=50.4%, P/L=$+95.89

**Filter-reapplied at 0.42:** n=108, 57-49-2, win%=53.8%, P/L=$+172.66, skip filter-EV=$-171.54 (negative skip EV = filters correctly rejecting losing low-prob picks)

### Sweep

| Value | n | W-L-P | Win% | Total P/L | Filter-EV (skips) | skip n |
|-------|---|-------|------|-----------|-------------------|--------|
|0.4 | 116 | 58-56-2 | 50.9% | $+119.16 | $-118.04 | 81 |
| **0.42** | 108 | 57-49-2 | 53.8% | $+172.66 | $-171.54 | 89 |
|0.44 | 102 | 54-46-2 | 54.0% | $+154.86 | $-153.74 | 95 |
|0.46 | 89 | 49-38-2 | 56.3% | $+165.56 | $-164.44 | 108 |

**Recommended action:** hold at 0.42. Peak replay P/L at 0.42 (+$172.66); 0.44/0.46 add volume but reduce P/L vs 0.42.

**Confidence:** high

---

## One-line summary for next slate

**Hold all six parameters at frozen values;** PITCHER_BLEND sweep deferred to v3 harness.
