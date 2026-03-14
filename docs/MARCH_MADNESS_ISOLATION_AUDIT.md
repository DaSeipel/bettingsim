# March Madness Logic — Isolation Audit

**Date:** 2026-03-14  
**Goal:** Confirm tournament multipliers (Veteran Edge, Closer Factor, 3P Variance) are strictly isolated from the core betting engine and other app tabs.

---

## 1. Stateless logic ✅

- **Veteran Edge, Closer Factor, 3P Variance** exist only in **`engine/bracket_analysis.py`**:
  - Constants: `VETERAN_EXP_MIN`, `NON_VETERAN_EXP_MAX`, `VETERAN_MARGIN_BOOST`, `CLOSER_GAME_MARGIN_THRESHOLD`, `ELITE_FT_PCT_MIN`, `CLOSER_FT_WIN_PROB_BOOST`, `MARGIN_SIGMA_BASE`, `MARGIN_SIGMA_ELITE_3P`, etc.
  - Applied only inside **`_build_get_winner_with_march_factors()`** in the **`winner(ta, tb, rng)`** closure used by the Monte Carlo simulation loop.
- No global state: March factors are applied only when the caller uses **`run_bracket_analysis()`** or the **`get_winner`** callable returned by **`_build_get_winner_with_march_factors()`**.

---

## 2. Core model integrity ✅

- **`engine/betting_models.py`**:
  - **`get_spread_predicted_margin()`** and **`predict_spread_prob()`** are **unchanged** in behavior. They use **`payload["feature_columns"]`** (from the saved model pkl) and **`_select_features(df, cols)`**, which only passes those columns to the model. The NCAAB spread model’s feature list does **not** include any March columns.
  - **`build_ncaab_feature_row_from_team_stats()`** no longer adds `free_throw_pct`, `three_point_pct`, or `roster_experience_years` to the row. A comment documents that March data is **only** read in **`engine/bracket_analysis.py`** via **`load_march_stats()`**, so the core engine stays pure for daily/historical use.
- **Standard NCAAB game prediction** (Overview, NCAAB tab, predict_games.py, value_plays_pipeline) uses the same **`get_spread_predicted_margin`** / **`predict_spread_prob`** with no March weights. **Pure numbers for regular season / non-bracket games.**

---

## 3. Database safety ✅

- **New columns** on **`ncaab_team_season_stats`**: `free_throw_pct`, `three_point_pct`, `roster_experience_years`.
- **Daily Betting / Historical tabs** do **not** query **`ncaab_team_season_stats`** directly. They use:
  - **`play_history`** (Mark Results, Record)
  - **`games_with_team_stats`** (Game Lookup, feature loading)
- **`ncaab_team_season_stats`** is read only by:
  - **`engine/betting_models.py`** → **`build_ncaab_feature_row_from_team_stats()`** (KenPom-style columns only; March columns are not read or written to the feature row).
  - **`engine/bracket_analysis.py`** → **`load_march_stats()`** (reads the three March columns for bracket only).
  - Scripts: **merge_ncaab_kenpom_into_games.py**, **update_ncaab_march_stats.py**, **audit_march_data.py**.
- Adding columns to an existing table does not break **SELECT**s that use explicit column lists or **SELECT *** (extra columns are ignored by existing code). **No breaking change to existing queries.**

---

## 4. Namespace / API check ✅

- The app is **Streamlit** (no REST `/api/` routes). There is no **`/api/predict`** or **`/api/bracket/analyze`**; the March tab does not overwrite any shared API.
- **March tab**:
  - Uses **`st.button("Run Bracket Analysis", key="march_run_btn")`** and calls **`run_bracket_analysis(...)`** only when that button is clicked.
  - Uses **session state** only for form inputs (**`march_bracket_csv`**, **`march_power_rankings_csv`**, **`march_n_sims`**, **`march_run_btn`**); no shared state with the prediction pipeline.
- **Value plays / daily prediction**:
  - Use **`_load_value_plays_cache()`**, **`consensus_spread()`**, **`predict_spread_prob()`**, and pipeline scripts. **No** call to **`run_bracket_analysis()`** or any bracket-specific logic.

---

## 5. Refactor applied

- **Removed** from **`engine/betting_models.py`** the block that added **`home/away_free_throw_pct`**, **`home/away_three_point_pct`**, and **`home/away_roster_experience_years`** to the feature row in **`build_ncaab_feature_row_from_team_stats()`**.
- **Replaced** with a comment stating that March multiplier data is **not** added there and is only read in **`engine/bracket_analysis.py`** via **`load_march_stats()`**.

---

## Conclusion

- **March Madness logic is strictly isolated** in **`engine/bracket_analysis.py`** and in the simulation loop’s **`get_winner`** callable.
- **Core model:** **`get_spread_predicted_margin`** and **`predict_spread_prob`** are unchanged and produce **pure** margins/probs for regular season and non-bracket use.
- **Database:** New columns on **`ncaab_team_season_stats`** do not break Daily Betting or Historical tabs.
- **Namespace:** March tab uses its own button and **`run_bracket_analysis()`**; it does not overwrite or share the prediction path used by the rest of the app.

**Running a standard NCAAB game prediction today will not be affected by tournament multipliers.**
