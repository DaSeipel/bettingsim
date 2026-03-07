# NCAAB team stats (KenPom-style)

Season-level team statistics for men’s college basketball, normalized across 2022–2025.

## Layout

| Path | Description |
|------|-------------|
| `team_stats_combined.csv` | All seasons in one file; includes `season` (2022–2025). |
| `team_stats_2022.csv` … `team_stats_2025.csv` | One file per season (same schema). |
| `raw/` | Original downloads (`cbb22.csv` … `cbb25.csv`) before normalization. |

## Schema (canonical columns)

- **season** – Year (2022–2025).
- **TEAM**, **CONF** – Team and conference.
- **G**, **W** – Games played, wins.
- **ADJOE**, **ADJDE** – Adjusted offensive/defensive efficiency.
- **BARTHAG** – Probability of beating an average D-I team.
- **EFG_O**, **EFG_D** – Effective FG% offense/defense.
- **TOR**, **TORD** – Turnover rate offense/defense.
- **ORB**, **DRB** – Offensive/defensive rebound rate.
- **FTR**, **FTRD** – Free throw rate offense/defense.
- **2P_O**, **2P_D**, **3P_O**, **3P_D** – Two-/three-point % offense/defense.
- **3PR**, **3PRD** – Three-point rate offense/defense (2025 only; NaN for 2022–2024).
- **ADJ_T** – Adjusted tempo.
- **WAB** – Wins above bubble.
- **POSTSEASON** – Tournament result (e.g. R64, S16, Champions); empty for 2025.
- **SEED** – NCAA tournament seed; may be `N/A` for non-bid teams.

## Regenerating from raw

From repo root:

```bash
python scripts/normalize_cbb_team_stats.py --input-dir data/ncaab/raw
```

To add a new season, place `cbb26.csv` (or similar) in `raw/`, add the mapping in `scripts/normalize_cbb_team_stats.py` (`SEASON_FILES` and optionally `CANONICAL_COLUMNS` if new columns appear), then re-run the script.
