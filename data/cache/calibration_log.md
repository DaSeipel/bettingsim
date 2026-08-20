# MLB calibration log

Dated parameter changes only. Does not alter model coefficients, blends, or reporting tier definitions.

| Date | Parameter | Old | New | Rationale | Diagnostic |
|------|-----------|-----|-----|-----------|------------|
| 2026-08-12 | `MIN_EDGE_DECIMAL` | 0.05 | 0.10 | v3 full-record: 5-10% tier -8.4% ROI n=105; both >=10% tiers positive | `data/cache/diagnostics/v3_fullrecord_2026-08-12.md` |

---

## CFB Phase 1 — 2026-08-19

**CFB SPREAD MODEL — FAILED AT GATE**

PIT opponent-adjusted efficiency ratings → projected margin vs closing spread. b₂ = 0.062 (p = 0.062) on restricted 2021+ pool. Projection RMSE 17.8 vs market 15.3. ATS ~50% across calibrated edge buckets. CLV 41% (close moves away from model). No subset pocket: G5 b₂ = 1.20, SE 1.06.

**CFB TOTALS MODEL — FAILED AT GATE**

Pace × PPA crude projection. corr(proj − OU, actual − OU) = +0.052 pooled (n=2995). b₂ = 0.027 (p = 0.127). Openers weaker than close (c = 0.041). No subset concentration. Market over rate 50.3% (z = 0.29) — no shading to exploit. Totals sigma 15.77 vs spread sigma 15.33.

**CONCLUSION:** Public advanced box-score stats do not beat CFB markets at either spreads or totals. Not revisiting without a genuinely new input class (injury/availability data, weather, or line-movement microstructure).

**RETAINED:** 11 seasons of games/lines/box scores, 138-team verified alias map, PIT ratings (r = 0.88–0.91 vs SP+ for 2021+), and the gate methodology (b₂ vs close, walk-forward, CLV, calibrated edge buckets).
