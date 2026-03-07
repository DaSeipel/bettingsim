"""
Weekly NCAAB retrain job: merge closing lines, run quick_train_ncaab, log date/games/val accuracy.
Scheduled for every Monday 6am ET via main.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path


def run_weekly_retrain() -> None:
    """
    (1) Run merge_historical_closing_into_games().
    (2) Run scripts/quick_train_ncaab.py.
    (3) Append date, games used, and validation accuracy to data/logs/retrain_log.txt.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    espn_db = root / "data" / "espn.db"
    odds_db = root / "data" / "odds.db"
    log_file = root / "data" / "logs" / "retrain_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # (1) Merge closing lines into games
    from engine.historical_odds import merge_historical_closing_into_games
    merge_historical_closing_into_games(espn_db_path=espn_db, odds_db_path=odds_db)

    # (2) Run quick train (KenPom merge, merge again, count, train if >= 50)
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "quick_train_ncaab.py")],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # (3) Parse and log
    today = date.today().isoformat()
    games_used: int | None = None
    val_accuracy: float | None = None

    # "NCAAB games with non-null closing_home_spread: 597"
    m = re.search(r"NCAAB games with non-null closing_home_spread:\s*(\d+)", stdout)
    if m:
        games_used = int(m.group(1))

    # "Total games used for training: 597" (from train_ncaab_kenpom.py when train runs)
    if games_used is None:
        m = re.search(r"Total games used for training:\s*(\d+)", stdout)
        if m:
            games_used = int(m.group(1))

    # "Validation accuracy (cover): 0.5234"
    m = re.search(r"Validation accuracy \(cover\):\s*([\d.]+)", stdout)
    if m:
        val_accuracy = float(m.group(1))

    line_parts = [today]
    if games_used is not None:
        line_parts.append(f"games={games_used}")
    else:
        line_parts.append("games=?")
    if val_accuracy is not None:
        line_parts.append(f"val_accuracy={val_accuracy:.4f}")
    else:
        line_parts.append("val_accuracy=skipped_or_unavailable")
    log_line = " ".join(line_parts) + "\n"

    with open(log_file, "a") as f:
        f.write(log_line)

    if proc.returncode != 0 and stderr:
        with open(log_file, "a") as f:
            f.write(f"  stderr: {stderr[:500]}\n")


if __name__ == "__main__":
    run_weekly_retrain()
    log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "retrain_log.txt"
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
        if lines:
            print("retrain_log.txt (last entry):", lines[-1].strip())
