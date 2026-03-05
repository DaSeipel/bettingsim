"""
Bobby Bottle's Betting Model - Main Entry Point
Bridges the engine and strategies to the Streamlit UI.
Launch the dashboard with: python main.py
Runs a daily 9am ET archive job via APScheduler to save plays before odds shift.
"""

import sys
import os
import threading


def _start_scheduler() -> None:
    """Start APScheduler in a daemon thread: 8am auto-result, 9am play-history archive (ET daily)."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from engine.archive_job import run_daily_archive
        from engine.auto_result_job import run_auto_result
        scheduler = BackgroundScheduler(timezone="America/New_York")
        scheduler.add_job(run_auto_result, CronTrigger(hour=8, minute=0), id="play_auto_result")
        scheduler.add_job(run_daily_archive, CronTrigger(hour=9, minute=0), id="play_history_archive")
        scheduler.start()
    except Exception:
        pass


def main() -> None:
    """Run the Streamlit dashboard (engine + strategies wired in app.py)."""
    # Ensure project root is on path for imports (engine, strategies)
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Start 8am ET auto-result and 9am ET archive jobs in background
    t = threading.Thread(target=_start_scheduler, daemon=True)
    t.start()

    # Launch Streamlit app
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.join(project_root, "app.py"), "--server.headless", "true"]
    stcli.main()


if __name__ == "__main__":
    main()
