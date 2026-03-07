"""
Bobby Bottle's Betting Model - Main Entry Point
Bridges the engine and strategies to the Streamlit UI.
Launch the dashboard with: python main.py
Scheduler (8am/9am ET daily, Monday 6am NCAAB retrain) is started when the app loads in app.py.
"""

import sys
import os


def main() -> None:
    """Run the Streamlit dashboard (engine + strategies wired in app.py)."""
    # Ensure project root is on path for imports (engine, strategies)
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Launch Streamlit app (app.py starts scheduler and startup snapshot/merge on first load)
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.join(project_root, "app.py"), "--server.headless", "true"]
    stcli.main()


if __name__ == "__main__":
    main()
