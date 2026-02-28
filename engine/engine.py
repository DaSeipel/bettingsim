"""
Betting simulation engine.
Runs a strategy over a historical dataset and tracks bankroll, wins/losses, ROI.
"""

from __future__ import annotations

import pandas as pd
from typing import Callable, Any


# Row dict passed to strategy: at least odds, model_prob, result; optionally event_id, event_name.
StrategyFn = Callable[[pd.Series, float], float]


class BettingEngine:
    """
    Simulates a series of bets over a historical dataset using a given strategy.
    Tracks bankroll, wins, losses, and ROI.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: StrategyFn,
        starting_bankroll: float,
    ) -> None:
        """
        Args:
            df: Historical dataset with columns [odds, model_prob, result].
                result: 1 = win, 0 = loss.
                Optional: event_id, event_name for display.
            strategy: Function (row: pd.Series, bankroll: float) -> stake (float). 0 = no bet.
            starting_bankroll: Initial bankroll.
        """
        self.df = df.reset_index(drop=True)
        self.strategy = strategy
        self.starting_bankroll = starting_bankroll
        self._bet_history: list[dict[str, Any]] = []
        self._bankroll_curve: list[float] = []

    def run(self) -> dict[str, Any]:
        """
        Run the simulation over all rows.
        Returns summary and bet history for display.
        """
        self._bet_history = []
        self._bankroll_curve = []
        bankroll = self.starting_bankroll
        self._bankroll_curve.append(bankroll)

        wins = 0
        losses = 0
        total_staked = 0.0
        total_profit = 0.0

        for step, (idx, row) in enumerate(self.df.iterrows()):
            stake = self.strategy(row, bankroll)
            if stake <= 0 or stake > bankroll:
                self._bankroll_curve.append(bankroll)
                continue

            odds = float(row["odds"])
            result = int(row["result"])
            total_staked += stake

            if result == 1:
                profit = stake * (odds - 1.0)
                wins += 1
                result_str = "Won"
            else:
                profit = -stake
                losses += 1
                result_str = "Lost"

            total_profit += profit
            bankroll += profit
            self._bankroll_curve.append(bankroll)

            record = {
                "step": step + 1,
                "event_id": row.get("event_id", idx),
                "event_name": row.get("event_name", f"Event {idx + 1}"),
                "odds": odds,
                "model_prob": row.get("model_prob", 1.0 / odds),
                "stake": round(stake, 2),
                "result": result_str,
                "profit": round(profit, 2),
                "bankroll_after": round(bankroll, 2),
            }
            self._bet_history.append(record)

        roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
        return {
            "bet_history": self._bet_history,
            "bankroll_curve": self._bankroll_curve,
            "final_bankroll": round(bankroll, 2),
            "wins": wins,
            "losses": losses,
            "total_bets": wins + losses,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 2),
        }
