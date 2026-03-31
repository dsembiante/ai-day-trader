"""
position_monitor.py — Enforces hold period rules and intraday exit discipline.

Runs on every scheduler cycle to audit all open positions against their
time-based exit constraints. Two exit mechanisms are managed here:

1. Hold Period Expiry — any position that has been held longer than its
   max_hold_days budget (set at entry time by position_sizer.py) is closed
   regardless of P&L. This prevents trades from drifting outside their
   intended risk window.

2. Intraday Forced Close — all positions classified as 'intraday' are closed
   before market close (3:45 PM cycle) to eliminate overnight gap risk.

Note: Stop-loss and take-profit exits are handled by Alpaca bracket orders
placed at entry time via trade_executor.py — not by this module.

Usage:
    from position_monitor import PositionMonitor
    monitor = PositionMonitor(trade_executor=executor)
    monitor.check_all_positions()
"""

import time
from database import Database
from config import config, HoldPeriod
from datetime import datetime, timedelta
from logger import log_error


class PositionMonitor:
    """
    Audits open positions on every cycle and triggers time-based exits.
    Requires a live TradeExecutor instance to place closing orders.
    """

    def __init__(self, trade_executor):
        self.db = Database()
        # Executor is injected rather than instantiated here to share the
        # same authenticated Alpaca client used by the rest of the crew
        self.executor = trade_executor

    # ── Public Interface ──────────────────────────────────────────────────────

    def check_all_positions(self):
        """
        Retrieve all open trades from the database and evaluate each one
        against its hold period constraint. Called at the start of every
        scheduler cycle before new signals are analysed.
        """
        open_trades = self.db.get_open_trades()
        for trade in open_trades:
            self._check_hold_expiry(trade)

    # ── Hold Period Enforcement ───────────────────────────────────────────────

    def _check_hold_expiry(self, trade: dict):
        """
        Close a position if it has exceeded its maximum allowed hold duration.

        max_hold_days is stored on the trade record at entry time (sourced from
        config: 1 for intraday, 5 for swing, 20 for position trades). Using the
        per-record value rather than re-reading config means the rule applied
        at entry is always the rule enforced at exit, even if config changes.

        Args:
            trade: A trade record dict as returned by Database.get_open_trades().
        """
        entry_time = datetime.fromisoformat(trade['entry_time'])
        days_held = (datetime.now() - entry_time).days

        # Fall back to 5 days (swing default) if the field is missing from
        # legacy records written before max_hold_days was added to the schema
        max_days = trade.get('max_hold_days', 5)

        if days_held >= max_days:
            print(
                f"Position {trade['ticker']} exceeded max hold period "
                f"({days_held} days). Closing."
            )
            try:
                # Place a market close order via Alpaca
                self.executor.close_position(trade['ticker'], trade['trade_type'])

                # Wait for Alpaca to record the fill before querying order history
                time.sleep(2)
                exit_price = self.executor.get_filled_exit_price(trade['ticker'])

                # Update the database record so the dashboard and reports
                # reflect the correct exit reason for post-trade analysis
                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='hold_period_expired',
                    exit_price=exit_price,
                )
            except Exception as e:
                # Log and continue — a failure on one position should not
                # prevent the monitor from checking the remaining positions
                log_error('position_monitor', trade['ticker'], str(e))

    # ── Intraday Exit Logic ───────────────────────────────────────────────────

    def is_intraday_close_time(self) -> bool:
        """
        Returns True after 3:30 PM local time — the window during which all
        intraday positions must be closed before the 4:00 PM market close.

        The 3:45 PM scheduler cycle checks this flag before calling
        close_all_intraday(), giving a 15-minute execution buffer.
        """
        now = datetime.now()
        return now.hour >= 15 and now.minute >= 30

    def close_all_intraday(self):
        """
        Force-close every open intraday position before market close.

        Called by the scheduler on the 3:45 PM cycle when is_intraday_close_time()
        returns True. Intraday positions must never be held overnight — the
        wider gap risk is outside the risk budget defined for this hold tier.

        Guard: if config.allow_intraday is False, no intraday positions should
        exist (get_hold_period_safe() upgrades them all to swing at entry).
        The early return here is a safety net for that case.

        Failures are logged individually; remaining intraday positions continue
        to be processed so a single bad close does not leave others open.
        """
        # No intraday positions can exist when PDT protection is active
        if not config.allow_intraday:
            print('⚠️  Intraday disabled — no intraday positions to close')
            return

        open_trades = self.db.get_open_trades()

        # Filter to intraday tier only — swing and position trades are unaffected
        intraday = [t for t in open_trades if t.get('hold_period') == 'intraday']

        for trade in intraday:
            try:
                self.executor.close_position(trade['ticker'], trade['trade_type'])

                # Wait for Alpaca to record the fill before querying order history
                time.sleep(2)
                exit_price = self.executor.get_filled_exit_price(trade['ticker'])

                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='intraday_forced_close',  # Distinct from hold_period_expired
                    exit_price=exit_price,
                )
            except Exception as e:
                log_error('intraday_close', trade['ticker'], str(e))
