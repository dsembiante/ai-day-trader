"""
circuit_breaker.py — Hard portfolio drawdown stop for all trading activity.

If the portfolio loses 10% or more from its peak value, the circuit breaker
triggers and check() returns False, which signals the scheduler and crew to
halt all new trade entries immediately. No positions are auto-liquidated —
that requires a manual decision after review.

Peak portfolio value is persisted to disk so the breaker survives process
restarts and continues tracking the high-water mark across sessions.

Design note:
    This is intentionally simple and has no reset mechanism in code.
    Re-enabling trading after a trigger requires a manual deletion of
    data/peak_value.json and a deliberate restart — by design.

Usage:
    from circuit_breaker import CircuitBreaker
    breaker = CircuitBreaker()
    if not breaker.check(current_value):
        # halt all trading
"""

from config import config
from logger import log_error
import json
from datetime import datetime


class CircuitBreaker:
    """
    Tracks the portfolio's all-time peak value and computes rolling drawdown.
    Call check() on every scheduler cycle before any agent work begins.
    """

    def __init__(self):
        # Path to the persisted high-water mark — shared across restarts
        self.peak_value_file = 'data/peak_value.json'
        self._load_peak()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_peak(self):
        """
        Load the previously recorded peak portfolio value from disk.
        On first run (or after a manual reset), peak_value is set to None
        so check() initialises it from the first observed portfolio value.
        """
        try:
            with open(self.peak_value_file) as f:
                self.peak_value = json.load(f)['peak']
        except:
            # File absent or corrupt — start fresh; not an error condition
            self.peak_value = None

    def _save_peak(self):
        """
        Persist the updated high-water mark so it survives process restarts.
        Creates the data/ directory if it was deleted or not yet present.
        """
        import os
        os.makedirs('data', exist_ok=True)
        with open(self.peak_value_file, 'w') as f:
            json.dump({'peak': self.peak_value}, f)

    # ── Core Check ────────────────────────────────────────────────────────────

    def check(self, current_portfolio_value: float) -> bool:
        """
        Evaluate whether trading should continue based on current drawdown.

        Updates the high-water mark whenever portfolio value reaches a new peak,
        then computes drawdown as a fraction of that peak. If drawdown meets or
        exceeds config.circuit_breaker_pct (default 10%), _trigger() is called
        and False is returned to halt the trading cycle.

        Args:
            current_portfolio_value: Total liquidation value of the portfolio
                                     as reported by Alpaca at cycle start.

        Returns:
            True  — drawdown is within acceptable range; trading may proceed.
            False — circuit breaker has fired; all new entries must be blocked.
        """
        # Update high-water mark if this is the first reading or a new peak
        if self.peak_value is None or current_portfolio_value > self.peak_value:
            self.peak_value = current_portfolio_value
            self._save_peak()

        # Drawdown expressed as a positive fraction (e.g. 0.12 = 12% below peak)
        drawdown = (self.peak_value - current_portfolio_value) / self.peak_value

        if drawdown >= config.circuit_breaker_pct:
            self._trigger(current_portfolio_value, drawdown)
            return False  # Caller must block all further trading this cycle

        return True  # Safe to proceed

    # ── Trigger Handler ───────────────────────────────────────────────────────

    def _trigger(self, portfolio_value: float, drawdown: float):
        """
        Log and surface the circuit breaker event.

        Writes a structured error log entry (picked up by the dashboard and
        notifier) and prints directly to stdout for immediate visibility in
        Railway / terminal runs.

        Note: This method does NOT liquidate positions — that is a manual
        decision. The log message instructs the operator to review before
        resuming trading.
        """
        msg = (
            f'CIRCUIT BREAKER TRIGGERED | '
            f'Portfolio: ${portfolio_value:,.2f} | '
            f'Peak: ${self.peak_value:,.2f} | '
            f'Drawdown: {drawdown:.1%} | '
            f'All trading halted. Manual review required.'
        )
        # Persists to errors.log and trade journal for dashboard surfacing
        log_error('circuit_breaker', 'ALL', msg)

        # Direct stdout alert — visible in Railway logs and local terminal
        print(f'🚨 {msg}')
