"""
logger.py — Observability and structured logging for the AI trading agent.

Three logging surfaces are maintained:

    errors.log        — Append-only flat file for any component/source error.
                        Consumed by the Risk Monitor tab and alerting.
    run_logs/         — One JSON file per scheduler cycle capturing the full
                        run summary: tickers analysed, trades executed, API
                        health, errors, duration, and circuit breaker status.
    trade_journal.json — Append-only JSON array of every trade event. Provides
                         a human-readable audit trail outside the SQLite database.

All functions are intentionally simple (no log levels, no handlers) — the goal
is reliable, inspectable output over a sophisticated logging framework.

Usage:
    from logger import log_error, log_run, log_trade, new_run_log, RunLog
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List
from config import config

# Create logs directory at import time so every module that calls log_error
# can assume the directory exists without performing its own mkdir check
os.makedirs(config.logs_dir, exist_ok=True)


# ── Run Log Schema ────────────────────────────────────────────────────────────

@dataclass
class RunLog:
    """
    Structured record of a single scheduler cycle.

    One RunLog is created at the start of each cycle via new_run_log(),
    populated as the cycle progresses, and written to disk by log_run()
    on completion. Fields map directly to the JSON output format.
    """
    run_id: str                                          # Timestamp-based unique ID (YYYYMMDD_HHMMSS)
    run_time: str                                        # ISO-8601 cycle start time
    run_mode: str                                        # fixed_6x or intraday_30min
    tickers_analyzed: List[str] = field(default_factory=list)  # Watchlist processed this cycle
    trades_executed: int = 0                             # Count of orders placed
    api_status: dict = field(default_factory=dict)       # Per-source availability snapshot
    errors: List[str] = field(default_factory=list)      # Non-fatal errors accumulated during the run
    duration_seconds: float = 0.0                        # Wall-clock time for the full cycle
    circuit_breaker_triggered: bool = False              # True if breaker fired and halted trading


# ── Error Logging ─────────────────────────────────────────────────────────────

def log_error(source: str, ticker: str, error: str):
    """
    Append a single error entry to errors.log.

    Called by any module that catches an exception it cannot recover from
    (data_collector, circuit_breaker, position_monitor, trade_executor).
    The flat-file format makes errors easy to grep and tail in production.

    Args:
        source: Component or data source that raised the error (e.g. 'alpaca', 'finnhub').
        ticker: Ticker symbol being processed, or 'ALL' for portfolio-wide errors.
        error:  Exception message or descriptive error string.
    """
    timestamp = datetime.now().isoformat()
    # Structured format: [timestamp] [SOURCE] [TICKER] message
    entry = f'[{timestamp}] [{source.upper()}] [{ticker}] {error}\n'

    error_file = os.path.join(config.logs_dir, 'errors.log')
    with open(error_file, 'a') as f:
        f.write(entry)

    # Also print to stdout so errors surface immediately in Railway logs
    # and local terminal runs; truncate at 100 chars to keep output readable
    print(f'⚠️  Error logged: {source} | {ticker} | {error[:100]}')


# ── Run Logging ───────────────────────────────────────────────────────────────

def log_run(run_log: RunLog = None, error: str = None):
    """
    Persist a cycle summary to the run_logs/ directory.

    Two call modes:
        log_run(run_log=rl)    — Full structured run summary at cycle end.
        log_run(error='...')   — Lightweight error-only record when the cycle
                                 itself crashes before a RunLog can be built.

    Files are named run_{run_id}.json (full) or run_{run_id}_error.json (error-only)
    so they sort chronologically in the filesystem and are easy to distinguish.

    Args:
        run_log: Completed RunLog dataclass instance. Mutually exclusive with error.
        error:   Error string for a crashed cycle. Mutually exclusive with run_log.
    """
    run_logs_dir = os.path.join(config.logs_dir, 'run_logs')
    os.makedirs(run_logs_dir, exist_ok=True)

    if run_log:
        # Full run summary — asdict() converts the dataclass to a JSON-serialisable dict
        filename = os.path.join(run_logs_dir, f'run_{run_log.run_id}.json')
        with open(filename, 'w') as f:
            json.dump(asdict(run_log), f, indent=2)

    elif error:
        # Minimal error record when the cycle crashes before a RunLog is available
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(run_logs_dir, f'run_{run_id}_error.json')
        with open(filename, 'w') as f:
            json.dump({
                'run_id':    run_id,
                'error':     error,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)


# ── Trade Journal ─────────────────────────────────────────────────────────────

def log_trade(trade_data: dict):
    """
    Append a trade event to the trade_journal.json flat file.

    The journal duplicates key trade data from SQLite into a human-readable
    JSON array, providing an audit trail that survives database corruption
    and is easy to inspect without SQL tooling.

    The full array is read, appended to, and rewritten on each call. This is
    acceptable given trade frequency (at most ~15 entries per day); for
    higher-frequency systems a streaming append format would be preferable.

    Args:
        trade_data: Dict of trade fields to append — typically the same dict
                    passed to Database.insert_trade().
    """
    journal_file = os.path.join(config.logs_dir, 'trade_journal.json')

    # Load existing entries; start fresh if the file is absent or corrupt
    trades = []
    if os.path.exists(journal_file):
        with open(journal_file) as f:
            try:
                trades = json.load(f)
            except:
                # Corrupt JSON — reset rather than crash; data is in SQLite
                trades = []

    trades.append(trade_data)

    with open(journal_file, 'w') as f:
        json.dump(trades, f, indent=2)


# ── Factory ───────────────────────────────────────────────────────────────────

def new_run_log(tickers: list) -> RunLog:
    """
    Create a fresh RunLog at the start of a new scheduler cycle.

    The run_id doubles as the log filename stem and is used by crew.py
    to correlate run logs with the trades executed in that cycle.

    Args:
        tickers: The watchlist being analysed in this cycle.

    Returns:
        An initialised RunLog ready to be populated as the cycle progresses.
    """
    return RunLog(
        run_id=datetime.now().strftime('%Y%m%d_%H%M%S'),
        run_time=datetime.now().isoformat(),
        run_mode=config.run_mode,
        tickers_analyzed=tickers,
    )
