"""
scheduler.py — Automated trading cycle runner for Railway deployment.

This is the process entrypoint for production. It drives the entire trading
loop according to the RUN_MODE set in .env:

    fixed_6x:        6 scheduled cycles per day at fixed market times.
    intraday_30min:  Every 30 minutes from 9:30 AM to 3:30 PM.

Both modes share the same 3:45 PM pre-close run (intraday position flush)
and 4:00 PM end-of-day report generation.

The CircuitBreaker is instantiated at module level so its high-water mark
state persists across all cycles within the same process lifetime without
requiring repeated disk reads.

Run locally:
    python scheduler.py

Deploy on Railway:
    Set the start command to: python scheduler.py
"""

import schedule, time
from datetime import datetime
from crew import run_trading_cycle
from report_generator import generate_daily_report
from circuit_breaker import CircuitBreaker
from position_monitor import PositionMonitor
from config import config, RunMode
from logger import log_run


# ── Module-Level Circuit Breaker ──────────────────────────────────────────────
# Single instance shared across all cycles in this process. The breaker loads
# its peak value from disk at instantiation and updates it in memory on each
# check, minimising file I/O while preserving state across scheduled runs.
cb = CircuitBreaker()


# ── Market Hours Guard ────────────────────────────────────────────────────────

def market_is_open() -> bool:
    """
    Guard against running trading logic outside NYSE market hours.

    Used as the first check in run_cycle() so scheduled jobs silently no-op
    on weekends and holidays without requiring schedule entries to be removed.
    Time checks use local system time — ensure the host timezone is set to
    US/Eastern in production (Railway environment variable TZ=America/New_York).

    Returns:
        True  — current time is within Monday–Friday 9:30 AM–4:00 PM window.
        False — outside market hours; cycle should be skipped.
    """
    now = datetime.now()

    # weekday(): Monday=0 … Sunday=6; skip Saturday (5) and Sunday (6)
    if now.weekday() >= 5:
        return False

    # Pre-open: before 9:30 AM
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return False

    # Post-close: 4:00 PM and beyond
    if now.hour >= 16:
        return False

    return True


# ── Cycle Functions ───────────────────────────────────────────────────────────

def run_cycle():
    """
    Execute one full trading analysis and order cycle.

    Checks market hours and delegates to crew.run_trading_cycle(), passing
    the shared circuit breaker so the crew can halt before placing orders
    if the breaker has fired. Exceptions are caught here rather than in the
    crew to ensure the scheduler loop itself never crashes.
    """
    if not market_is_open():
        return  # Silent skip — expected on weekends / holidays

    print(f'{datetime.now()} — Starting trading cycle ({config.run_mode})')
    try:
        run_trading_cycle(cb)
    except Exception as e:
        print(f'Error: {e}')
        # Persist the error for dashboard surfacing and post-mortem review
        log_run(error=str(e))


def pre_close_run():
    """
    3:45 PM special cycle — runs a normal analysis pass then flushes all
    open intraday positions before the 4:00 PM market close.

    TradeExecutor is imported lazily here (rather than at module level) to
    avoid initialising the Alpaca client until it is actually needed,
    keeping startup time fast when the scheduler is configured but market
    is not yet open.
    """
    # Run a final normal cycle first to catch any last signals
    run_cycle()

    # Force-close all intraday positions to eliminate overnight gap risk
    from trade_executor import TradeExecutor
    monitor = PositionMonitor(TradeExecutor())
    monitor.close_all_intraday()


def end_of_day():
    """
    4:00 PM post-market job — generates the daily PDF performance report.
    Runs after market close so all fills and P&L are final before the
    report is compiled.
    """
    print(f'{datetime.now()} — Generating end of day report')
    generate_daily_report()


# ── Schedule Configuration ────────────────────────────────────────────────────
# Jobs are registered at module load time based on the RUN_MODE in config.
# Both modes append the 3:45 PM pre-close run and 4:00 PM EOD report.

if config.run_mode == RunMode.FIXED_6X:
    # Six evenly-spaced cycles capture the open, mid-morning, lunch,
    # early afternoon, pre-close, and close periods of the trading day.
    print('Starting in FIXED 6X DAILY mode')
    schedule.every().day.at('09:30').do(run_cycle)   # Market open
    schedule.every().day.at('11:00').do(run_cycle)   # Mid-morning
    schedule.every().day.at('13:00').do(run_cycle)   # Post-lunch
    schedule.every().day.at('14:30').do(run_cycle)   # Early afternoon
    schedule.every().day.at('15:45').do(pre_close_run)  # Pre-close flush
    schedule.every().day.at('16:00').do(end_of_day)  # EOD report

elif config.run_mode == RunMode.INTRADAY_30MIN:
    # Fire every 30 minutes throughout the trading session.
    # A 9:00 slot is registered in the hour loop but skipped explicitly
    # because the market doesn't open until 9:30.
    print('Starting in 30-MINUTE INTRADAY mode')
    for hour in range(9, 16):
        for minute in ['00', '30']:
            time_str = f'{hour:02d}:{minute}'
            # Skip 9:00 AM — market is not yet open at that time
            if hour == 9 and minute == '00':
                continue
            schedule.every().day.at(time_str).do(run_cycle)

    # Pre-close flush and EOD report run identically in both modes
    schedule.every().day.at('15:45').do(pre_close_run)
    schedule.every().day.at('16:00').do(end_of_day)


# ── Process Entrypoint ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Trading scheduler started in {config.run_mode} mode')
    # Poll every 30 seconds — fine-grained enough for minute-level scheduling
    # without burning CPU. schedule.run_pending() is non-blocking.
    while True:
        schedule.run_pending()
        time.sleep(30)
