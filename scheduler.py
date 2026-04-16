"""
scheduler.py — Automated trading cycle runner for Railway deployment.

This is the process entrypoint for production. It drives the entire trading
loop according to the RUN_MODE set in .env:

    fixed_6x:        6 scheduled cycles per day at fixed market times.
    intraday_30min:  Every 30 minutes from 9:30 AM to 3:30 PM.
    intraday_10min:  5min open/close windows, 15min mid-day (9:30 AM–3:50 PM).
    intraday_smart:  ORB-aware schedule — skips 9:30–10:00 AM to let the opening
                     range form, then 5min from 10:00–11:00 AM, 15min from
                     11:00 AM–2:30 PM, and 5min from 2:30–3:50 PM. (default)

All modes force-close all intraday positions at 3:50 PM and generate an
end-of-day PDF report at 4:00 PM.

The CircuitBreaker is instantiated at module level so its high-water mark
state persists across all cycles within the same process lifetime without
requiring repeated disk reads.

Run locally:
    python scheduler.py

Deploy on Railway:
    Set the start command to: python scheduler.py
"""

import schedule, time, threading
from datetime import datetime
from zoneinfo import ZoneInfo
from crew import run_trading_cycle, run_single_ticker, run_position_monitor_only
from report_generator import generate_daily_report
from circuit_breaker import CircuitBreaker
from position_monitor import PositionMonitor
from config import config, RunMode
from logger import log_run
from news_monitor import NewsMonitor


# ── Module-Level Circuit Breaker ──────────────────────────────────────────────
# Single instance shared across all cycles in this process. The breaker loads
# its peak value from disk at instantiation and updates it in memory on each
# check, minimising file I/O while preserving state across scheduled runs.
cb = CircuitBreaker()

# Tracks the last date the morning stale-order cleanup ran so it fires exactly
# once per trading day (first cycle that passes market_is_open()).
_last_cleanup_date = None


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
    now = datetime.now(ZoneInfo('America/New_York'))

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

    On the first cycle of each trading day, runs a stale-order cleanup before
    any analysis to ensure no orders from the prior session survive overnight.
    """
    global _last_cleanup_date

    if not market_is_open():
        return  # Silent skip — expected on weekends / holidays

    # Morning stale-order cleanup — runs exactly once on the first cycle of each
    # trading day, before any analysis or order placement.
    et_today = datetime.now(ZoneInfo('America/New_York')).date()
    if _last_cleanup_date != et_today:
        print(f'🧹 First cycle of {et_today} — running morning stale order cleanup')
        try:
            from trade_executor import TradeExecutor
            executor = TradeExecutor()
            executor.cancel_stale_orders()
            executor.close_stale_intraday_positions()
        except Exception as e:
            print(f'[morning_cleanup] stale order cancel failed: {e}')
        _last_cleanup_date = et_today

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


def run_monitor_check():
    """
    5-minute lightweight position exit check — no entry evaluation, no Groq calls.

    Runs throughout the trading day on all schedule modes. Catches stop-loss,
    take-profit, and profit threshold exits faster than the full cycle interval.
    """
    if not market_is_open():
        return
    try:
        run_position_monitor_only()
    except Exception as e:
        print(f'[monitor_check] Error: {e}')


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

elif config.run_mode == RunMode.INTRADAY_10MIN:
    # Smart intraday schedule — high frequency at open and close, reduced mid-day:
    #   9:30 AM – 10:30 AM : every 5 minutes  (volatile open window)
    #   10:30 AM – 3:00 PM : every 15 minutes (quieter mid-day)
    #   3:00 PM – 3:50 PM  : every 5 minutes  (pre-close momentum window)
    #   3:50 PM            : force-close all intraday positions
    print('Starting in SMART INTRADAY mode (Entries: 9:00AM-12:00PM CT only | Position monitoring: 9:00AM-2:50PM CT | EOD close: 2:50PM CT)')

    # 9:30 AM – 10:25 AM every 5 minutes
    for minute in range(30, 60, 5):
        schedule.every().day.at(f'09:{minute:02d}').do(run_cycle)
    for minute in range(0, 26, 5):
        schedule.every().day.at(f'10:{minute:02d}').do(run_cycle)

    # 10:30 AM – 2:45 PM every 15 minutes
    for hour in range(10, 15):
        for minute in range(0, 60, 15):
            if hour == 10 and minute < 30:
                continue  # Already covered by the 5-min open block above
            if hour == 14 and minute > 45:
                continue  # 3:00 PM block takes over
            schedule.every().day.at(f'{hour:02d}:{minute:02d}').do(run_cycle)

    # 3:00 PM – 3:45 PM every 5 minutes
    for minute in range(0, 46, 5):
        schedule.every().day.at(f'15:{minute:02d}').do(run_cycle)

    # 3:50 PM — final cycle + force-close all intraday positions
    schedule.every().day.at('15:50').do(pre_close_run)
    schedule.every().day.at('16:00').do(end_of_day)

elif config.run_mode == RunMode.INTRADAY_SMART:
    # ORB-aware schedule — skips the opening range formation window entirely:
    #   9:30 AM – 10:00 AM : NO trading (wait for opening range to form)
    #   10:00 AM – 11:00 AM: every 5 minutes  (first hour after ORB confirmation)
    #   11:00 AM – 2:30 PM : every 15 minutes (quiet mid-day)
    #   2:30 PM – 3:50 PM  : every 5 minutes  (closing volatility window)
    #   3:50 PM            : force-close all intraday positions
    print('Starting in SMART INTRADAY mode (Entries: 9:00AM-12:00PM CT only | Position monitoring: 9:00AM-2:50PM CT | EOD close: 2:50PM CT)')

    # 10:00 AM – 10:55 AM every 5 minutes
    for minute in range(0, 60, 5):
        schedule.every().day.at(f'10:{minute:02d}').do(run_cycle)

    # 11:00 AM – 2:15 PM every 15 minutes
    for hour in range(11, 15):
        for minute in range(0, 60, 15):
            if hour == 14 and minute > 15:
                continue  # 2:30 PM block takes over at 14:30
            schedule.every().day.at(f'{hour:02d}:{minute:02d}').do(run_cycle)

    # 2:30 PM – 3:45 PM every 5 minutes
    for minute in range(30, 60, 5):
        schedule.every().day.at(f'14:{minute:02d}').do(run_cycle)
    for minute in range(0, 46, 5):
        schedule.every().day.at(f'15:{minute:02d}').do(run_cycle)

    # 3:50 PM — final cycle + force-close all intraday positions
    schedule.every().day.at('15:50').do(pre_close_run)
    schedule.every().day.at('16:00').do(end_of_day)


# ── 5-Minute Position Monitor (all run modes) ────────────────────────────────
# Lightweight exit checks run every 5 minutes from market open to pre-close,
# independent of the full entry-evaluation schedule. Provides faster reaction
# to stop-loss / take-profit / dynamic profit threshold triggers between full
# cycles, especially during the 15-minute mid-day window.
for _hour in range(9, 16):
    for _minute in range(0, 60, 5):
        if _hour == 9 and _minute < 30:
            continue  # Before market open
        if _hour == 15 and _minute > 45:
            continue  # pre_close_run handles 3:50 PM
        schedule.every().day.at(f'{_hour:02d}:{_minute:02d}').do(run_monitor_check)


# ── News Monitor Loop ─────────────────────────────────────────────────────────

def news_monitor_loop():
    """
    Background thread that polls for breaking news during market hours and
    immediately triggers run_single_ticker() for any high-impact headlines.

    Checks every 60 seconds. NewsMonitor.get_breaking_news() has its own
    internal rate-limit guard so rapid calls are safe — it returns [] until
    its minimum interval has elapsed.
    """
    monitor = NewsMonitor()
    finnhub_error_logged = False
    while True:
        if market_is_open():
            try:
                items = monitor.get_breaking_news()
                for item in items:
                    run_single_ticker(
                        item['ticker'],
                        item['headline'],
                        item['position_size_multiplier'],
                    )
            except Exception as e:
                if not finnhub_error_logged:
                    print(f'News monitor error: {e}')
                    finnhub_error_logged = True
        time.sleep(60)


# ── Process Entrypoint ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Trading scheduler started — Full cycle: 10min (entries + exits), Position monitor: 5min (exits only)')

    # Start news monitor as a daemon thread — exits automatically when the
    # main process exits, no cleanup required.
    threading.Thread(target=news_monitor_loop, daemon=True).start()

    # Poll every 30 seconds — fine-grained enough for minute-level scheduling
    # without burning CPU. schedule.run_pending() is non-blocking.
    while True:
        schedule.run_pending()
        time.sleep(30)
