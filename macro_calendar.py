"""
macro_calendar.py — Economic calendar awareness for the trading system.

Checks whether today is a scheduled high-impact macro event day. On such days
the trading system raises its confidence threshold and delays new entries until
10:30 AM ET to reduce exposure to gap-and-reverse patterns.

Data sources (in priority order):
    1. Daily cache file    — avoids repeated API calls within a session
    2. Static FOMC list   — Fed meeting dates published ~1 year in advance;
                            update _FOMC_DATES each November for the next year.
                            Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
    3. FRED release/dates — CPI, NFP, GDP, PPI exact release dates from BLS/BEA.
                            Uses the already-configured FRED_API_KEY from config.

On FRED API failure the function falls back gracefully to the static FOMC list
only; non-FOMC days are treated as normal when FRED is unreachable.

Usage:
    from macro_calendar import check_high_impact_day
    is_high_impact, event_name = check_high_impact_day()
"""

import json
import os
import time
import requests
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config import config
from logger import log_error


# ── FOMC Meeting Dates (static — update each November) ────────────────────────
# Rate-decision day only (final day of each two-day meeting).
# Verify / extend at: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_DATES: frozenset = frozenset({
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10',
    # 2026 — estimated from historical Fed schedule; verify before each year
    '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09',
})

# FRED release IDs mapped to human-readable event names.
# Confirmed working via FRED API on 2026-04-09.
_FRED_RELEASES: dict = {
    'CPI': (10, 'CPI (Consumer Price Index)'),
    'NFP': (50, 'NFP (Nonfarm Payrolls)'),
    'GDP': (53, 'GDP'),
    'PPI': (46, 'PPI (Producer Price Index)'),
}


def check_high_impact_day(today: Optional[date] = None) -> tuple:
    """
    Return (True, event_name) if today is a high-impact macro event day,
    (False, '') otherwise.

    Checks in order:
        1. Daily cache — free if already checked this session
        2. Static FOMC date list
        3. FRED release/dates API for CPI, NFP, GDP, PPI

    Args:
        today: Date to check. Defaults to today in America/New_York timezone.

    Returns:
        Tuple of (is_high_impact: bool, event_name: str).
        event_name is '' when is_high_impact is False.
    """
    if today is None:
        today = datetime.now(ZoneInfo('America/New_York')).date()

    today_str = today.strftime('%Y-%m-%d')
    cache_path = os.path.join(config.cache_dir, f'macro_events_{today_str}.json')

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            return cached['is_high_impact'], cached['event_name']
        except Exception:
            pass  # Fall through to live check

    # ── FOMC static list ──────────────────────────────────────────────────────
    if today_str in _FOMC_DATES:
        return _cache_and_return(cache_path, True, 'FOMC Rate Decision')

    # ── FRED release/dates API ────────────────────────────────────────────────
    if config.fred_api_key:
        try:
            for _label, (release_id, event_name) in _FRED_RELEASES.items():
                release_dates = _fetch_fred_release_dates(release_id, today)
                if today_str in release_dates:
                    return _cache_and_return(cache_path, True, event_name)
        except Exception as e:
            print(f'[macro_calendar] FRED unavailable after retries: {e} — treating as normal day')
            # FRED unreachable — safe default: treat as normal day

    return _cache_and_return(cache_path, False, '')


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fetch_fred_release_dates(release_id: int, today: date) -> set:
    """
    Fetch FRED release dates for the given release within ±1 day of today.

    Narrow window keeps the response small. FRED returns the exact calendar
    dates on which each data series is scheduled to be released publicly.

    Retries up to 2 times with exponential backoff on 5xx errors. Client
    errors (4xx) are raised immediately — they indicate bad config, not
    transient failures.
    """
    params = {
        'release_id':   release_id,
        'api_key':      config.fred_api_key,
        'file_type':    'json',
        'realtime_start': (today - timedelta(days=1)).strftime('%Y-%m-%d'),
        'realtime_end':   (today + timedelta(days=1)).strftime('%Y-%m-%d'),
        'include_release_dates_with_no_data': 'true',
    }
    last_exc: Exception = RuntimeError('no attempts made')
    for attempt in range(3):
        try:
            r = requests.get(
                'https://api.stlouisfed.org/fred/release/dates',
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            return {d['date'] for d in r.json().get('release_dates', [])}
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise  # 4xx — bad key or bad params, retrying won't help
            last_exc = exc
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < 2:
            time.sleep(2 ** attempt)  # 1 s, then 2 s
    raise last_exc


def _cache_and_return(path: str, is_high_impact: bool, event_name: str) -> tuple:
    try:
        with open(path, 'w') as f:
            json.dump({'is_high_impact': is_high_impact, 'event_name': event_name}, f)
    except Exception:
        pass
    return is_high_impact, event_name
