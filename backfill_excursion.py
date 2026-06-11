#!/usr/bin/env python3
"""
backfill_excursion.py — Backfill max_favorable_excursion_bar_pct and
max_adverse_excursion_bar_pct for closed trades where those columns are NULL.

Math is identical to Database.compute_and_store_excursion():
  Longs:  MFE = (max_high - entry_price) / entry_price   (clamped >= 0)
          MAE = (min_low  - entry_price) / entry_price   (clamped <= 0)
  Shorts: MFE = (entry_price - min_low)  / entry_price   (clamped >= 0)
          MAE = (entry_price - max_high) / entry_price   (clamped <= 0)

Values are stored as fractions (0.015 = 1.5%) with MFE >= 0 and MAE <= 0,
matching the gain_pct convention used throughout the codebase.

Usage:
    python backfill_excursion.py              # dry-run (default) — prints only
    python backfill_excursion.py --execute    # write to DB

Database URL: reads DATABASE_URL_PUBLIC first (needed from local — the plain
DATABASE_URL is Railway-internal and only reachable inside the Railway network),
then DATABASE_URL, from .env or environment.
"""

import os
import sys
import time
import argparse
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(
    description='Backfill MFE/MAE bar excursion for historical closed trades.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        'Examples:\n'
        '  python backfill_excursion.py              # dry-run: inspect without writing\n'
        '  python backfill_excursion.py --execute    # commit changes to the database\n'
    ),
)
ap.add_argument(
    '--execute', action='store_true',
    help='Write computed values to the database. Omitting this flag is a dry-run.',
)
args = ap.parse_args()
DRY_RUN = not args.execute

mode_label = '[DRY-RUN]' if DRY_RUN else '[EXECUTE]'
print(f'{mode_label} backfill_excursion.py')
if DRY_RUN:
    print('  No writes will be made. Pass --execute to commit changes.\n')

# ── DB connection ─────────────────────────────────────────────────────────────
# autocommit=True throughout: no transaction ever sits open across a bar fetch.
# DATABASE_URL_PUBLIC is the Railway-external URL required when running locally.
DATABASE_URL = os.getenv('DATABASE_URL_PUBLIC') or os.getenv('DATABASE_URL', '')
if not DATABASE_URL:
    sys.exit('ERROR: Set DATABASE_URL_PUBLIC (or DATABASE_URL) in .env')

print('Connecting to database...')
try:
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    conn.autocommit = True
except Exception as e:
    sys.exit(f'DB connection failed: {e}')
print('  Connected.\n')

# ── Alpaca client ─────────────────────────────────────────────────────────────
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import config

alpaca = StockHistoricalDataClient(
    api_key=config.alpaca_api_key,
    secret_key=config.alpaca_secret_key,
)

# ── Fetch candidates (idempotent: only trades missing bar-based excursion) ─────
with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
    cur.execute("""
        SELECT trade_id, ticker, trade_type, entry_price, entry_time, exit_time
        FROM   trades
        WHERE  status = 'closed'
          AND  (   max_favorable_excursion_bar_pct IS NULL
                OR max_adverse_excursion_bar_pct   IS NULL )
          AND  entry_price IS NOT NULL
          AND  entry_price > 0
          AND  entry_time  IS NOT NULL
          AND  exit_time   IS NOT NULL
        ORDER  BY entry_time
    """)
    trades = cur.fetchall()

print(f'Trades needing backfill: {len(trades)}')
if not trades:
    print('Nothing to do.')
    conn.close()
    sys.exit(0)

print()

UTC = timezone.utc

# ── Counters ──────────────────────────────────────────────────────────────────
n_processed = n_updated = n_skipped = 0
skip_reasons: dict[str, int] = {}

def _record_skip(reason: str) -> None:
    global n_skipped
    n_skipped += 1
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

# ── Per-trade loop ────────────────────────────────────────────────────────────
for i, trade in enumerate(trades, 1):
    trade_id    = trade['trade_id']
    ticker      = trade['ticker']
    trade_type  = (trade['trade_type'] or 'buy').lower()
    entry_price = float(trade['entry_price'])
    n_processed += 1

    pfx = f'[{i}/{len(trades)}] {ticker:<6} ({trade_type})'

    # ── Parse entry/exit timestamps ───────────────────────────────────────────
    try:
        entry_dt = datetime.fromisoformat(trade['entry_time'])
        exit_dt  = datetime.fromisoformat(trade['exit_time'])
    except Exception as e:
        print(f'  {pfx} SKIP — time parse error: {e}')
        _record_skip('time_parse_error')
        continue

    # Naive ISO strings on Railway are UTC; attach tzinfo so Alpaca accepts them
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=UTC)
    if exit_dt.tzinfo is None:
        exit_dt = exit_dt.replace(tzinfo=UTC)

    if exit_dt <= entry_dt:
        print(f'  {pfx} SKIP — exit_time ({exit_dt}) not after entry_time ({entry_dt})')
        _record_skip('bad_time_range')
        continue

    # ── Fetch 1-minute bars ───────────────────────────────────────────────────
    # No DB transaction is open here — autocommit means the connection is idle
    # while waiting for the Alpaca HTTP response (matching the discipline in
    # compute_and_store_excursion which calls conn.rollback() before the fetch).
    try:
        bars = alpaca.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=entry_dt - timedelta(minutes=2),
            end=exit_dt   + timedelta(minutes=2),
        ))
        df = bars.df.reset_index()
    except Exception as e:
        err_str = str(e)
        is_rate_limit = '429' in err_str or 'rate limit' in err_str.lower()
        sleep_sec = 30 if is_rate_limit else 3
        print(f'  {pfx} SKIP — Alpaca error (sleep {sleep_sec}s): {err_str}')
        _record_skip('alpaca_rate_limit' if is_rate_limit else 'alpaca_error')
        time.sleep(sleep_sec)
        continue

    if df.empty:
        print(f'  {pfx} SKIP — no bars returned from Alpaca')
        _record_skip('no_bars')
        continue

    # ── Filter to hold window (±1 min buffer — matches compute_and_store_excursion) ──
    ts = df['timestamp']
    if hasattr(ts.dt, 'tz') and ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    df = df.assign(timestamp_utc=ts)
    mask = (
        (df['timestamp_utc'] >= entry_dt - timedelta(minutes=1)) &
        (df['timestamp_utc'] <= exit_dt   + timedelta(minutes=1))
    )
    df = df[mask]

    if df.empty:
        print(f'  {pfx} SKIP — no bars in hold window')
        _record_skip('no_bars_in_window')
        continue

    # ── Compute MFE / MAE — identical to Database.compute_and_store_excursion() ──
    max_high = float(df['high'].max())
    min_low  = float(df['low'].min())
    is_long  = trade_type in ('buy', 'long')

    if is_long:
        mfe = (max_high - entry_price) / entry_price
        mae = (min_low  - entry_price) / entry_price
    else:
        mfe = (entry_price - min_low)  / entry_price
        mae = (entry_price - max_high) / entry_price

    mfe = max(mfe, 0.0)   # favorable excursion is always non-negative
    mae = min(mae, 0.0)   # adverse excursion is always non-positive

    entry_str = entry_dt.strftime('%Y-%m-%d %H:%M')
    exit_str  = exit_dt.strftime('%H:%M')
    print(
        f'  {pfx}  {entry_str} → {exit_str}  '
        f'MFE={mfe * 100:+.3f}%  MAE={mae * 100:+.3f}%'
        + ('  [would write]' if DRY_RUN else '  [writing]')
    )

    if DRY_RUN:
        print(
            f'    UPDATE trades '
            f'SET max_favorable_excursion_bar_pct = {mfe:.6f}, '
            f'max_adverse_excursion_bar_pct = {mae:.6f} '
            f"WHERE trade_id = '{trade_id}';"
        )
        n_updated += 1
    else:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE trades '
                    'SET max_favorable_excursion_bar_pct = %s, '
                    '    max_adverse_excursion_bar_pct   = %s '
                    'WHERE trade_id = %s',
                    (mfe, mae, trade_id),
                )
            # autocommit=True: each UPDATE commits as its own transaction —
            # if the script is interrupted it resumes from the last NULL row.
            n_updated += 1
        except Exception as e:
            print(f'    DB write error: {e}')
            _record_skip('db_write_error')

    # Modest sleep between fetches to stay inside Alpaca's rate limits
    time.sleep(0.3)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print('═' * 60)
print(f' {mode_label} Backfill complete')
print('═' * 60)
print(f'  Trades scanned  : {n_processed}')
action_lbl = 'Would update' if DRY_RUN else 'Updated'
print(f'  {action_lbl:<14}: {n_updated}')
print(f'  Skipped         : {n_skipped}')
if skip_reasons:
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f'    {reason}: {count}')
if DRY_RUN and n_updated > 0:
    print()
    print('  Re-run with --execute to write changes to the database.')
print()
conn.close()
