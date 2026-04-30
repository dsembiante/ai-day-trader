"""
mfe_reconstruction.py — One-time true MFE/MAE reconstruction for all closed trades.

For each closed trade, fetches 1-minute Alpaca bars covering the hold period
and computes:
  - True MFE (max favorable excursion): intraday peak in the trade's direction
  - True MAE (max adverse excursion):   intraday trough against the trade's direction

Usage:
    # Pass public Railway URL as first argument:
    python mfe_reconstruction.py "postgresql://postgres:PASSWORD@roundhouse.proxy.rlwy.net:PORT/railway"

    # Or set DATABASE_URL_PUBLIC in .env and run:
    python mfe_reconstruction.py

Output:
    data/mfe_reconstruction.csv   — full per-trade table
    Bucketed summary printed to stdout
"""

import os
import csv
import sys
import time
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import config

load_dotenv()

# ── Connection URL ─────────────────────────────────────────────────────────────
# Priority: command-line arg > DATABASE_URL_PUBLIC env var > DATABASE_URL
if len(sys.argv) > 1:
    DATABASE_URL = sys.argv[1]
elif os.getenv('DATABASE_URL_PUBLIC'):
    DATABASE_URL = os.getenv('DATABASE_URL_PUBLIC')
else:
    DATABASE_URL = os.getenv('DATABASE_URL', '')

if not DATABASE_URL:
    sys.exit('ERROR: No database URL. Pass as argument or set DATABASE_URL_PUBLIC in .env')

print(f'Connecting to database...')
conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True

# ── Alpaca data client ────────────────────────────────────────────────────────
alpaca = StockHistoricalDataClient(
    api_key=config.alpaca_api_key,
    secret_key=config.alpaca_secret_key,
)

# ── Fetch all closed trades ───────────────────────────────────────────────────
with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
    cur.execute("""
        SELECT trade_id, ticker, trade_type,
               entry_price, exit_price,
               entry_time, exit_time,
               pnl_pct, exit_reason
        FROM trades
        WHERE status = 'closed'
          AND entry_price IS NOT NULL
          AND exit_price  IS NOT NULL
          AND entry_time  IS NOT NULL
          AND exit_time   IS NOT NULL
          AND pnl_pct     IS NOT NULL
        ORDER BY entry_time
    """)
    trades = cur.fetchall()

print(f'Found {len(trades)} closed trades with complete data')
if not trades:
    sys.exit('Nothing to reconstruct.')

# ── Per-trade MFE/MAE reconstruction ─────────────────────────────────────────
UTC = timezone.utc
rows = []
skipped = []

for i, trade in enumerate(trades, 1):
    trade_id    = trade['trade_id']
    ticker      = trade['ticker']
    trade_type  = trade['trade_type']
    entry_price = float(trade['entry_price'])
    exit_price  = float(trade['exit_price'])
    pnl_pct_raw = float(trade['pnl_pct'])   # decimal fraction, e.g. 0.015 = 1.5%
    exit_reason = trade['exit_reason'] or ''

    # ── Parse entry/exit times ────────────────────────────────────────────────
    # Stored as TEXT via datetime.now().isoformat() on Railway (UTC).
    # Some rows may carry an explicit +00:00 offset from exit_time_override.
    try:
        entry_dt = datetime.fromisoformat(trade['entry_time'])
        exit_dt  = datetime.fromisoformat(trade['exit_time'])
    except Exception as e:
        print(f'  [{i}/{len(trades)}] SKIP {trade_id} — time parse error: {e}')
        skipped.append((trade_id, ticker, f'time parse: {e}'))
        continue

    # Ensure UTC-aware for Alpaca query
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=UTC)
    if exit_dt.tzinfo is None:
        exit_dt = exit_dt.replace(tzinfo=UTC)

    hold_minutes = max((exit_dt - entry_dt).total_seconds() / 60, 1.0)

    # ── Fetch 1-minute bars covering the hold period ──────────────────────────
    # 2-minute buffer on each side so the entry/exit bar is always included
    # even when the timestamp falls mid-minute.
    try:
        bars = alpaca.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=entry_dt - timedelta(minutes=2),
            end=exit_dt   + timedelta(minutes=2),
        ))
        df = bars.df.reset_index()
    except Exception as e:
        print(f'  [{i}/{len(trades)}] SKIP {trade_id} {ticker} — Alpaca error: {e}')
        skipped.append((trade_id, ticker, f'Alpaca: {e}'))
        time.sleep(1)  # back-off on API errors
        continue

    if df.empty:
        print(f'  [{i}/{len(trades)}] SKIP {trade_id} {ticker} — no bars returned')
        skipped.append((trade_id, ticker, 'no bars'))
        continue

    # Align timestamps to UTC for filtering
    ts = df['timestamp']
    if hasattr(ts.dt, 'tz') and ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    df = df.assign(timestamp_utc=ts)

    # Strict filter: bars whose 1-minute window overlaps the hold period.
    # Use entry_dt-1min to include the bar that was open at entry.
    mask = (df['timestamp_utc'] >= entry_dt - timedelta(minutes=1)) & \
           (df['timestamp_utc'] <= exit_dt   + timedelta(minutes=1))
    df = df[mask]

    if df.empty:
        print(f'  [{i}/{len(trades)}] SKIP {trade_id} {ticker} — no bars in hold window')
        skipped.append((trade_id, ticker, 'no bars in window'))
        continue

    # ── MFE / MAE from intraday highs and lows ────────────────────────────────
    max_high = float(df['high'].max())
    min_low  = float(df['low'].min())

    is_long = trade_type.lower() in ('buy', 'long')

    if is_long:
        true_mfe = (max_high   - entry_price) / entry_price
        true_mae = (entry_price - min_low)    / entry_price
    else:
        true_mfe = (entry_price - min_low)    / entry_price
        true_mae = (max_high   - entry_price) / entry_price

    # MFE can't be negative (if price never moved favorably, MFE = 0)
    true_mfe = max(true_mfe, 0.0)
    # MAE can't be negative
    true_mae = max(true_mae, 0.0)

    # Convert to percentage
    true_mfe_pct  = true_mfe  * 100
    true_mae_pct  = true_mae  * 100
    final_pnl_pct = pnl_pct_raw * 100

    rows.append({
        'trade_id':              trade_id,
        'ticker':                ticker,
        'trade_type':            trade_type,
        'entry_price':           round(entry_price, 4),
        'exit_price':            round(exit_price,  4),
        'hold_duration_minutes': round(hold_minutes, 1),
        'final_pnl_pct':         round(final_pnl_pct, 4),
        'true_mfe_pct':          round(true_mfe_pct,  4),
        'true_mae_pct':          round(true_mae_pct,  4),
        'exit_reason':           exit_reason,
    })

    print(
        f'  [{i}/{len(trades)}] {ticker:<5} {trade_type:<5} '
        f'hold={hold_minutes:5.1f}min  '
        f'PnL={final_pnl_pct:+7.3f}%  '
        f'MFE={true_mfe_pct:6.3f}%  '
        f'MAE={true_mae_pct:6.3f}%  '
        f'{exit_reason}'
    )

# ── Write CSV ─────────────────────────────────────────────────────────────────
csv_path = 'data/mfe_reconstruction.csv'
fieldnames = [
    'trade_id', 'ticker', 'trade_type', 'entry_price', 'exit_price',
    'hold_duration_minutes', 'final_pnl_pct', 'true_mfe_pct', 'true_mae_pct',
    'exit_reason',
]
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'\nWrote {len(rows)} rows → {csv_path}')
if skipped:
    print(f'Skipped {len(skipped)}: {skipped}')

# ── Bucketed analysis ─────────────────────────────────────────────────────────

def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

losing  = [r for r in rows if r['final_pnl_pct'] <  0]
winning = [r for r in rows if r['final_pnl_pct'] >  0]
flat    = [r for r in rows if r['final_pnl_pct'] == 0]

print(f'\n{"═"*70}')
print(f' LOSING TRADES  n={len(losing)}   WINNING TRADES  n={len(winning)}   FLAT  n={len(flat)}')
print(f'{"═"*70}')

# ── Losing: MFE buckets ───────────────────────────────────────────────────────
lose_buckets = [
    (0.00, 0.10, 'MFE 0.00–0.10%  (straight-down)'),
    (0.10, 0.20, 'MFE 0.10–0.20%  (brief positive)'),
    (0.20, 0.35, 'MFE 0.20–0.35%  (partial range)'),
    (0.35, 0.60, 'MFE 0.35–0.60%  (medium peak)'),
    (0.60, 1.00, 'MFE 0.60–1.00%  (high peak)'),
    (1.00, 9999, 'MFE > 1.00%     (very high peak)'),
]

print(f'\n LOSING TRADES — by true MFE bucket')
print(f'  {"Bucket":<38} {"n":>4}  {"avg MFE":>8}  {"avg MAE":>8}  {"avg PnL":>8}')
print(f'  {"-"*38} {"----":>4}  {"-------":>8}  {"-------":>8}  {"-------":>8}')
for lo, hi, label in lose_buckets:
    group = [r for r in losing if lo <= r['true_mfe_pct'] < hi]
    if group:
        print(
            f'  {label:<38} {len(group):>4}  '
            f'{avg([r["true_mfe_pct"] for r in group]):>7.3f}%  '
            f'{avg([r["true_mae_pct"] for r in group]):>7.3f}%  '
            f'{avg([r["final_pnl_pct"] for r in group]):>7.3f}%'
        )
    else:
        print(f'  {label:<38} {0:>4}')

# ── Winning: final P&L buckets ────────────────────────────────────────────────
win_buckets = [
    (0.00, 0.20, 'PnL 0.00–0.20%'),
    (0.20, 0.35, 'PnL 0.20–0.35%'),
    (0.35, 0.60, 'PnL 0.35–0.60%'),
    (0.60, 1.00, 'PnL 0.60–1.00%'),
    (1.00, 9999, 'PnL > 1.00%'),
]

print(f'\n WINNING TRADES — by final P&L bucket (with MFE for exit quality)')
print(f'  {"Bucket":<22} {"n":>4}  {"avg PnL":>8}  {"avg MFE":>8}  {"MFE-PnL gap":>11}')
print(f'  {"-"*22} {"----":>4}  {"-------":>8}  {"-------":>8}  {"-----------":>11}')
for lo, hi, label in win_buckets:
    group = [r for r in winning if lo <= r['final_pnl_pct'] < hi]
    if group:
        a_pnl = avg([r['final_pnl_pct'] for r in group])
        a_mfe = avg([r['true_mfe_pct']  for r in group])
        print(
            f'  {label:<22} {len(group):>4}  '
            f'{a_pnl:>7.3f}%  '
            f'{a_mfe:>7.3f}%  '
            f'{a_mfe - a_pnl:>10.3f}%'
        )
    else:
        print(f'  {label:<22} {0:>4}')

# ── Winning: per-trade MFE vs exit detail ─────────────────────────────────────
print(f'\n WINNING TRADES — per-trade exit quality (MFE vs final PnL)')
print(f'  {"ticker":<6} {"hold_min":>8}  {"PnL%":>7}  {"MFE%":>7}  {"gap%":>7}  exit_reason')
print(f'  {"-"*6} {"--------":>8}  {"------":>7}  {"------":>7}  {"------":>7}  {"----------"}')
for r in sorted(winning, key=lambda x: x['true_mfe_pct'], reverse=True):
    gap = r['true_mfe_pct'] - r['final_pnl_pct']
    print(
        f'  {r["ticker"]:<6} {r["hold_duration_minutes"]:>8.1f}  '
        f'{r["final_pnl_pct"]:>+7.3f}%  '
        f'{r["true_mfe_pct"]:>7.3f}%  '
        f'{gap:>+7.3f}%  '
        f'{r["exit_reason"]}'
    )

# ── Summary stats ─────────────────────────────────────────────────────────────
if losing:
    actionable = [r for r in losing if r['true_mfe_pct'] >= 0.20]
    print(f'\n LOSING TRADES WITH MFE >= 0.20% (partial exits could have helped): '
          f'{len(actionable)}/{len(losing)} '
          f'({100*len(actionable)/len(losing):.0f}%)')

if winning:
    early_exit = [r for r in winning if (r['true_mfe_pct'] - r['final_pnl_pct']) > 0.15]
    print(f' WINNING TRADES where MFE exceeded exit by >0.15% (left money on table): '
          f'{len(early_exit)}/{len(winning)} '
          f'({100*len(early_exit)/len(winning):.0f}%)')

print()
conn.close()
