#!/usr/bin/env python3
"""
backfill_trades.py — One-time script to recover exit prices for historical trades.

Problem:
    Before the fix in database.update_trade_status(), bracket order exits (stop-loss
    and take-profit fills) were never recorded in the DB. Those trade rows have
    exit_price=NULL, pnl=NULL, and status='open' even though the position is gone,
    causing the dashboard to show $0 P&L and 0% win rate.

Solution:
    1. Query all trades WHERE exit_price IS NULL
    2. For each unique ticker, fetch all closed/filled orders from Alpaca
    3. Match each DB trade to its exit order by side + timestamp
       (exit order filled_at must be after trade entry_time)
    4. Update the DB with exit_price, pnl, pnl_pct, status='closed', exit_reason

Usage:
    python backfill_trades.py                    # Dry run — prints what would change
    python backfill_trades.py --apply            # Write updates to the database
    python backfill_trades.py --ticker NVDA      # Dry run for one ticker
    python backfill_trades.py --ticker NVDA --apply
"""

import argparse
import sys
from datetime import datetime, timezone
from collections import defaultdict

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from config import config
from database import Database


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_utc(dt: datetime) -> datetime:
    """Return a UTC-aware datetime regardless of whether dt is naive or aware."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_entry_time(entry_time_str: str) -> datetime | None:
    """Parse the ISO entry_time string from the DB into a UTC-aware datetime."""
    if not entry_time_str:
        return None
    try:
        dt = datetime.fromisoformat(entry_time_str)
        return _to_utc(dt)
    except Exception:
        return None


def _order_side_str(order) -> str:
    """Return 'buy' or 'sell' from an Alpaca order's side enum."""
    return str(order.side).split('.')[-1].lower()


def _order_type_str(order) -> str:
    """Return 'limit', 'stop', 'market', etc. from an Alpaca order's type enum."""
    return str(order.type).split('.')[-1].lower()


def _order_filled_at(order) -> datetime | None:
    """
    Return the UTC-aware filled_at time for an order.

    Prefer filled_at; fall back to submitted_at if filled_at is absent (some
    older Alpaca SDK versions don't expose filled_at on every order object).
    """
    ts = getattr(order, 'filled_at', None) or getattr(order, 'submitted_at', None)
    return _to_utc(ts) if ts else None


def _classify_exit(exit_order, trade: dict) -> str:
    """
    Infer the exit reason from the order type.

    Bracket legs are limit (take-profit) or stop/stop_limit (stop-loss).
    Manual market closes from our code are market orders.
    """
    otype = _order_type_str(exit_order)
    if 'stop' in otype:
        return 'bracket_stop_loss'
    if otype == 'limit':
        return 'bracket_take_profit'
    return 'bracket_fill'  # market or unknown — grouped as generic bracket fill


def _compute_pnl(trade: dict, exit_price: float) -> tuple[float | None, float | None]:
    """Compute dollar P&L and percentage P&L for a closed trade."""
    entry_price = trade.get('entry_price')
    shares = trade.get('shares')
    trade_type = (trade.get('trade_type') or 'buy').lower()

    if not entry_price or not shares or entry_price <= 0 or shares <= 0:
        return None, None

    is_long = trade_type in ('buy', 'long')
    pnl = (exit_price - entry_price) * shares if is_long \
          else (entry_price - exit_price) * shares
    pnl_pct = pnl / (entry_price * shares)
    return pnl, pnl_pct


# ── Alpaca Order Fetching ──────────────────────────────────────────────────────

def fetch_closed_orders_for_ticker(client: TradingClient, ticker: str) -> list:
    """
    Fetch up to 500 closed (filled) orders for a given ticker, sorted by
    fill time ascending (oldest first) so we can scan chronologically.

    Cancelled orders are included in CLOSED status but filtered out below
    because their filled_avg_price is None.
    """
    try:
        req = GetOrdersRequest(
            symbols=[ticker],
            status=QueryOrderStatus.CLOSED,
            limit=500,
        )
        orders = client.get_orders(filter=req)

        # Keep only actually filled orders (not cancelled)
        filled = [o for o in orders if o.filled_avg_price is not None]

        # Sort ascending by fill time so we find the first exit after entry
        filled.sort(key=lambda o: (_order_filled_at(o) or datetime.min.replace(tzinfo=timezone.utc)))
        return filled

    except Exception as e:
        print(f'  ⚠️  Alpaca fetch failed for {ticker}: {e}')
        return []


def find_exit_order(orders: list, trade: dict) -> object | None:
    """
    Return the first Alpaca order that represents the closing fill for this trade.

    Matching rules:
        1. Side must be the opposite of the trade's entry direction:
             Long (buy) entry  → exit side = sell
             Short entry       → exit side = buy
        2. filled_at must be strictly after entry_time (position wasn't open yet
           at any earlier fill)
        3. First qualifying order by fill time wins (handles the case where the
           same ticker was traded multiple times — each exit matches its own entry)

    Intentionally excludes entry-type orders: for a long, any sell after
    entry_time is a candidate; a new buy would be a re-entry, not the exit.
    """
    entry_dt = _parse_entry_time(trade.get('entry_time'))
    if entry_dt is None:
        return None

    trade_type = (trade.get('trade_type') or 'buy').lower()
    is_long = trade_type in ('buy', 'long')
    exit_side = 'sell' if is_long else 'buy'

    for order in orders:
        if _order_side_str(order) != exit_side:
            continue  # Wrong direction

        filled_at = _order_filled_at(order)
        if filled_at is None or filled_at <= entry_dt:
            continue  # Not filled yet, or filled before our entry

        return order  # First qualifying fill after entry_time

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Backfill exit prices for trades with NULL exit_price'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Write updates to the database (default: dry run, no writes)',
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        metavar='SYMBOL',
        help='Limit backfill to a single ticker symbol (e.g. --ticker NVDA)',
    )
    args = parser.parse_args()

    dry_run = not args.apply
    mode_label = 'DRY RUN' if dry_run else 'APPLY'
    print(f'\n{"=" * 60}')
    print(f'  backfill_trades.py  [{mode_label}]')
    print(f'{"=" * 60}')
    if dry_run:
        print('  No changes will be written. Pass --apply to commit.\n')
    else:
        print('  Updates will be written to the database.\n')

    # ── Connect ────────────────────────────────────────────────────────────────
    try:
        db = Database()
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        sys.exit(1)

    client = TradingClient(
        config.alpaca_api_key,
        config.alpaca_secret_key,
        paper=config.trading_mode == 'paper',
    )

    # ── Fetch NULL trades ──────────────────────────────────────────────────────
    all_trades = db.get_all_trades()
    null_trades = [t for t in all_trades if t.get('exit_price') is None]

    if args.ticker:
        null_trades = [t for t in null_trades if t.get('ticker') == args.ticker.upper()]

    if not null_trades:
        ticker_note = f' for {args.ticker.upper()}' if args.ticker else ''
        print(f'✅ No trades with NULL exit_price found{ticker_note}. Nothing to backfill.')
        return

    print(f'Found {len(null_trades)} trade(s) with NULL exit_price.\n')

    # ── Group by ticker to batch Alpaca API calls ─────────────────────────────
    by_ticker: dict[str, list] = defaultdict(list)
    for trade in null_trades:
        ticker = trade.get('ticker')
        if ticker:
            by_ticker[ticker].append(trade)

    # ── Process each ticker ────────────────────────────────────────────────────
    updated = 0
    no_match = 0

    for ticker, trades in sorted(by_ticker.items()):
        print(f'── {ticker} ({len(trades)} trade(s)) ────────────────────────')

        closed_orders = fetch_closed_orders_for_ticker(client, ticker)
        print(f'  Fetched {len(closed_orders)} closed/filled Alpaca order(s)')

        # Sort trades oldest-first so find_exit_order() advances through
        # orders chronologically and doesn't re-use the same fill
        trades_sorted = sorted(
            trades,
            key=lambda t: _parse_entry_time(t.get('entry_time')) or datetime.min.replace(tzinfo=timezone.utc),
        )

        # Track which orders have already been claimed to avoid double-matching
        # when the same ticker was traded multiple times
        claimed_order_ids: set = set()

        for trade in trades_sorted:
            trade_id    = trade.get('trade_id')
            entry_price = trade.get('entry_price')
            shares      = trade.get('shares')
            entry_time  = trade.get('entry_time')
            trade_type  = (trade.get('trade_type') or 'buy').lower()
            hold_period = trade.get('hold_period', '?')

            print(
                f'\n  Trade {trade_id}  |  {trade_type.upper()}  |  '
                f'entry ${entry_price}  x  {shares} shares  |  '
                f'entered {entry_time}  |  hold={hold_period}'
            )

            # Find a matching exit order not yet claimed by an earlier trade
            unclaimed = [o for o in closed_orders if str(o.id) not in claimed_order_ids]
            exit_order = find_exit_order(unclaimed, trade)

            if exit_order is None:
                print(f'  ⚠️  No matching exit order found — skipping')
                no_match += 1
                continue

            exit_price  = float(exit_order.filled_avg_price)
            filled_at   = _order_filled_at(exit_order)
            exit_reason = _classify_exit(exit_order, trade)
            pnl, pnl_pct = _compute_pnl(trade, exit_price)
            order_type  = _order_type_str(exit_order)

            # Mark this order as claimed so a later trade for the same ticker
            # doesn't match the same fill
            claimed_order_ids.add(str(exit_order.id))

            # Build the result line
            pnl_str = f'${pnl:+.2f} ({pnl_pct*100:+.2f}%)' if pnl is not None else 'pnl=unknown'
            print(
                f'  ✅ Match: order {exit_order.id}  |  {order_type}  '
                f'filled @ ${exit_price:.4f}  on {filled_at}'
            )
            print(f'     → {exit_reason}  |  P&L: {pnl_str}')

            if not dry_run:
                try:
                    db.update_trade_status(
                        trade_id,
                        status='closed',
                        exit_reason=exit_reason,
                        exit_price=exit_price,
                    )
                    updated += 1
                    print(f'     ✓ DB updated')
                except Exception as e:
                    print(f'     ❌ DB update failed: {e}')
            else:
                updated += 1  # Count as "would update" in dry run

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    if dry_run:
        print(f'  DRY RUN complete — {updated} trade(s) would be updated, {no_match} skipped (no order match).')
        print(f'  Run with --apply to write these changes.')
    else:
        print(f'  APPLY complete — {updated} trade(s) updated, {no_match} skipped (no order match).')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
