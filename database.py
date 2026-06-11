"""
database.py — PostgreSQL trade journal and performance ledger.

Two tables are maintained:
    trades             — One row per trade entry, updated in-place on exit.
    daily_performance  — One row per calendar day, written by the scheduler
                         at end-of-day for report generation.

PostgreSQL is used so both the scheduler service and the Streamlit dashboard
service on Railway can share the same database. The connection string is read
from the DATABASE_URL environment variable injected by Railway.

Usage:
    from database import Database
    db = Database()
    db.insert_trade(trade_dict)
    open_trades = db.get_open_trades()
"""

import os
import time
import functools
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from config import config
from logger import log_error


# ── Resilience config ──────────────────────────────────────────────────────────
# connect_timeout=10          : abort if the TCP handshake / auth takes > 10 s
# keepalives*                 : OS-level TCP probes detect a silently-dead peer in
#                               ~80 s (idle=30 + interval=10 × count=5) rather than
#                               waiting for the OS default (hours)
# statement_timeout=30 000 ms : any query still running after 30 s is aborted by
#                               the server — prevents the scheduler main thread from
#                               hanging indefinitely on a half-dead connection
# idle_in_transaction=60 000  : a transaction left open with no new queries for
#                               > 60 s is rolled back automatically — cleans up the
#                               known read-transaction leak without an app-side fix;
#                               safe because normal inter-query pauses are < 1 s
_CONN_KWARGS = dict(
    connect_timeout=10,
    keepalives=1,
    keepalives_idle=30,
    keepalives_interval=10,
    keepalives_count=5,
    options='-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000',
)

_RETRY_DELAYS = (1, 2, 4)   # seconds between reconnect attempts (3 retries total)


def _db_retry(method):
    """
    Wrap a Database public method with reconnect-and-retry on connection errors.

    Trigger: psycopg2.OperationalError or InterfaceError — covers:
      - "SSL connection has been closed unexpectedly"
      - "connection already closed"
      - any TCP-level failure detected mid-query

    Policy: up to 3 retries after the initial attempt, with 1 s / 2 s / 4 s
    backoff. On each retry the stale connection is closed and a fresh one is
    opened. If all four attempts fail the last exception is re-raised to the
    caller — it must NOT be swallowed here.

    Non-connection errors (IntegrityError, ProgrammingError, …) are not caught
    and propagate immediately without retry.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        delays = (0,) + _RETRY_DELAYS   # attempt 0 = no sleep, then 1/2/4 s
        last_exc = None
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                if self.conn.closed:
                    self._reconnect()
                return method(self, *args, **kwargs)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                last_exc = exc
                if attempt == len(delays) - 1:
                    raise
                log_error('db_retry', method.__name__,
                          f'attempt {attempt + 1}/{len(delays) - 1}: {exc}')
                try:
                    self.conn.close()
                except Exception:
                    pass
                try:
                    self._reconnect()
                except Exception:
                    pass   # next iteration re-checks conn.closed and retries
        raise last_exc     # unreachable; satisfies type checkers
    return wrapper


class Database:
    """
    Thin wrapper around a PostgreSQL connection exposing domain-specific query
    methods. All writes use parameterised queries to prevent SQL injection.
    """

    def __init__(self):
        database_url = os.getenv('DATABASE_URL', '')
        if not database_url:
            raise RuntimeError('DATABASE_URL environment variable is not set')
        self._db_url = database_url
        self.conn = self._make_connection()
        self._create_tables()

    def _make_connection(self):
        """Open a fresh psycopg2 connection with all resilience params set."""
        conn = psycopg2.connect(self._db_url, **_CONN_KWARGS)
        conn.autocommit = False
        return conn

    def _reconnect(self):
        """
        Close the stale connection and open a new one.
        Only replaces self.conn on success so callers can re-check conn.closed
        if this raises.
        """
        try:
            self.conn.close()
        except Exception:
            pass
        new_conn = self._make_connection()   # raises OperationalError if unreachable
        self.conn = new_conn                 # assign only after successful connect

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_tables(self):
        """
        Idempotently create the database schema using IF NOT EXISTS guards.
        Safe to call on every startup — existing data is never modified.

        Schema creation and column migrations are committed in separate
        transactions so a failed migration cannot abort the CREATE TABLE
        statements (critical for PostgreSQL on Railway where a failed DDL
        inside a transaction poisons all subsequent commands in that block).
        """
        # ── Step 1: Core schema — committed independently ─────────────────────
        with self.conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id                      SERIAL PRIMARY KEY,
                    trade_id                TEXT UNIQUE,
                    ticker                  TEXT,
                    trade_type              TEXT,
                    order_type              TEXT,
                    hold_period             TEXT,
                    max_hold_days           INTEGER,
                    entry_price             REAL,
                    exit_price              REAL,
                    shares                  REAL,
                    position_size_usd       REAL,
                    stop_loss_price         REAL,
                    take_profit_price       REAL,
                    pnl                     REAL,
                    pnl_pct                 REAL,
                    status                  TEXT DEFAULT 'open',
                    exit_reason             TEXT,
                    confidence_at_entry     REAL,
                    bull_reasoning          TEXT,
                    bear_reasoning          TEXT,
                    risk_manager_reasoning  TEXT,
                    hold_period_reasoning   TEXT,
                    data_sources_available  TEXT,
                    atr_pct                 REAL,
                    entry_time              TEXT,
                    exit_time               TEXT
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id                          SERIAL PRIMARY KEY,
                    date                        TEXT UNIQUE,
                    portfolio_value             REAL,
                    daily_pnl                   REAL,
                    daily_pnl_pct               REAL,
                    total_trades                INTEGER,
                    winning_trades              INTEGER,
                    losing_trades               INTEGER,
                    intraday_trades             INTEGER,
                    swing_trades                INTEGER,
                    position_trades             INTEGER,
                    circuit_breaker_triggered   INTEGER DEFAULT 0,
                    api_failures                TEXT
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                    id          INTEGER PRIMARY KEY DEFAULT 1,
                    peak_value  REAL,
                    updated_at  TEXT
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS blocked_trades (
                    id                      SERIAL PRIMARY KEY,
                    block_time              TIMESTAMP NOT NULL DEFAULT NOW(),
                    ticker                  TEXT NOT NULL,
                    trade_type              TEXT NOT NULL,
                    filter_name             TEXT NOT NULL,
                    confidence              REAL,
                    distance_from_vwap_pct  REAL,
                    last_3_bars_velocity_pct REAL,
                    would_be_entry_price    REAL,
                    strategy_used           TEXT,
                    price_at_1130           REAL,
                    hypothetical_pnl_pct    REAL,
                    backfill_completed_at   TIMESTAMP
                )
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_blocked_trades_block_time
                ON blocked_trades (block_time)
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_blocked_trades_ticker
                ON blocked_trades (ticker)
            ''')
        self.conn.commit()

        # ── Step 2: Column migrations — each in its own isolated transaction ──
        # Uses information_schema to check existence before ALTER TABLE so
        # PostgreSQL never sees a failing DDL that would abort the transaction.
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'atr_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN atr_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()  # Isolated — schema tables already committed above

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'max_favorable_excursion_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN max_favorable_excursion_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'max_adverse_excursion_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN max_adverse_excursion_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS max_favorable_excursion_bar_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS max_adverse_excursion_bar_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'strategy_used'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN strategy_used TEXT")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'vix_at_entry'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN vix_at_entry REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'spy_change_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN spy_change_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'orb_score'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN orb_score INTEGER")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'orb_direction'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN orb_direction TEXT")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'gap_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN gap_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'distance_from_vwap_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN distance_from_vwap_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'distance_from_orb_high_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN distance_from_orb_high_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'distance_from_orb_low_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN distance_from_orb_low_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'minutes_since_orb_breakout'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN minutes_since_orb_breakout REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'last_3_bars_velocity_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN last_3_bars_velocity_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'last_bar_range_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN last_bar_range_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'price_vs_recent_high_pct'
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE trades ADD COLUMN price_vs_recent_high_pct REAL")
            self.conn.commit()
        except Exception:
            self.conn.rollback()

    # ── Write Operations ──────────────────────────────────────────────────────

    @_db_retry
    def insert_trade(self, trade: dict):
        """
        Insert a new trade record or update an existing one with the same trade_id.

        ON CONFLICT DO UPDATE handles the edge case where a scheduler retry
        attempts to re-insert a trade that was partially committed in a prior run.

        Args:
            trade: Dict whose keys exactly match the trades table column names.
                   Built and validated by trade_executor.py before calling here.
        """
        columns = ', '.join(trade.keys())
        placeholders = ', '.join(['%s'] * len(trade))
        update_clause = ', '.join(
            f'{col} = EXCLUDED.{col}' for col in trade.keys() if col != 'trade_id'
        )
        sql = (
            f'INSERT INTO trades ({columns}) VALUES ({placeholders}) '
            f'ON CONFLICT (trade_id) DO UPDATE SET {update_clause}'
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, list(trade.values()))
        self.conn.commit()

    @_db_retry
    def update_trade_status(self, trade_id, status, exit_reason=None, exit_price=None, exit_time_override=None):
        """
        Record the outcome of a closed trade.

        Called by trade_executor.py (stop/take-profit fills) and
        position_monitor.py (hold period expiry, intraday forced close).

        When exit_price is provided, computes pnl and pnl_pct from the stored
        entry_price, shares, and trade_type so the dashboard and reports always
        have accurate P&L figures without requiring callers to calculate it.
        exit_time_override, if provided, is written as exit_time instead of
        datetime.now(). Use this when the actual broker fill timestamp is known.
        """
        pnl = None
        pnl_pct = None

        if exit_price is not None:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    'SELECT entry_price, shares, trade_type FROM trades WHERE trade_id=%s',
                    (trade_id,)
                )
                row = cur.fetchone()
            if row:
                entry_price = row['entry_price']
                shares      = row['shares']
                trade_type  = row.get('trade_type', 'buy') or 'buy'
                if entry_price and shares and entry_price > 0 and shares > 0:
                    is_long = trade_type in ('buy', 'long')
                    pnl     = (exit_price - entry_price) * shares if is_long \
                              else (entry_price - exit_price) * shares
                    pnl_pct = pnl / (entry_price * shares)

        exit_time_val = exit_time_override if exit_time_override is not None else datetime.now().isoformat()
        with self.conn.cursor() as cur:
            cur.execute(
                'UPDATE trades SET status=%s, exit_reason=%s, exit_price=%s, '
                'pnl=%s, pnl_pct=%s, exit_time=%s WHERE trade_id=%s',
                (status, exit_reason, exit_price, pnl, pnl_pct,
                 exit_time_val, trade_id)
            )
        self.conn.commit()

        if status == 'closed':
            self.compute_and_store_excursion(trade_id)

    @_db_retry
    def upgrade_trade_to_swing(self, trade_id):
        """
        Upgrade an intraday trade to a swing trade in place of force-closing it.

        Called by close_all_intraday() when a position has > 3% unrealized gain
        at 3:45 PM in a bull market regime. Sets hold_period to 'swing' and
        max_hold_days to the swing budget from config so _check_hold_expiry()
        enforces the correct exit window going forward.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                'UPDATE trades SET hold_period=%s, max_hold_days=%s WHERE trade_id=%s',
                ('swing', config.swing_max_days, trade_id)
            )
        self.conn.commit()

    # ── Read Operations ───────────────────────────────────────────────────────

    @_db_retry
    def get_all_trades(self) -> list:
        """
        Return all trades ordered by entry time descending (most recent first).
        Used by the Trade Journal tab in the Streamlit dashboard.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM trades ORDER BY entry_time DESC')
            rows = cur.fetchall()
        self.conn.rollback()
        return [dict(row) for row in rows]

    @_db_retry
    def get_open_trades(self) -> list:
        """
        Return all currently open positions.
        Called by position_monitor.py on every cycle to check exit conditions.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trades WHERE status='open'")
            rows = cur.fetchall()
        self.conn.rollback()
        return [dict(row) for row in rows]

    @_db_retry
    def get_losing_gap_fade_tickers_today(self) -> dict:
        """
        Return tickers where a gap_fade trade closed at a loss today (ET).

        Used by the one-strike block to skip re-entry after a losing gap_fade.
        Strict inequality: pnl_pct < 0 only — breakeven exits do not block.

        Returns dict keyed by ticker:
            {'loss': float (dollar P&L), 'exit_time': 'HH:MM AM/PM'}
        """
        today_prefix = datetime.now().strftime('%Y-%m-%d')
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT ticker, pnl, exit_time
                FROM trades
                WHERE strategy_used = 'gap_fade'
                  AND pnl_pct < 0
                  AND status = 'closed'
                  AND exit_time >= %s
                ORDER BY exit_time DESC
            """, (f'{today_prefix}T00:00:00',))
            rows = cur.fetchall()

        self.conn.rollback()
        result = {}
        for row in rows:
            ticker = row['ticker']
            if ticker in result:
                continue  # Keep most recent (DESC order)
            exit_time_str = ''
            try:
                exit_time_str = datetime.fromisoformat(row['exit_time']).strftime('%I:%M %p')
            except Exception:
                pass
            result[ticker] = {
                'loss': float(row['pnl']) if row['pnl'] is not None else 0.0,
                'exit_time': exit_time_str,
            }
        return result

    @_db_retry
    def update_entry_price(self, trade_id: str, entry_price: float):
        """
        Patch the entry_price on an open trade record.
        Called when the price was NULL or $0.00 at insertion time and is
        recovered from Alpaca's order fill history during re-evaluation.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                'UPDATE trades SET entry_price=%s WHERE trade_id=%s',
                (entry_price, trade_id),
            )
        self.conn.commit()

    @_db_retry
    def update_mfe_mae(self, trade_id: str, mfe_pct: 'float | None', mae_pct: 'float | None'):
        """
        Persist the current max favorable and max adverse excursion percentages
        for an open trade. Called by position_monitor on every check cycle.
        Values are fractions (0.015 = 1.5%) matching gain_pct conventions.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                'UPDATE trades SET max_favorable_excursion_pct=%s, max_adverse_excursion_pct=%s WHERE trade_id=%s',
                (mfe_pct, mae_pct, trade_id),
            )
        self.conn.commit()

    @_db_retry
    def compute_and_store_excursion(self, trade_id: str):
        """
        After a trade closes, fetch 1-minute Alpaca bars over the hold period and
        overwrite max_favorable_excursion_pct / max_adverse_excursion_pct with values
        derived from bar highs and lows. Observation-only — swallows all exceptions.

        Values stored as fractions (0.015 = 1.5%) matching the gain_pct convention
        used throughout the codebase. MFE >= 0 (favorable), MAE <= 0 in normal cases.
        """
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    'SELECT ticker, trade_type, entry_price, entry_time, exit_time '
                    'FROM trades WHERE trade_id=%s',
                    (trade_id,)
                )
                row = cur.fetchone()
            # End the SELECT transaction before the Alpaca bar fetch — that call
            # can take 30–60 s and would otherwise idle the transaction past the
            # server's idle_in_transaction_session_timeout, killing the connection.
            self.conn.rollback()

            if not row or not row['entry_price'] or not row['entry_time'] or not row['exit_time']:
                return

            ticker      = row['ticker']
            trade_type  = (row['trade_type'] or 'buy').lower()
            entry_price = float(row['entry_price'])

            # Naive strings are Eastern-naive: the Railway container runs with
            # TZ=America/New_York so datetime.now() returns ET wall-clock time
            # without tzinfo. Use replace(tzinfo=ET) not replace(tzinfo=UTC) to
            # avoid shifting the bar window ~4–5 h into pre-market. Aware strings
            # (e.g. +00:00 from order.filled_at via exit_time_override) are left
            # unchanged by the tzinfo-is-None guard.
            entry_dt = datetime.fromisoformat(row['entry_time'])
            exit_dt  = datetime.fromisoformat(row['exit_time'])
            _ET = ZoneInfo('America/New_York')
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=_ET).astimezone(timezone.utc)
            if exit_dt.tzinfo is None:
                exit_dt = exit_dt.replace(tzinfo=_ET).astimezone(timezone.utc)

            # Fetch 1-minute bars with a 2-minute buffer so partial edge bars are included
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                _alpaca = StockHistoricalDataClient(
                    api_key=config.alpaca_api_key,
                    secret_key=config.alpaca_secret_key,
                )
                bars = _alpaca.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=entry_dt - timedelta(minutes=2),
                    end=exit_dt   + timedelta(minutes=2),
                ))
                df = bars.df.reset_index()
            except Exception as bar_exc:
                log_error('excursion_bar_fetch', ticker, str(bar_exc))
                return

            if df.empty:
                return

            # Align timestamps to UTC then filter to the hold window (±1 min buffer)
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
                return

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

            with self.conn.cursor() as cur:
                cur.execute(
                    'UPDATE trades SET max_favorable_excursion_bar_pct=%s, '
                    'max_adverse_excursion_bar_pct=%s WHERE trade_id=%s',
                    (mfe, mae, trade_id),
                )
            self.conn.commit()

        except Exception as exc:
            try:
                log_error('compute_excursion', trade_id, str(exc))
            except Exception:
                pass

    @_db_retry
    def get_last_closed_trade(self, ticker: str) -> dict | None:
        """
        Return the most recent closed trade for a ticker today, or None if none exists.
        Used by crew.py to enforce the loss cooloff period before re-entering a ticker.
        """
        today = datetime.now().date().isoformat()
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT pnl, exit_time FROM trades
                WHERE ticker = %s
                  AND status = 'closed'
                  AND DATE(exit_time) = %s
                ORDER BY exit_time DESC
                LIMIT 1
                """,
                (ticker, today),
            )
            row = cur.fetchone()
        self.conn.rollback()
        return dict(row) if row else None

    @_db_retry
    def get_recent_closed_trade_by_direction(
        self, ticker: str, trade_type: str, minutes: int = 10
    ) -> dict | None:
        """
        Return the most recent closed trade for ticker in the same direction
        within the last `minutes` minutes, or None if none exists.

        Used by crew.py to enforce the per-direction re-entry cooldown gate.
        trade_type comparison includes both canonical forms ('buy'/'long',
        'short'/'sell_short') so direction matching is robust to LLM variation.
        """
        is_long = trade_type in ('buy', 'long')
        direction_types = ('buy', 'long') if is_long else ('short', 'sell_short')
        # exit_time is stored as a naive ISO string via datetime.now().isoformat()
        # (local time, no timezone).  Casting to ::timestamptz would interpret it
        # as UTC and produce a ~5-hour mismatch on Eastern-time machines.  Instead,
        # build the cutoff in Python so both sides use the same clock.
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT trade_type, exit_time FROM trades
                WHERE ticker = %s
                  AND status = 'closed'
                  AND trade_type = ANY(%s)
                  AND exit_time >= %s
                ORDER BY exit_time DESC
                LIMIT 1
                """,
                (ticker, list(direction_types), cutoff),
            )
            row = cur.fetchone()
        self.conn.rollback()
        return dict(row) if row else None

    @_db_retry
    def get_performance_by_hold_period(self) -> dict:
        """
        Aggregate closed trade statistics broken out by hold period tier.
        """
        result = {}
        for hp in ['intraday', 'swing', 'position']:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*), SUM(pnl), AVG(pnl) FROM trades "
                    "WHERE status='closed' AND hold_period=%s",
                    (hp,)
                )
                row = cur.fetchone()
            result[hp] = {
                'count': row[0] or 0,
                'total_pnl': row[1] or 0,
                'avg_pnl': row[2] or 0,
            }
        self.conn.rollback()
        return result

    @_db_retry
    def get_performance_metrics(self) -> dict:
        """
        Return overall aggregate performance stats across all closed trades.
        Called by report_generator.py when building any report section that
        needs portfolio-level summary figures.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), SUM(pnl), AVG(pnl), "
                "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), "
                "SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) "
                "FROM trades WHERE status='closed'"
            )
            row = cur.fetchone()
        self.conn.rollback()
        total      = row[0] or 0
        gross_win  = row[4] or 0
        gross_loss = row[5] or 0
        return {
            'total_trades':  total,
            'total_pnl':     row[1] or 0,
            'avg_pnl':       row[2] or 0,
            'win_rate':      (row[3] / total) if total > 0 else 0,
            'profit_factor': (gross_win / gross_loss) if gross_loss > 0 else 0,
        }

    @_db_retry
    def get_circuit_breaker_peak(self):
        """
        Return the stored portfolio peak value, or None if not yet set.
        Called by CircuitBreaker on startup to restore the high-water mark.
        """
        with self.conn.cursor() as cur:
            cur.execute('SELECT peak_value FROM circuit_breaker_state WHERE id = 1')
            row = cur.fetchone()
        self.conn.rollback()
        return row[0] if row else None

    @_db_retry
    def set_circuit_breaker_peak(self, peak_value: float):
        """
        Upsert the portfolio peak value so it survives service restarts/redeploys.
        Called by CircuitBreaker whenever a new high-water mark is recorded.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                'INSERT INTO circuit_breaker_state (id, peak_value, updated_at) VALUES (1, %s, %s) '
                'ON CONFLICT (id) DO UPDATE SET peak_value = EXCLUDED.peak_value, updated_at = EXCLUDED.updated_at',
                (peak_value, datetime.now().isoformat())
            )
        self.conn.commit()

    @_db_retry
    def save_daily_performance(self, portfolio_value: float):
        """
        Insert or update today's daily performance snapshot.

        Aggregates all closed trades for the current calendar day from the
        trades table, counts API errors and circuit breaker events from
        errors.log, and upserts one row into daily_performance.

        ON CONFLICT DO UPDATE ensures the row is refreshed if this is called
        more than once on the same date (e.g. a test run followed by EOD).

        Args:
            portfolio_value: End-of-day Alpaca account balance in USD.
        """
        today = datetime.now().date().isoformat()

        # ── Aggregate today's closed trades ───────────────────────────────────
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                                  AS total_trades,
                    COALESCE(SUM(pnl), 0)                                     AS daily_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)                  AS winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END)                  AS losing_trades,
                    SUM(CASE WHEN hold_period = 'intraday'  THEN 1 ELSE 0 END) AS intraday_trades,
                    SUM(CASE WHEN hold_period = 'swing'     THEN 1 ELSE 0 END) AS swing_trades,
                    SUM(CASE WHEN hold_period = 'position'  THEN 1 ELSE 0 END) AS position_trades
                FROM trades
                WHERE status = 'closed'
                  AND DATE(exit_time) = %s
                """,
                (today,),
            )
            row = cur.fetchone()

        total_trades    = row[0] or 0
        daily_pnl       = float(row[1] or 0.0)
        winning_trades  = row[2] or 0
        losing_trades   = row[3] or 0
        intraday_trades = row[4] or 0
        swing_trades    = row[5] or 0
        position_trades = row[6] or 0

        # Starting portfolio value approximation: end-of-day balance minus gains
        starting_value = portfolio_value - daily_pnl
        daily_pnl_pct  = (daily_pnl / starting_value) if starting_value > 0 else 0.0

        # ── API errors and circuit breaker from errors.log ────────────────────
        api_failure_count      = 0
        circuit_breaker_fired  = 0
        error_file = os.path.join(config.logs_dir, 'errors.log')
        if os.path.exists(error_file):
            with open(error_file) as f:
                for line in f:
                    if today in line:
                        api_failure_count += 1
                        if 'CIRCUIT_BREAKER' in line.upper():
                            circuit_breaker_fired = 1

        # ── Upsert ────────────────────────────────────────────────────────────
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daily_performance
                    (date, portfolio_value, daily_pnl, daily_pnl_pct,
                     total_trades, winning_trades, losing_trades,
                     intraday_trades, swing_trades, position_trades,
                     circuit_breaker_triggered, api_failures)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    portfolio_value           = EXCLUDED.portfolio_value,
                    daily_pnl                 = EXCLUDED.daily_pnl,
                    daily_pnl_pct             = EXCLUDED.daily_pnl_pct,
                    total_trades              = EXCLUDED.total_trades,
                    winning_trades            = EXCLUDED.winning_trades,
                    losing_trades             = EXCLUDED.losing_trades,
                    intraday_trades           = EXCLUDED.intraday_trades,
                    swing_trades              = EXCLUDED.swing_trades,
                    position_trades           = EXCLUDED.position_trades,
                    circuit_breaker_triggered = EXCLUDED.circuit_breaker_triggered,
                    api_failures              = EXCLUDED.api_failures
                """,
                (
                    today, portfolio_value, daily_pnl, daily_pnl_pct,
                    total_trades, winning_trades, losing_trades,
                    intraday_trades, swing_trades, position_trades,
                    circuit_breaker_fired, str(api_failure_count),
                ),
            )
        self.conn.commit()

    @_db_retry
    def get_daily_performance(self) -> list:
        """
        Return all daily performance snapshots ordered by date descending.
        Used by the Performance tab and report_generator.py for period summaries.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM daily_performance ORDER BY date DESC')
            rows = cur.fetchall()
        self.conn.rollback()
        return [dict(row) for row in rows]
