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
import psycopg2
import psycopg2.extras
from datetime import datetime
from config import config


class Database:
    """
    Thin wrapper around a PostgreSQL connection exposing domain-specific query
    methods. All writes use parameterised queries to prevent SQL injection.
    """

    def __init__(self):
        database_url = os.getenv('DATABASE_URL', '')
        if not database_url:
            raise RuntimeError('DATABASE_URL environment variable is not set')

        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = False
        self._create_tables()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_tables(self):
        """
        Idempotently create the database schema using IF NOT EXISTS guards.
        Safe to call on every startup — existing data is never modified.
        """
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
                    status                  TEXT DEFAULT \'open\',
                    exit_reason             TEXT,
                    confidence_at_entry     REAL,
                    bull_reasoning          TEXT,
                    bear_reasoning          TEXT,
                    risk_manager_reasoning  TEXT,
                    hold_period_reasoning   TEXT,
                    data_sources_available  TEXT,
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
        self.conn.commit()

    # ── Write Operations ──────────────────────────────────────────────────────

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

    def update_trade_status(self, trade_id, status, exit_reason=None, exit_price=None):
        """
        Record the outcome of a closed trade.

        Called by trade_executor.py (stop/take-profit fills) and
        position_monitor.py (hold period expiry, intraday forced close).
        """
        with self.conn.cursor() as cur:
            cur.execute(
                'UPDATE trades SET status=%s, exit_reason=%s, exit_price=%s, exit_time=%s '
                'WHERE trade_id=%s',
                (status, exit_reason, exit_price, datetime.now().isoformat(), trade_id)
            )
        self.conn.commit()

    # ── Read Operations ───────────────────────────────────────────────────────

    def get_all_trades(self) -> list:
        """
        Return all trades ordered by entry time descending (most recent first).
        Used by the Trade Journal tab in the Streamlit dashboard.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM trades ORDER BY entry_time DESC')
            return [dict(row) for row in cur.fetchall()]

    def get_open_trades(self) -> list:
        """
        Return all currently open positions.
        Called by position_monitor.py on every cycle to check exit conditions.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trades WHERE status='open'")
            return [dict(row) for row in cur.fetchall()]

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
        return result

    def get_performance_metrics(self) -> dict:
        """
        Return overall aggregate performance stats across all closed trades.
        Called by report_generator.py when building any report section that
        needs portfolio-level summary figures.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), SUM(pnl), AVG(pnl), "
                "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) "
                "FROM trades WHERE status='closed'"
            )
            row = cur.fetchone()
        total = row[0] or 0
        return {
            'total_trades': total,
            'total_pnl':    row[1] or 0,
            'avg_pnl':      row[2] or 0,
            'win_rate':     (row[3] / total) if total > 0 else 0,
        }

    def get_daily_performance(self) -> list:
        """
        Return all daily performance snapshots ordered by date descending.
        Used by the Performance tab and report_generator.py for period summaries.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM daily_performance ORDER BY date DESC')
            return [dict(row) for row in cur.fetchall()]
