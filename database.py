"""
database.py — SQLite trade journal and performance ledger.

Two tables are maintained:
    trades             — One row per trade entry, updated in-place on exit.
    daily_performance  — One row per calendar day, written by the scheduler
                         at end-of-day for report generation.

SQLite is used over a hosted database so the system runs without external
infrastructure (no Postgres, no connection pooling). check_same_thread=False
allows the Streamlit dashboard thread and the scheduler thread to share the
same connection safely for read-heavy dashboard queries.

Usage:
    from database import Database
    db = Database()
    db.insert_trade(trade_dict)
    open_trades = db.get_open_trades()
"""

import sqlite3, os
from datetime import datetime
from config import config


class Database:
    """
    Thin wrapper around a SQLite connection exposing domain-specific query
    methods. All writes use parameterised queries to prevent SQL injection.
    """

    def __init__(self):
        # Ensure the data directory exists before SQLite tries to open the file
        os.makedirs('data', exist_ok=True)

        # check_same_thread=False: Streamlit runs on a separate thread from the
        # scheduler; this flag allows both to share the connection. Safe here
        # because SQLite serialises writes internally.
        self.conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._create_tables()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_tables(self):
        """
        Idempotently create the database schema using IF NOT EXISTS guards.
        Safe to call on every startup — existing data is never modified.

        trades:
            Stores the full lifecycle of every trade from entry to exit.
            Reasoning fields (bull, bear, risk_manager) are preserved verbatim
            for post-trade analysis and dashboard display.

        daily_performance:
            Aggregated end-of-day snapshot written by the scheduler. Used by
            report_generator.py and the Performance tab in the dashboard.
        """
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS trades (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id                TEXT UNIQUE,        -- UUID assigned at entry
                ticker                  TEXT,
                trade_type              TEXT,               -- buy / sell / short / cover
                order_type              TEXT,               -- market / limit / stop
                hold_period             TEXT,               -- intraday / swing / position
                max_hold_days           INTEGER,            -- enforced by position_monitor.py
                entry_price             REAL,
                exit_price              REAL,               -- NULL until trade is closed
                shares                  REAL,
                position_size_usd       REAL,
                stop_loss_price         REAL,
                take_profit_price       REAL,
                pnl                     REAL,               -- Realised P&L in dollars
                pnl_pct                 REAL,               -- Realised P&L as a fraction
                status                  TEXT DEFAULT 'open',
                exit_reason             TEXT,               -- e.g. stop_loss, take_profit, hold_period_expired
                confidence_at_entry     REAL,               -- Risk manager confidence score (0–1)
                bull_reasoning          TEXT,               -- Preserved for audit trail
                bear_reasoning          TEXT,
                risk_manager_reasoning  TEXT,
                hold_period_reasoning   TEXT,
                data_sources_available  TEXT,               -- JSON-serialised DataSourceStatus
                entry_time              TEXT,               -- ISO-8601
                exit_time               TEXT                -- ISO-8601, NULL until closed
            );

            CREATE TABLE IF NOT EXISTS daily_performance (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                date                        TEXT UNIQUE,    -- YYYY-MM-DD
                portfolio_value             REAL,
                daily_pnl                   REAL,
                daily_pnl_pct               REAL,
                total_trades                INTEGER,
                winning_trades              INTEGER,
                losing_trades               INTEGER,
                intraday_trades             INTEGER,        -- Count by hold period for strategy review
                swing_trades                INTEGER,
                position_trades             INTEGER,
                circuit_breaker_triggered   INTEGER DEFAULT 0,  -- 1 if breaker fired this day
                api_failures                TEXT            -- JSON list of failed source names
            );
        ''')
        self.conn.commit()

    # ── Write Operations ──────────────────────────────────────────────────────

    def insert_trade(self, trade: dict):
        """
        Insert a new trade record or replace an existing one with the same trade_id.

        INSERT OR REPLACE handles the edge case where a scheduler retry attempts
        to re-insert a trade that was partially committed in a prior run.

        Args:
            trade: Dict whose keys exactly match the trades table column names.
                   Built and validated by trade_executor.py before calling here.
        """
        placeholders = ','.join(['?' for _ in trade])
        columns = ','.join(trade.keys())
        self.conn.execute(
            f'INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})',
            list(trade.values())
        )
        self.conn.commit()

    def update_trade_status(self, trade_id, status, exit_reason=None, exit_price=None):
        """
        Record the outcome of a closed trade.

        Called by trade_executor.py (stop/take-profit fills) and
        position_monitor.py (hold period expiry, intraday forced close).

        Args:
            trade_id:    UUID of the trade to update.
            status:      New status string, typically 'closed'.
            exit_reason: Human-readable reason code (e.g. 'stop_loss',
                         'take_profit', 'hold_period_expired', 'intraday_forced_close').
            exit_price:  Actual fill price; None if unavailable at close time.
        """
        from datetime import datetime
        self.conn.execute(
            'UPDATE trades SET status=?, exit_reason=?, exit_price=?, exit_time=? '
            'WHERE trade_id=?',
            (status, exit_reason, exit_price, datetime.now().isoformat(), trade_id)
        )
        self.conn.commit()

    # ── Read Operations ───────────────────────────────────────────────────────

    def get_all_trades(self) -> list:
        """
        Return all trades ordered by entry time descending (most recent first).
        Used by the Trade Journal tab in the Streamlit dashboard.
        """
        cur = self.conn.execute('SELECT * FROM trades ORDER BY entry_time DESC')
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_open_trades(self) -> list:
        """
        Return all currently open positions.
        Called by position_monitor.py on every cycle to check exit conditions.
        """
        cur = self.conn.execute("SELECT * FROM trades WHERE status='open'")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_performance_by_hold_period(self) -> dict:
        """
        Aggregate closed trade statistics broken out by hold period tier.

        Returns a dict keyed by hold period name, each containing:
            count:      Number of closed trades in this tier
            total_pnl:  Cumulative realised P&L in dollars
            avg_pnl:    Average P&L per trade in dollars

        Used by the Performance and Analytics tabs in the dashboard to
        compare strategy effectiveness across intraday, swing, and position tiers.
        """
        result = {}
        for hp in ['intraday', 'swing', 'position']:
            cur = self.conn.execute(
                "SELECT COUNT(*), SUM(pnl), AVG(pnl) FROM trades "
                "WHERE status='closed' AND hold_period=?",
                (hp,)
            )
            row = cur.fetchone()
            result[hp] = {
                'count': row[0] or 0,
                'total_pnl': row[1] or 0,
                'avg_pnl': row[2] or 0,
            }
        return result

    def get_daily_performance(self) -> list:
        """
        Return all daily performance snapshots ordered by date descending.
        Used by the Performance tab and report_generator.py for period summaries.
        """
        cur = self.conn.execute('SELECT * FROM daily_performance ORDER BY date DESC')
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
