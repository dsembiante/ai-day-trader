"""
config.py — Centralized configuration for the AI trading agent.

All runtime settings, API credentials, risk parameters, and file paths
are defined here. Values are loaded from the .env file at startup via
python-dotenv. Defaults are set for safe paper-trading operation.

Usage:
    from config import config
    key = config.alpaca_api_key
"""

from pydantic import BaseModel
from enum import Enum
import os
from dotenv import load_dotenv

# Load .env into environment before any os.getenv calls
load_dotenv()


# ── Enumerations ──────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    """Controls whether orders are sent to the paper or live Alpaca account."""
    PAPER = 'paper'
    LIVE = 'live'


class RunMode(str, Enum):
    """
    Determines the scheduling strategy for the trading loop.
    - fixed_6x:        Runs at 6 fixed intervals throughout the trading day.
    - intraday_30min:  Runs every 30 minutes during market hours.
    - intraday_10min:  Runs every 10 minutes during market hours (day trading).
    - intraday_smart:  5min at open/close, 15min mid-day (default day trading mode).
    """
    FIXED_6X = 'fixed_6x'
    INTRADAY_30MIN = 'intraday_30min'
    INTRADAY_10MIN = 'intraday_10min'
    INTRADAY_SMART = 'intraday_smart'


class HoldPeriod(str, Enum):
    """
    Classifies a trade's intended holding duration.
    Each period has its own stop-loss, take-profit, and max-days budget
    defined in the Config class below.
    """
    INTRADAY = 'intraday'   # Same-day exit
    SWING = 'swing'          # 2–5 day hold
    POSITION = 'position'    # Multi-week trend trade


# ── Main Config ───────────────────────────────────────────────────────────────

class Config(BaseModel):
    """
    Single source of truth for all application settings.
    Pydantic validates types at instantiation, catching misconfigured
    .env values before they reach trading logic.
    """

    # ── Alpaca Brokerage ──────────────────────────────────────────────────────
    # Keys are loaded from .env — never hard-code credentials here.
    alpaca_api_key: str = os.getenv('ALPACA_API_KEY', '')
    alpaca_secret_key: str = os.getenv('ALPACA_SECRET_KEY', '')
    alpaca_base_url: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Default to paper mode; must be explicitly overridden in .env for live trading
    trading_mode: TradingMode = TradingMode(os.getenv('TRADING_MODE', 'paper'))

    # ── Run Mode ──────────────────────────────────────────────────────────────
    # Controls how the scheduler fires agent cycles during market hours
    run_mode: RunMode = RunMode(os.getenv('RUN_MODE', 'intraday_smart'))

    # ── External Data Sources ─────────────────────────────────────────────────
    finnhub_api_key: str = os.getenv('FINNHUB_API_KEY', '')  # Real-time quotes & news
    fred_api_key: str = os.getenv('FRED_API_KEY', '')        # Macro economic data
    groq_api_key: str = os.getenv('GROQ_API_KEY', '')        # LLM inference

    # ── LLM Settings ──────────────────────────────────────────────────────────
    # llama-3.3-70b-versatile provides strong reasoning at low latency via Groq
    groq_model: str = 'llama-3.3-70b-versatile'
    temperature: float = 0.2    # Low temperature for deterministic trading decisions
    max_tokens: int = 1024      # Reduced from 2048 — cuts token usage per agent call ~50%

    # ── Groq Retry Policy ─────────────────────────────────────────────────────
    # Rate-limit and transient errors are retried with a fixed delay
    groq_max_retries: int = 3
    groq_retry_delay: float = 2.0  # Seconds between retry attempts

    # ── PDT Rule Protection ───────────────────────────────────────────────────
    # Pattern Day Trader rule requires $25,000 minimum balance to place more
    # than 3 intraday round-trips in a 5-day rolling window. Keep False until
    # the account is consistently above $25,000 to avoid PDT violations.
    # Set True only when account balance is reliably above $25,000.
    allow_intraday: bool = True

    # ── Risk Management ───────────────────────────────────────────────────────
    min_position_pct: float = 0.10      # Floor: 10% of portfolio at min confidence (~$4,000 on $40k)
    max_position_pct: float = 0.25      # Ceiling: 25% of portfolio at max confidence (~$10,000 on $40k)
    circuit_breaker_pct: float = 0.10   # Hard stop: halt all trading at 10% drawdown
    confidence_threshold: float = 0.82  # Minimum agent confidence score to enter a trade
    max_positions: int = 15             # Maximum concurrent open positions
    min_signals_required: int = 2       # Minimum agreeing signals before executing a trade
    max_same_direction_positions: int = 15  # Max concurrent longs OR shorts at one time
    loss_cooloff_minutes: int = 15         # Minutes to wait before re-entering a ticker after a losing exit
    profitable_exit_cooldown_minutes: int = 5  # Minutes before re-entering same ticker/direction after a profitable exit

    # ── Hold Period Exit Rules ────────────────────────────────────────────────
    # Each hold period tier has independent stop-loss, take-profit, and time limits.
    # The position_monitor.py module enforces these on every monitoring cycle.

    # Intraday — exits same day; tight stops to limit overnight gap risk
    intraday_stop_loss_pct: float = 0.015   # Short-side fallback (executor sanity check only)
    long_stop_loss_pct: float = 0.012       # Long-side cap: tighter than shorts to cut losers sooner
    intraday_take_profit_pct: float = 0.04
    intraday_max_days: int = 1

    # Swing — short-term momentum; wider stops to absorb normal daily volatility
    swing_stop_loss_pct: float = 0.015
    swing_take_profit_pct: float = 0.025
    swing_max_days: int = 5

    # Position — trend-following; widest stops to stay in strong moves
    position_stop_loss_pct: float = 0.08
    position_take_profit_pct: float = 0.20
    position_max_days: int = 20

    # ── Multi-Strategy Configuration ─────────────────────────────────────────
    gap_fade_enabled: bool = True            # Gap fade strategy (9:45–10:30 AM ET)
    vwap_reversion_enabled: bool = False     # VWAP reversion strategy (12:00–2:30 PM ET)
    gap_fade_min_gap_pct: float = 5.0        # Minimum gap % to qualify for gap fade
    gap_fade_window_end: str = "10:45"       # ET time to stop gap fade entries
    vwap_reversion_window_start: str = "12:00"  # ET time to start VWAP reversion
    vwap_reversion_window_end: str = "14:30"    # ET time to stop VWAP reversion

    # ── Watchlist ─────────────────────────────────────────────────────────────
    # Symbols scanned on every agent cycle. Mix of mega-cap tech, financials,
    # and broad market ETFs for diversified signal generation.
    watchlist: list = [
        'AAPL', 'AMD', 'AMZN', 'AVGO', 'GOOGL',
        'META', 'MSFT', 'MU', 'NFLX', 'NVDA',
        'PLTR', 'QCOM', 'SMCI', 'TSLA', 'UBER',
    ]

    # ── File Paths ────────────────────────────────────────────────────────────
    # Relative to the project root; directories are created at startup if missing
    db_path: str = 'data/trading.db'    # SQLite trade journal
    cache_dir: str = 'data/cache'       # Cached API responses to reduce rate-limit exposure
    reports_dir: str = 'reports'        # Generated PDF reports
    logs_dir: str = 'logs'             # Application and error logs


# ── Singleton Instance ────────────────────────────────────────────────────────
# Import this object throughout the codebase rather than instantiating Config
# directly, ensuring all modules share the same validated configuration.
config = Config()
