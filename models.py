"""
models.py — Pydantic schemas for the AI trading agent.

These models serve as the data contracts between every layer of the system:
data collection → agent analysis → trade decisions → execution.
Pydantic validates and coerces all values at construction time, ensuring
bad data is caught before it reaches order placement logic.

Usage:
    from models import MarketData, TradeDecision, AgentAnalysis
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum
from config import HoldPeriod


# ── Trade & Order Enumerations ────────────────────────────────────────────────

class TradeType(str, Enum):
    """Direction of a trade instruction."""
    BUY = 'buy'       # Open a long position
    SELL = 'sell'     # Close a long position
    SHORT = 'short'   # Open a short position
    COVER = 'cover'   # Close a short position


class OrderType(str, Enum):
    """
    Alpaca order execution type.
    Default is LIMIT to avoid adverse fills on illiquid opens/closes.
    """
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


# ── Data Source Health ────────────────────────────────────────────────────────

class DataSourceStatus(BaseModel):
    """
    Tracks which external data sources were reachable during a collection cycle.
    data_collector.py degrades gracefully when sources are down and records the
    outage here so downstream agents know which signals may be missing.
    """
    alpaca: bool = True    # Real-time price & account data
    finnhub: bool = True   # News sentiment & alternative data
    yfinance: bool = True  # Technical indicators & fundamentals
    fred: bool = True      # Macro economic series
    groq: bool = True      # LLM inference availability


# ── Market Data ───────────────────────────────────────────────────────────────

class MarketData(BaseModel):
    """
    Aggregated snapshot of a single ticker at a point in time.
    Fields are Optional where the source may be unavailable — agents must
    handle None values and fall back to reduced-signal analysis rather than
    raising exceptions.
    """
    ticker: str
    current_price: float
    volume: int

    # Technical indicators — None when yfinance is unreachable
    rsi: Optional[float] = None             # Relative Strength Index (0–100)
    macd: Optional[float] = None            # MACD histogram value
    moving_avg_50: Optional[float] = None   # 50-day SMA
    moving_avg_200: Optional[float] = None  # 200-day SMA (golden/death cross ref)

    # Intraday indicators — None when yfinance intraday data is unavailable
    vwap: Optional[float] = None                # Today's VWAP (close * volume weighted)
    price_above_vwap: Optional[bool] = None     # True if current price > VWAP
    atr_pct: Optional[float] = None             # 14-day ATR as % of current price

    # Opening range breakout (9:30–10:00 AM EST)
    opening_range_high: Optional[float] = None  # High of first 30-min candle range
    opening_range_low: Optional[float] = None   # Low of first 30-min candle range
    orb_breakout_up: Optional[bool] = None      # True if price broke above opening range high
    orb_breakout_down: Optional[bool] = None    # True if price broke below opening range low

    # Pre-market gap (via fast_info)
    gap_pct: Optional[float] = None             # (last_price - prev_close) / prev_close * 100
    gap_is_bullish: Optional[bool] = None       # True if gap_pct > 0.5%
    gap_is_bearish: Optional[bool] = None       # True if gap_pct < -0.5%

    # Volume confirmation vs 20-day average
    volume_ratio: Optional[float] = None        # today_volume / avg_20day_volume
    volume_confirmed: Optional[bool] = None     # True if volume_ratio > 1.20

    # Fundamentals — None when yfinance is unreachable
    pe_ratio: Optional[float] = None              # Trailing P/E ratio
    forward_pe: Optional[float] = None            # Forward P/E ratio
    revenue_growth: Optional[float] = None        # YoY revenue growth (decimal, e.g. 0.12 = 12%)
    eps: Optional[float] = None                   # Trailing earnings per share
    next_earnings_date: Optional[str] = None      # ISO date of next earnings report
    analyst_recommendation: Optional[str] = None  # 'buy', 'hold', 'sell', etc.

    # Sentiment — None when finnhub is unreachable
    news_sentiment: Optional[float] = None  # Normalised score: -1.0 (bearish) to 1.0 (bullish)
    news_headlines: List[str] = []          # Raw headlines passed to the LLM for context

    # Macro — None when FRED is unreachable
    macro_context: Optional[str] = None    # Free-text summary of relevant macro conditions

    # Tracks which sources contributed to this snapshot
    data_sources_used: DataSourceStatus = DataSourceStatus()

    # ISO-8601 timestamp auto-set at collection time
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Agent Analysis ────────────────────────────────────────────────────────────

class AgentAnalysis(BaseModel):
    """
    Output produced by an individual CrewAI analyst agent (bull, bear, or macro).
    Strict field constraints enforce that agents return substantive, structured
    responses rather than terse or empty outputs.
    """
    ticker: str
    recommendation: TradeType

    # Confidence must be in [0, 1]; the risk manager gates execution at 0.75
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Minimum 50 chars ensures the agent produces a meaningful justification,
    # not a placeholder like "looks good"
    reasoning: str = Field(..., min_length=50)

    # At least 2 distinct factors required to prevent single-signal overconfidence
    key_factors: List[str] = Field(..., min_items=2)

    # The agent's recommended hold tier (intraday / swing / position)
    recommended_hold_period: HoldPeriod
    hold_period_reasoning: str  # Why this duration was chosen given current conditions


# ── Trade Decision ────────────────────────────────────────────────────────────

class TradeDecision(BaseModel):
    """
    Final synthesized decision produced by the risk manager agent after
    reviewing all analyst outputs. This object is passed directly to
    trade_executor.py for order placement.

    All price levels (entry, stop, take-profit) are calculated by
    position_sizer.py based on the hold_period and current price before
    this model is populated.
    """
    ticker: str
    execute: bool  # Master switch — False means the cycle produced no actionable signal

    trade_type: Optional[TradeType] = None           # None when execute=False
    order_type: OrderType = OrderType.LIMIT           # Limit orders by default to control slippage
    hold_period: HoldPeriod = HoldPeriod.SWING        # Default to swing if unspecified
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Position sizing — populated by position_sizer.py, None until calculated
    position_size_usd: Optional[float] = None

    # Price levels — populated by position_sizer.py based on hold period params in config
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    # Maximum calendar days before position_monitor.py forces an exit regardless of P&L
    max_hold_days: int = 5

    # Reasoning strings from each agent — preserved for audit trail and dashboard display
    bull_reasoning: str = ''
    bear_reasoning: str = ''
    risk_manager_reasoning: str = ''
    hold_period_reasoning: str = ''

    # Snapshot of source availability at decision time — stored with the trade record
    data_sources_available: DataSourceStatus = DataSourceStatus()

    @field_validator('confidence')
    @classmethod
    def confidence_required_if_executing(cls, v, info):
        """
        Enforce the minimum confidence threshold before any order can be placed.
        This mirrors config.confidence_threshold (0.75) as a hard schema-level
        guard so execution cannot be triggered even if the caller bypasses config.
        """
        if info.data.get('execute') and v < 0.75:
            raise ValueError('Cannot execute trade with confidence below 0.75')
        return v
