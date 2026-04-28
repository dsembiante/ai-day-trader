"""
crew.py — Orchestrates the full per-ticker analysis and trade execution cycle.

This module is the core runtime loop. For each ticker in the watchlist it:
    1. Collects market data (with graceful degradation via data_collector.py)
    2. Spins up a 4-agent CrewAI crew (bull → bear → risk → portfolio)
    3. Parses the final TradeDecision from the crew output
    4. Runs position sizing to populate price levels and dollar amounts
    5. Submits the order to Alpaca via trade_executor.py
    6. Persists the trade record to SQLite and the flat JSON journal

Module-level singletons (collector, sizer, executor, db) are instantiated
once and shared across all tickers and all scheduler cycles within the same
process, avoiding redundant client initialisation and database connections.

Entry point:
    run_trading_cycle(circuit_breaker) — called by scheduler.py on each cycle.
"""

from crewai import Crew, Process
from agents import (
    create_bull_agent, create_bear_agent,
    create_risk_manager, create_portfolio_manager,
    create_gap_fade_analyst,
    create_vwap_reversion_analyst,
)
from tasks import (
    create_bull_task, create_bear_task,
    create_risk_manager_task, create_portfolio_task,
    create_exit_bull_task, create_exit_bear_task,
    create_gap_fade_task,
    create_vwap_reversion_task,
)
from models import TradeDecision
from data_collector import DataCollector
from position_sizer import PositionSizer
from position_monitor import PositionMonitor
from trade_executor import TradeExecutor
from circuit_breaker import CircuitBreaker
from database import Database
from logger import log_error, log_trade, new_run_log, log_run
from config import config, HoldPeriod
from datetime import datetime, time
from zoneinfo import ZoneInfo
from macro_calendar import check_high_impact_day
import json
import uuid
import yfinance as yf


# ── Session Momentum Helpers ─────────────────────────────────────────────────

def _get_session_phase(et_now: datetime) -> str:
    t = et_now.time()
    if t < time(11, 0):
        return 'morning'
    elif t < time(13, 0):
        return 'midday'
    else:
        return 'afternoon'


def _get_vwap_margin_pct(price: float, vwap: float) -> float:
    """Return how far price is above/below VWAP as a percentage."""
    return (price - vwap) / vwap * 100


def _get_price_vs_orb_high(price: float, orb_high: float) -> float:
    """Return current price distance from ORB high as a percentage."""
    return (price - orb_high) / orb_high * 100


def _get_2bar_momentum(ticker: str) -> float | None:
    """
    Return the 2-bar price change % using the last 3 one-minute closes.
    (most_recent_close - close_2_bars_ago) / close_2_bars_ago * 100
    Returns None on any fetch or data error — caller must treat None as pass-through.
    """
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import timedelta
        bars = collector.alpaca.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(minutes=15),
        ))
        df = bars.df.reset_index()
        if df is None or len(df) < 3:
            return None
        close_now = float(df['close'].iloc[-1])
        close_2ago = float(df['close'].iloc[-3])
        if close_2ago == 0:
            return None
        return (close_now - close_2ago) / close_2ago * 100
    except Exception:
        return None


# ── Module-Level Singletons ───────────────────────────────────────────────────
# Instantiated once at import time and reused for every ticker across all
# scheduler cycles within the process lifetime. This avoids opening new
# database connections and API clients on every call to run_trading_cycle().
collector = DataCollector()
sizer     = PositionSizer()
executor  = TradeExecutor()
db        = Database()
cb        = CircuitBreaker()  # Used by run_single_ticker for news-triggered trades


# ── Lightweight Position Monitor ─────────────────────────────────────────────

def run_position_monitor_only():
    """
    Lightweight exit check — runs position monitoring only, no entry evaluation.

    Called by scheduler.py every 5 minutes throughout the trading day.
    No Groq/LLM calls are made. Covers:
        - Bracket exit reconciliation
        - Stop-loss / take-profit / time-based exits
        - Dynamic profit threshold exits
        - Market reversal coverage

    The full run_trading_cycle() handles entries on its own schedule; this
    function exists solely to catch exits faster between full cycles.
    """
    et_now = datetime.now(ZoneInfo('America/New_York'))
    if et_now.weekday() >= 5 or not (time(9, 30) <= et_now.time() <= time(15, 50)):
        return

    try:
        open_positions = executor.get_open_positions()
        if not open_positions:
            return  # Nothing to monitor

        print(f'[monitor_check] {et_now.strftime("%H:%M ET")} — checking {len(open_positions)} open positions...')
        monitor = PositionMonitor(executor)
        monitor.reconcile_bracket_exits()
        monitor.reconcile_manual_closes()
        monitor.check_all_positions()
        monitor.check_dynamic_exits()

        reversal = monitor.check_market_reversal()
        if reversal in ('cover_longs', 'cover_shorts'):
            target_types = ('buy', 'long') if reversal == 'cover_longs' else ('short',)
            for trade in db.get_open_trades():
                if trade.get('trade_type') in target_types:
                    try:
                        executor.close_position(trade['ticker'], trade['trade_type'])
                        import time as _time; _time.sleep(2)
                        exit_price = executor.get_filled_exit_price(trade['ticker'])
                        db.update_trade_status(
                            trade['trade_id'],
                            status='closed',
                            exit_reason=f'market_reversal_{reversal}',
                            exit_price=exit_price,
                        )
                    except Exception as e:
                        log_error('monitor_check_reversal', trade['ticker'], str(e))
    except Exception as e:
        print(f'[monitor_check] Error: {e}')


# ── Multi-Strategy Pipeline Functions ────────────────────────────────────────

def run_gap_fade_ticker(
    ticker, market_data, db, executor, config,
    et_now, market_regime, vix_regime,
):
    """Gap fade pipeline for a single ticker. Returns immediately if not qualified."""
    if market_data.gap_pct is None or abs(market_data.gap_pct) < config.gap_fade_min_gap_pct:
        return

    _earnings_date = market_data.next_earnings_date
    if _earnings_date and _earnings_date == et_now.date().isoformat():
        print(f'⏭️  {ticker} — gap fade skipped: earnings today')
        return

    open_positions  = executor.get_open_positions()
    portfolio_value = executor.get_portfolio_value()
    _total_exposure = sum(abs(float(p.get('market_value', 0))) for p in open_positions)
    if portfolio_value and _total_exposure / portfolio_value >= 0.80:
        print(f'⏭️  {ticker} — gap fade skipped: exposure cap reached')
        return

    agent  = create_gap_fade_analyst()
    task   = create_gap_fade_task(agent, ticker, market_data)
    _crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = _crew.kickoff()

    if hasattr(result, 'json_dict') and result.json_dict:
        raw_dict = result.json_dict
    else:
        raw = result.raw if hasattr(result, 'raw') else str(result)
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[-1]
        if raw.endswith('```'):
            raw = raw.rsplit('```', 1)[0]
        raw_dict = json.loads(raw.strip())

    execute    = raw_dict.get('execute', False)
    confidence = float(raw_dict.get('confidence', 0.0))
    direction  = raw_dict.get('direction')
    reasoning  = raw_dict.get('reasoning', '')
    print(
        f'🤖 {ticker} [GAP FADE]: execute={execute} | '
        f'confidence={confidence:.2f} | direction={direction} | '
        f'{reasoning[:80]}'
    )

    if not execute or confidence < config.confidence_threshold:
        return

    trade_str = 'buy' if direction == 'long' else 'short'
    hold      = HoldPeriod.INTRADAY
    sizing    = sizer.calculate(portfolio_value, market_data.current_price, confidence, hold)

    _regime_mult = 1.0
    if market_regime == 'bull' and trade_str in ('short', 'sell_short'):
        _regime_mult = 0.85
    elif market_regime == 'sideways':
        _regime_mult = 0.85
    elif market_regime == 'bear' and trade_str == 'buy':
        _regime_mult = 0.85
    _vix_mult      = 0.80 if vix_regime == 'HIGH VOLATILITY' else 1.0
    _adjusted_size = round(sizing['position_usd'] * _regime_mult * _vix_mult, 2)

    stop_loss   = sizer.get_stop_loss(
        market_data.current_price, trade_str, hold,
        atr_pct=market_data.atr_pct, ticker=ticker,
    )
    take_profit = (
        raw_dict.get('gap_fade_target')
        or sizer.get_take_profit(
            market_data.current_price, trade_str, hold,
            atr_pct=market_data.atr_pct, ticker=ticker,
        )
    )

    decision = TradeDecision(
        ticker=ticker,
        execute=True,
        trade_type=trade_str,
        order_type='limit',
        hold_period=HoldPeriod.INTRADAY.value,
        confidence=confidence,
        position_size_usd=_adjusted_size,
        entry_price=market_data.current_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        max_hold_days=1,
        risk_manager_reasoning=reasoning,
        hold_period_reasoning='intraday gap fade',
        data_sources_available=market_data.data_sources_used,
    )

    order_result  = executor.execute_trade(decision)
    _order_status = order_result.get('status', 'unknown')
    _order_id     = order_result.get('order_id', '')
    print(f'📋 {ticker} [GAP FADE] order: {_order_status}{f" | id={_order_id[:8]}" if _order_id else ""}')

    if order_result.get('status') == 'placed':
        actual_entry_price = market_data.current_price
        try:
            import time as _time; _time.sleep(2)
            for _pos in executor.get_open_positions():
                if _pos['ticker'] == ticker and _pos.get('avg_entry_price'):
                    actual_entry_price = _pos['avg_entry_price']
                    break
        except Exception:
            pass
        trade_record = {
            'trade_id':               str(uuid.uuid4()),
            'ticker':                 ticker,
            'trade_type':             decision.trade_type,
            'order_type':             decision.order_type,
            'hold_period':            decision.hold_period,
            'max_hold_days':          1,
            'entry_price':            actual_entry_price,
            'exit_price':             None,
            'shares':                 sizing['shares'],
            'position_size_usd':      _adjusted_size,
            'stop_loss_price':        stop_loss,
            'take_profit_price':      take_profit,
            'pnl':                    None,
            'pnl_pct':                None,
            'status':                 'open',
            'exit_reason':            None,
            'confidence_at_entry':    confidence,
            'bull_reasoning':         '',
            'bear_reasoning':         '',
            'risk_manager_reasoning': reasoning,
            'hold_period_reasoning':  'intraday gap fade',
            'data_sources_available': str(market_data.data_sources_used.model_dump()),
            'atr_pct':                market_data.atr_pct,
            'entry_time':             datetime.now().isoformat(),
            'exit_time':              None,
        }
        try:
            db.insert_trade(trade_record)
            print(f'✅ Gap fade trade record saved to DB: {ticker}')
        except Exception as e:
            print(f'❌ DB insert failed for {ticker}: {e}')
            log_error('database_insert', ticker, str(e))
        log_trade(trade_record)
        return True


def run_vwap_reversion_ticker(
    ticker, market_data, db, executor, config,
    et_now, market_regime, vix_regime,
):
    """VWAP reversion pipeline for a single ticker. Returns immediately if not qualified."""
    if not market_data.vwap or not market_data.current_price:
        return
    vwap_margin_pct = _get_vwap_margin_pct(market_data.current_price, market_data.vwap)
    if abs(vwap_margin_pct) < 1.5:
        return

    _earnings_date = market_data.next_earnings_date
    if _earnings_date and _earnings_date == et_now.date().isoformat():
        print(f'⏭️  {ticker} — VWAP reversion skipped: earnings today')
        return

    open_positions  = executor.get_open_positions()
    portfolio_value = executor.get_portfolio_value()
    _total_exposure = sum(abs(float(p.get('market_value', 0))) for p in open_positions)
    if portfolio_value and _total_exposure / portfolio_value >= 0.80:
        print(f'⏭️  {ticker} — VWAP reversion skipped: exposure cap reached')
        return

    agent  = create_vwap_reversion_analyst()
    task   = create_vwap_reversion_task(agent, ticker, market_data)
    _crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = _crew.kickoff()

    if hasattr(result, 'json_dict') and result.json_dict:
        raw_dict = result.json_dict
    else:
        raw = result.raw if hasattr(result, 'raw') else str(result)
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[-1]
        if raw.endswith('```'):
            raw = raw.rsplit('```', 1)[0]
        raw_dict = json.loads(raw.strip())

    execute    = raw_dict.get('execute', False)
    confidence = float(raw_dict.get('confidence', 0.0))
    direction  = raw_dict.get('direction')
    reasoning  = raw_dict.get('reasoning', '')
    print(
        f'🤖 {ticker} [VWAP REVERSION]: execute={execute} | '
        f'confidence={confidence:.2f} | direction={direction} | '
        f'{reasoning[:80]}'
    )

    if not execute or confidence < config.confidence_threshold:
        return

    trade_str = 'buy' if direction == 'long' else 'short'
    hold      = HoldPeriod.INTRADAY
    sizing    = sizer.calculate(portfolio_value, market_data.current_price, confidence, hold)

    _regime_mult = 1.0
    if market_regime == 'bull' and trade_str in ('short', 'sell_short'):
        _regime_mult = 0.85
    elif market_regime == 'sideways':
        _regime_mult = 0.85
    elif market_regime == 'bear' and trade_str == 'buy':
        _regime_mult = 0.85
    _vix_mult      = 0.80 if vix_regime == 'HIGH VOLATILITY' else 1.0
    _adjusted_size = round(sizing['position_usd'] * _regime_mult * _vix_mult, 2)

    stop_loss   = sizer.get_stop_loss(
        market_data.current_price, trade_str, hold,
        atr_pct=market_data.atr_pct, ticker=ticker,
    )
    take_profit = (
        raw_dict.get('vwap_target')
        or sizer.get_take_profit(
            market_data.current_price, trade_str, hold,
            atr_pct=market_data.atr_pct, ticker=ticker,
        )
    )

    decision = TradeDecision(
        ticker=ticker,
        execute=True,
        trade_type=trade_str,
        order_type='limit',
        hold_period=HoldPeriod.INTRADAY.value,
        confidence=confidence,
        position_size_usd=_adjusted_size,
        entry_price=market_data.current_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        max_hold_days=1,
        risk_manager_reasoning=reasoning,
        hold_period_reasoning='intraday vwap reversion',
        data_sources_available=market_data.data_sources_used,
    )

    order_result  = executor.execute_trade(decision)
    _order_status = order_result.get('status', 'unknown')
    _order_id     = order_result.get('order_id', '')
    print(f'📋 {ticker} [VWAP REVERSION] order: {_order_status}{f" | id={_order_id[:8]}" if _order_id else ""}')

    if order_result.get('status') == 'placed':
        actual_entry_price = market_data.current_price
        try:
            import time as _time; _time.sleep(2)
            for _pos in executor.get_open_positions():
                if _pos['ticker'] == ticker and _pos.get('avg_entry_price'):
                    actual_entry_price = _pos['avg_entry_price']
                    break
        except Exception:
            pass
        trade_record = {
            'trade_id':               str(uuid.uuid4()),
            'ticker':                 ticker,
            'trade_type':             decision.trade_type,
            'order_type':             decision.order_type,
            'hold_period':            decision.hold_period,
            'max_hold_days':          1,
            'entry_price':            actual_entry_price,
            'exit_price':             None,
            'shares':                 sizing['shares'],
            'position_size_usd':      _adjusted_size,
            'stop_loss_price':        stop_loss,
            'take_profit_price':      take_profit,
            'pnl':                    None,
            'pnl_pct':                None,
            'status':                 'open',
            'exit_reason':            None,
            'confidence_at_entry':    confidence,
            'bull_reasoning':         '',
            'bear_reasoning':         '',
            'risk_manager_reasoning': reasoning,
            'hold_period_reasoning':  'intraday vwap reversion',
            'data_sources_available': str(market_data.data_sources_used.model_dump()),
            'atr_pct':                market_data.atr_pct,
            'entry_time':             datetime.now().isoformat(),
            'exit_time':              None,
        }
        try:
            db.insert_trade(trade_record)
            print(f'✅ VWAP reversion trade record saved to DB: {ticker}')
        except Exception as e:
            print(f'❌ DB insert failed for {ticker}: {e}')
            log_error('database_insert', ticker, str(e))
        log_trade(trade_record)


# ── Main Cycle ────────────────────────────────────────────────────────────────

def run_trading_cycle(circuit_breaker: CircuitBreaker):
    """
    Execute one full analysis and trading cycle across the entire watchlist.

    Called by scheduler.py at each scheduled interval. Steps:
        1. Position monitor — close any positions that have exceeded their hold period
        2. Market reversal check — cover positions on wrong side of SPY move
        3. Circuit breaker check — blocks new entries only; protection always runs first
        4. Per-ticker crew run — collect data → analyse → decide → size → execute
        5. Persist run summary to logs

    Args:
        circuit_breaker: Shared CircuitBreaker instance from scheduler.py.
                         Passed in (rather than instantiated here) so the peak
                         value high-water mark persists across cycles.
    """
    # ── Market Hours Gate ─────────────────────────────────────────────────────
    # Reject cycles outside regular trading hours (9:30 AM – 3:45 PM ET,
    # weekdays only) to prevent pre-market or after-hours order submission.
    et_now = datetime.now(ZoneInfo('America/New_York'))
    market_open       = time(9, 30)
    orb_cutoff        = time(9, 45)   # ORB formation window — no new entries before 9:45 AM ET
    market_close      = time(15, 45)
    if et_now.weekday() >= 5 or not (market_open <= et_now.time() <= market_close):
        print(f'⏰ Outside market hours ({et_now.strftime("%a %H:%M ET")}) — skipping trading cycle')
        return

    run_log = new_run_log(config.watchlist)
    start_time = datetime.now()

    # ── Portfolio Value ───────────────────────────────────────────────────────
    # Fetched first — required for both the circuit breaker check and position
    # sizing later in the cycle.
    portfolio_value = executor.get_portfolio_value()

    # ── Gate 1: Position Monitoring ───────────────────────────────────────────
    # Protective exits always run regardless of circuit breaker state — these
    # close existing risk, not open new positions.
    monitor = PositionMonitor(executor)
    print(f'[position_monitor] Running checks on {len(executor.get_open_positions())} open positions...')
    monitor.reconcile_bracket_exits()
    monitor.reconcile_manual_closes()
    monitor.check_all_positions()
    monitor.check_dynamic_exits()

    # ── Market Reversal Check ─────────────────────────────────────────────────
    # If SPY has moved > 2% from today's open, immediately close all positions
    # on the wrong side before running new analysis.
    reversal = monitor.check_market_reversal()
    if reversal in ('cover_longs', 'cover_shorts'):
        target_types = ('buy', 'long') if reversal == 'cover_longs' else ('short',)
        for trade in db.get_open_trades():
            if trade.get('trade_type') in target_types:
                try:
                    executor.close_position(trade['ticker'], trade['trade_type'])
                    import time as _time; _time.sleep(2)
                    exit_price = executor.get_filled_exit_price(trade['ticker'])
                    db.update_trade_status(
                        trade['trade_id'],
                        status='closed',
                        exit_reason=f'market_reversal_{reversal}',
                        exit_price=exit_price,
                    )
                except Exception as e:
                    log_error('market_reversal_close', trade['ticker'], str(e))

    # ── Gate 2: Circuit Breaker ───────────────────────────────────────────────
    # Only blocks new trade entries — position monitoring above always runs first.
    if not circuit_breaker.check(portfolio_value):
        print('🚨 Circuit breaker active — new entries blocked, position monitoring completed')
        run_log.circuit_breaker_triggered = True
        log_run(run_log)
        return

    # ── Gate 3: Time-Window Flags ─────────────────────────────────────────────
    # Each strategy has its own active window. The per-ticker loop runs as long
    # as at least one strategy has an open window; the momentum pipeline gets a
    # continue gate inside the loop when its own window has closed.
    momentum_entries_open = et_now.time() < time(11, 30)
    gap_fade_entries_open = et_now.time() < time(10, 45)
    vwap_reversion_open   = (
        config.vwap_reversion_enabled and
        time(12, 0) <= et_now.time() <= time(14, 30)
    )

    if not momentum_entries_open and not vwap_reversion_open:
        if et_now.time() < time(12, 0):
            print(f'⛔ Past 11:30 AM ET — entries closed, monitoring positions only')
        log_run(run_log)
        return

    # Snapshot of open positions after any expired ones have been closed.
    # Passed to the portfolio task to enforce max_positions and duplicate checks.
    open_positions = executor.get_open_positions()
    alpaca_held_tickers = {p['ticker'] for p in open_positions}
    db_open_tickers     = {t['ticker'] for t in db.get_open_trades()}

    # Item 2: Header for per-position detail block
    _held_for_log = alpaca_held_tickers | db_open_tickers
    if _held_for_log:
        print(f'[position_monitor] Checking {len(_held_for_log)} positions:')

    trades_executed           = 0
    _cycle_analyzed           = 0   # Item 4: tickers that completed crew analysis
    _cycle_errors             = 0   # Item 4: tickers that threw an exception
    _cycle_high_vol_gated     = 0   # Item 4: tickers with ATR >= 4%
    _cycle_strategies_skipped = 0   # Item 4: sum of per-strategy skips from Item 3

    # ── Market Regime Detection ───────────────────────────────────────────────
    # Detected once per cycle using SPY golden/death cross — shared across all
    # tickers so agents operate with consistent macro context. Position sizing
    # is scaled down in bear/sideways markets to reduce risk exposure.
    market_regime = collector.get_market_regime()

    # ── Intraday Crash Override ───────────────────────────────────────────────
    # If SPY is down more than 2% intraday, force bear regime regardless of the
    # golden/death cross classification — prevents longs during market crashes.
    try:
        spy_info = yf.Ticker('SPY').fast_info
        if spy_info.last_price and spy_info.previous_close and spy_info.previous_close > 0:
            spy_intraday_chg = (spy_info.last_price - spy_info.previous_close) / spy_info.previous_close
            if spy_intraday_chg <= -0.02:
                print(f'🚨 SPY down {spy_intraday_chg*100:.2f}% intraday — overriding regime to BEAR')
                market_regime = 'bear'
    except Exception as e:
        log_error('spy_intraday_check', 'SPY', str(e))

    print(f'📈 Market regime: {market_regime.upper()}')
    print(f'📊 Open positions: {len(open_positions)} ({sum(abs(p.get("market_value", 0)) for p in open_positions) / portfolio_value * 100:.1f}% of portfolio deployed)')

    if market_regime == 'bear':
        print('🐻 Bear market detected — favoring shorts, position sizing via multipliers')
    elif market_regime == 'sideways':
        print('➡️  Sideways market — being selective, position sizing via multipliers')

    # ── VIX-Based Confidence Threshold ───────────────────────────────────────
    # Fetched once per cycle (cached daily by get_vix). VIX failure never aborts
    # the cycle — defaults to NORMAL regime and the standard 0.82 threshold.
    vix_level = collector.get_vix()
    if vix_level is not None:
        if vix_level > 25:
            vix_regime = 'HIGH VOLATILITY'
            config.confidence_threshold = 0.80
        elif vix_level < 15:
            vix_regime = 'LOW VOLATILITY'
            config.confidence_threshold = 0.87
        else:
            vix_regime = 'NORMAL'
            config.confidence_threshold = 0.82
        print(f'📉 VIX: {vix_level:.1f} ({vix_regime}) — confidence threshold: {config.confidence_threshold}')
    else:
        vix_regime = 'NORMAL'
        config.confidence_threshold = 0.82
        print('⚠️  VIX unavailable — defaulting to NORMAL regime, confidence threshold 0.82')

    # ── Economic Calendar Check ───────────────────────────────────────────────
    # Checked once per day (result cached to data/cache/macro_events_YYYYMMDD.json).
    # On CPI, NFP, GDP, PPI, or FOMC days: raise confidence threshold to at least
    # 0.87 and block all new entries until 10:30 AM ET — the additional 45 minutes
    # beyond the standard 15-minute ORB window reduces exposure to gap-and-reverse
    # patterns that are most common immediately after high-impact releases.
    is_high_impact, macro_event = check_high_impact_day()
    high_impact_cutoff = time(10, 30)
    if is_high_impact:
        config.confidence_threshold = max(config.confidence_threshold, 0.87)
        print(f'⚠️  HIGH IMPACT MACRO DAY: {macro_event} — confidence threshold raised to {config.confidence_threshold}')

    # Snapshot the cycle-level threshold so per-ticker ATR adjustments can
    # temporarily raise it for high-volatility tickers and restore it cleanly
    # before the next ticker is evaluated.
    _cycle_confidence_threshold = config.confidence_threshold

    # ── Agent Instantiation ───────────────────────────────────────────────────
    # Agents are created once per cycle (not per ticker) and reused.
    # Each agent holds the same shared LLM client from agents.py, so creating
    # them once avoids redundant LLM client setup across the watchlist.
    bull_agent      = create_bull_agent()
    bear_agent      = create_bear_agent()
    risk_agent      = create_risk_manager()
    portfolio_agent = create_portfolio_manager()

    # Build a lookup of open DB trades by ticker for exit evaluation
    db_open_trades_by_ticker = {t['ticker']: t for t in db.get_open_trades()}

    def _parse_task_output(task_obj):
        """Item 6: Safely parse a CrewAI task's output to a dict. Returns None on any failure."""
        try:
            out = getattr(task_obj, 'output', None)
            if out is None:
                return None
            if hasattr(out, 'json_dict') and out.json_dict:
                return out.json_dict
            raw = getattr(out, 'raw', None)
            if raw:
                raw = raw.strip()
                if raw.startswith('```'):
                    raw = raw.split('\n', 1)[-1]
                if raw.endswith('```'):
                    raw = raw.rsplit('```', 1)[0]
                return json.loads(raw.strip())
        except Exception:
            pass
        return None

    # ── Per-Ticker Loop ───────────────────────────────────────────────────────
    for ticker in config.watchlist:
        try:
            # ── Exit Re-evaluation for Held Positions ─────────────────────────
            # If we hold this ticker, run a 1-agent exit evaluation before
            # deciding whether to skip it for new entry analysis.
            if ticker in alpaca_held_tickers or ticker in db_open_tickers:
                db_trade = db_open_trades_by_ticker.get(ticker)
                trade_type = db_trade.get('trade_type', 'buy') if db_trade else 'buy'
                entry_price = db_trade.get('entry_price') if db_trade else None
                is_long = trade_type in ('buy', 'long')

                # Recovery: if entry_price is NULL or $0.00, recover from Alpaca in order:
                # 1. avg_entry_price from the live position (most reliable)
                # 2. Order history fill price (unreliable if limit entry still pending)
                # 3. current market price as last-resort so exit logic still fires
                if db_trade and not entry_price:
                    alpaca_pos_for_recovery = alpaca_positions.get(ticker, {})
                    recovered = (
                        alpaca_pos_for_recovery.get('avg_entry_price')
                        or executor.get_filled_entry_price(ticker, trade_type)
                    )
                    if recovered:
                        entry_price = recovered
                        db.update_entry_price(db_trade['trade_id'], recovered)
                        print(f'🔧 {ticker} — recovered entry price ${recovered:.2f} from Alpaca avg_entry_price / fill history')
                    else:
                        print(f'⚠️  {ticker} — entry_price NULL and recovery failed — exit signals may be impaired')

                print(f'\n🔄 Re-evaluating open position: {ticker} ({trade_type})')

                try:
                    market_data = collector.collect(ticker)
                    if not market_data.data_sources_used.alpaca:
                        print(f'⚠️  No price data for {ticker} — skipping exit evaluation')
                        continue

                    exit_summary = f'''
                        Ticker: {ticker}
                        Price: ${market_data.current_price:.2f}
                        Volume: {market_data.volume:,}

                        VWAP Analysis:
                        VWAP: {f'${market_data.vwap:.2f}' if market_data.vwap else 'N/A'}
                        Price above VWAP: {market_data.price_above_vwap if market_data.price_above_vwap is not None else 'N/A'}

                        Opening Range Breakout:
                        ORB breakout up: {market_data.orb_breakout_up if market_data.orb_breakout_up is not None else 'N/A'}
                        ORB breakdown: {market_data.orb_breakout_down if market_data.orb_breakout_down is not None else 'N/A'}

                        Gap Analysis:
                        Pre-market gap: {f'{market_data.gap_pct:.2f}%' if market_data.gap_pct is not None else 'N/A'}
                        Bullish gap: {market_data.gap_is_bullish if market_data.gap_is_bullish is not None else 'N/A'}
                        Bearish gap: {market_data.gap_is_bearish if market_data.gap_is_bearish is not None else 'N/A'}

                        Volume:
                        Volume ratio vs 20-day avg: {f'{market_data.volume_ratio:.2f}x' if market_data.volume_ratio else 'N/A'}
                        Volume confirmed: {market_data.volume_confirmed if market_data.volume_confirmed is not None else 'N/A'}

                        RSI: {f'{market_data.rsi:.1f}' if market_data.rsi else 'N/A'}
                        MACD: {f'{market_data.macd:.4f}' if market_data.macd else 'N/A'}
                        Market Regime: {market_regime.upper()}
                        VIX: {f'{market_data.vix:.1f}' if market_data.vix is not None else 'N/A'}
                    '''

                    exit_task = (
                        create_exit_bull_task(bull_agent, ticker, exit_summary, entry_price)
                        if is_long else
                        create_exit_bear_task(bear_agent, ticker, exit_summary, entry_price)
                    )
                    exit_agent = bull_agent if is_long else bear_agent

                    exit_crew = Crew(
                        agents=[exit_agent],
                        tasks=[exit_task],
                        process=Process.sequential,
                        verbose=False,
                    )
                    exit_result = exit_crew.kickoff()

                    # Parse exit decision
                    if hasattr(exit_result, 'json_dict') and exit_result.json_dict:
                        exit_data = exit_result.json_dict
                    else:
                        raw = exit_result.raw if hasattr(exit_result, 'raw') else str(exit_result)
                        raw = raw.strip()
                        if raw.startswith('```'):
                            raw = raw.split('\n', 1)[-1]
                        if raw.endswith('```'):
                            raw = raw.rsplit('```', 1)[0]
                        exit_data = json.loads(raw.strip())

                    should_exit = exit_data.get('exit', False)
                    exit_confidence = float(exit_data.get('confidence', 0.0))
                    exit_reasoning = exit_data.get('reasoning', '')

                    # Item 2: Per-position detail log
                    try:
                        _pos_entry  = entry_price or market_data.current_price
                        _pos_shares = (db_trade.get('shares') or 0) if db_trade else 0
                        _pos_cur    = market_data.current_price
                        _pos_side   = 'long' if is_long else 'short'
                        if _pos_shares > 0 and _pos_entry:
                            _pos_pnl_usd = (
                                (_pos_cur - _pos_entry) * _pos_shares if is_long
                                else (_pos_entry - _pos_cur) * _pos_shares
                            )
                            _pos_pnl_pct = _pos_pnl_usd / (_pos_entry * _pos_shares) * 100
                            _pos_pnl_str = f'${_pos_pnl_usd:+.2f} ({_pos_pnl_pct:+.2f}%)'
                        else:
                            _pos_pnl_str = 'N/A'
                        _pos_mfe_raw = db_trade.get('max_favorable_excursion_pct') if db_trade else None
                        _pos_mfe_str = f'{_pos_mfe_raw * 100:+.2f}%' if _pos_mfe_raw is not None else 'N/A'
                        _pos_entry_dt = None
                        if db_trade and db_trade.get('entry_time'):
                            try:
                                _pos_entry_dt = datetime.fromisoformat(db_trade['entry_time'])
                            except Exception:
                                pass
                        _pos_mins    = int((datetime.now() - _pos_entry_dt).total_seconds() / 60) if _pos_entry_dt else 0
                        _exit_dec    = 'EXIT' if (should_exit and exit_confidence >= 0.75) else 'HOLD'
                        _exit_rsn    = exit_reasoning.replace('\n', ' ')[:60].strip()
                        print(f'  {ticker} {_pos_side} @ ${_pos_entry:.2f} | now ${_pos_cur:.2f} | P&L {_pos_pnl_str} | MFE {_pos_mfe_str} | {_pos_mins}min')
                        print(f'    Exit eval: confidence={exit_confidence:.2f} | decision={_exit_dec} | reason={_exit_rsn}')
                    except Exception:
                        pass  # Position detail is non-critical observability

                    if should_exit and exit_confidence >= 0.75:
                        print(f'✅ Agent recommends EXIT {ticker} — {exit_reasoning}')
                        executor.close_position(ticker, trade_type)
                        import time as _time; _time.sleep(2)
                        exit_price_filled = executor.get_filled_exit_price(ticker)
                        if db_trade:
                            db.update_trade_status(
                                db_trade['trade_id'],
                                status='closed',
                                exit_reason='agent_exit_recommendation',
                                exit_price=exit_price_filled,
                            )
                    else:
                        print(f'⏸️  Agent recommends HOLD {ticker} — {exit_reasoning}')

                except Exception as e:
                    log_error('exit_evaluation', ticker, str(e))
                    print(f'❌ Exit evaluation error for {ticker}: {e}')

                # Always skip new entry analysis for held tickers regardless of exit outcome
                continue

            # ── Loss Cooloff Gate ─────────────────────────────────────────────
            # If the most recent closed trade for this ticker today was a loss,
            # skip re-entry until loss_cooloff_minutes have elapsed. Prevents
            # walking back into the same bearish conditions that just stopped us out.
            last_trade = db.get_last_closed_trade(ticker)
            if last_trade and last_trade.get('pnl') is not None and last_trade['pnl'] < 0:
                try:
                    exit_dt = datetime.fromisoformat(last_trade['exit_time'])
                    minutes_since_exit = (datetime.now() - exit_dt).total_seconds() / 60
                    if minutes_since_exit < config.loss_cooloff_minutes:
                        print(
                            f'⏸️ {ticker} — 15min cooloff after loss exit '
                            f'({minutes_since_exit:.0f}min ago) — skipping'
                        )
                        continue
                except Exception:
                    pass  # Malformed exit_time — allow evaluation to proceed

            print(f'\n📊 Analyzing {ticker}...')

            # ── Data Collection ───────────────────────────────────────────────
            # collect() returns partial data on source failure — DataSourceStatus
            # tracks which sources were reachable so agents can adjust confidence.
            market_data = collector.collect(ticker)

            # Without a price from Alpaca we cannot size a position — skip entirely
            if not market_data.data_sources_used.alpaca:
                print(f'⚠️  Skipping {ticker} — no price data available')
                continue

            # ── VWAP Re-entry Gate ────────────────────────────────────────────
            # On a re-entry (ticker already has at least one closed trade today),
            # require price to be clearly on the correct side of VWAP before
            # running the full agent cycle. Blocks entries on exhausted momentum
            # where price is drifting near or through VWAP after a prior exit.
            # First entries of the day (last_trade is None) bypass this gate.
            if (
                last_trade is not None
                and market_data.vwap
                and market_data.current_price
            ):
                vwap_margin = _get_vwap_margin_pct(market_data.current_price, market_data.vwap)
                if vwap_margin >= 0 and vwap_margin < 0.30:
                    print(
                        f'⏭️ {ticker} — re-entry requires price clearly above VWAP '
                        f'(current: {vwap_margin:.2f}%) — skipping'
                    )
                    continue
                elif vwap_margin < 0 and vwap_margin > -0.30:
                    print(
                        f'⏭️ {ticker} — short re-entry requires price clearly below VWAP '
                        f'(current: {vwap_margin:.2f}%) — skipping'
                    )
                    continue

            # ── Per-Ticker ATR Volatility Regime ──────────────────────────────
            # Restore cycle-level threshold before evaluating this ticker.
            # The previous ticker may have temporarily raised it for high-vol.
            config.confidence_threshold = _cycle_confidence_threshold

            _ticker_atr = market_data.atr_pct or 0.0
            _is_high_vol = _ticker_atr >= 4.0
            if _is_high_vol:
                config.confidence_threshold = max(config.confidence_threshold, 0.85)
                _cycle_high_vol_gated += 1
                print(
                    f'⚠️ {ticker} ATR {_ticker_atr:.2f}% — high volatility, '
                    f'requiring 3/4 signals and 0.85 confidence'
                )

            # ── Multi-Strategy Dispatchers ────────────────────────────────────
            gap_fade_traded = False
            if config.gap_fade_enabled and gap_fade_entries_open:
                gap_fade_traded = bool(run_gap_fade_ticker(
                    ticker, market_data, db, executor, config,
                    et_now, market_regime, vix_regime,
                ))

            if vwap_reversion_open:
                run_vwap_reversion_ticker(
                    ticker, market_data, db, executor, config,
                    et_now, market_regime, vix_regime,
                )

            # ── Strategy Eligibility Log (Item 3) ────────────────────────────
            _gf_status = (
                'disabled'             if not config.gap_fade_enabled
                else 'skipped(window_closed)' if not gap_fade_entries_open
                else 'traded'          if gap_fade_traded
                else 'evaluated(no_signal)'
            )
            _vwap_status = (
                'disabled'             if not config.vwap_reversion_enabled
                else 'skipped(window_closed)' if not vwap_reversion_open
                else 'evaluated'
            )
            _momentum_status = (
                'skipped(window_closed)'   if not momentum_entries_open
                else 'skipped(gap_fade_traded)' if gap_fade_traded
                else 'eligible'
            )
            print(f'[strategies] {ticker} — gap_fade={_gf_status} | momentum={_momentum_status} | vwap={_vwap_status}')
            _cycle_strategies_skipped += sum(
                1 for _s in (_gf_status, _momentum_status, _vwap_status)
                if _s.startswith('skipped')
            )

            # Momentum — skip if gap fade already traded this ticker
            # or if outside the momentum entry window
            if not momentum_entries_open or gap_fade_traded:
                continue

            # ── Market Data Summary ───────────────────────────────────────────
            # Pre-format all signals into a single string injected into each
            # agent prompt. Inline formatting handles None values gracefully
            # so the LLM never sees Python's 'None' string in the context.
            summary = f'''
                Ticker: {ticker}
                Price: ${market_data.current_price:.2f}
                Volume: {market_data.volume:,}

                VWAP Analysis:
                VWAP: {f'${market_data.vwap:.2f}' if market_data.vwap else 'N/A'}
                Price above VWAP: {market_data.price_above_vwap if market_data.price_above_vwap is not None else 'N/A'}

                Opening Range Breakout:
                Opening range high: {f'${market_data.opening_range_high:.2f}' if market_data.opening_range_high else 'N/A'}
                Opening range low: {f'${market_data.opening_range_low:.2f}' if market_data.opening_range_low else 'N/A'}
                ORB breakout up: {market_data.orb_breakout_up if market_data.orb_breakout_up is not None else 'N/A'}
                ORB breakdown: {market_data.orb_breakout_down if market_data.orb_breakout_down is not None else 'N/A'}

                Gap Analysis:
                Pre-market gap: {f'{market_data.gap_pct:.2f}%' if market_data.gap_pct is not None else 'N/A'}
                Bullish gap: {market_data.gap_is_bullish if market_data.gap_is_bullish is not None else 'N/A'}
                Bearish gap: {market_data.gap_is_bearish if market_data.gap_is_bearish is not None else 'N/A'}

                Volume:
                Volume ratio vs 20-day avg: {f'{market_data.volume_ratio:.2f}x' if market_data.volume_ratio else 'N/A'}
                Volume confirmed: {market_data.volume_confirmed if market_data.volume_confirmed is not None else 'N/A'}

                ATR%: {f'{market_data.atr_pct:.2f}%' if market_data.atr_pct else 'N/A'}
                Volatility Regime: {f'⚠️ HIGH VOLATILITY TICKER (ATR: {market_data.atr_pct:.2f}%) — require 3/4 signals minimum and be conservative. Normal price noise on this ticker can look like valid signals. Confidence must be >= 0.85 to execute.' if _is_high_vol else 'Normal — standard 2/4 signal threshold applies.'}
                RSI: {f'{market_data.rsi:.1f}' if market_data.rsi else 'N/A'}
                MACD: {f'{market_data.macd:.4f}' if market_data.macd else 'N/A'}
                50-day MA: {f'{market_data.moving_avg_50:.2f}' if market_data.moving_avg_50 else 'N/A'}
                200-day MA: {f'{market_data.moving_avg_200:.2f}' if market_data.moving_avg_200 else 'N/A'}
                P/E Ratio: {f'{market_data.pe_ratio:.1f}' if market_data.pe_ratio else 'N/A'}
                Forward P/E: {f'{market_data.forward_pe:.1f}' if market_data.forward_pe else 'N/A'}
                EPS: {f'${market_data.eps:.2f}' if market_data.eps else 'N/A'}
                Revenue Growth: {f'{market_data.revenue_growth*100:.1f}%' if market_data.revenue_growth else 'N/A'}
                Next Earnings: {market_data.next_earnings_date or 'N/A'}
                Analyst Recommendation: {market_data.analyst_recommendation or 'N/A'}
                Market Regime: {market_regime.upper()}
                VIX: {f'{market_data.vix:.1f} ({"HIGH VOLATILITY" if market_data.vix > 25 else "LOW VOLATILITY" if market_data.vix < 15 else "NORMAL"})' if market_data.vix is not None else 'N/A'}
                News headlines: {market_data.news_headlines[:5]}
                Macro context: {market_data.macro_context or 'N/A'}
                Data sources available: {market_data.data_sources_used.model_dump()}

                Session Momentum Filter:
                Session phase: {_get_session_phase(et_now)}
                VWAP margin %: {f'{_get_vwap_margin_pct(market_data.current_price, market_data.vwap):.2f}%' if market_data.vwap and market_data.current_price else 'N/A'}
                Price vs ORB high %: {f'{_get_price_vs_orb_high(market_data.current_price, market_data.opening_range_high):.2f}%' if market_data.opening_range_high and market_data.current_price else 'N/A'}
            '''

            # ── Task Creation ─────────────────────────────────────────────────
            # Tasks are created fresh per ticker because the description prompt
            # embeds the ticker symbol and market data summary.
            bull_task      = create_bull_task(bull_agent, ticker, summary)
            bear_task      = create_bear_task(bear_agent, ticker, summary)
            risk_task      = create_risk_manager_task(risk_agent, ticker, bull_task, bear_task)
            portfolio_task = create_portfolio_task(portfolio_agent, ticker, risk_task, open_positions)

            # ── Crew Execution ────────────────────────────────────────────────
            # Process.sequential runs tasks in order: bull → bear → risk → portfolio.
            # CrewAI passes each task's output into the next via the context= wiring
            # defined in tasks.py. verbose=False suppresses per-step LLM output.
            crew = Crew(
                agents=[bull_agent, bear_agent, risk_agent, portfolio_agent],
                tasks=[bull_task, bear_task, risk_task, portfolio_task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()

            # ── Decision Parsing ──────────────────────────────────────────────
            # CrewAI may return output as a parsed dict (json_dict) or as a raw
            # string. Try the structured path first; fall back to JSON parsing.
            # Strip markdown code fences (```json ... ``` or ``` ... ```) that
            # the LLM sometimes wraps around its JSON response — json.loads
            # cannot handle the backtick markers.
            if hasattr(result, 'json_dict') and result.json_dict:
                raw_dict = result.json_dict
            else:
                raw = result.raw if hasattr(result, 'raw') else str(result)
                # Remove markdown code fences if present
                raw = raw.strip()
                if raw.startswith('```'):
                    raw = raw.split('\n', 1)[-1]  # Drop the opening ```[json] line
                if raw.endswith('```'):
                    raw = raw.rsplit('```', 1)[0]  # Drop the closing ``` line
                raw_dict = json.loads(raw.strip())

            # ── Safety Override: enforce Risk Manager hierarchy ────────────────
            # The Portfolio Manager is forbidden from flipping execute=false to
            # execute=true. Extract the Risk Manager's decision from its task
            # output and override the portfolio result if it tried to flip it.
            # Also normalise any hallucinated trade_type='long' → 'buy'.
            _VALID_TRADE_TYPES = {'buy', 'sell', 'short', 'cover'}
            if isinstance(raw_dict.get('trade_type'), str):
                if raw_dict['trade_type'] not in _VALID_TRADE_TYPES:
                    raw_dict['trade_type'] = None  # Will be caught downstream

            risk_execute = None
            try:
                risk_out = risk_task.output
                if hasattr(risk_out, 'json_dict') and risk_out.json_dict:
                    risk_execute = risk_out.json_dict.get('execute')
                elif hasattr(risk_out, 'raw') and risk_out.raw:
                    _r = risk_out.raw.strip()
                    if _r.startswith('```'):
                        _r = _r.split('\n', 1)[-1]
                    if _r.endswith('```'):
                        _r = _r.rsplit('```', 1)[0]
                    risk_execute = json.loads(_r.strip()).get('execute')
            except Exception:
                pass  # If we can't read the risk task output, leave override logic to prompt

            if risk_execute is False and raw_dict.get('execute') is True:
                print(f'⚠️  Safety override: Portfolio Manager attempted to flip execute=false to execute=true for {ticker} — blocked')
                raw_dict['execute']           = False
                raw_dict['trade_type']        = None
                raw_dict['entry_price']       = None
                raw_dict['stop_loss_price']   = None
                raw_dict['take_profit_price'] = None
                raw_dict['position_size_usd'] = None

            decision = TradeDecision(**raw_dict)

            # ── Item 6: Per-agent verdict chain ───────────────────────────────
            try:
                _bd = _parse_task_output(bull_task)
                if _bd:
                    _kf  = _bd.get('key_factors', [])
                    _kfl = _kf if isinstance(_kf, list) else [str(_kf)]
                    _sig = next((str(k) for k in _kfl if '/4' in str(k)), '?/4 signals')
                    _rsn = str(_bd.get('reasoning', ''))[:80].replace('\n', ' ')
                    print(f'🐂 {ticker} Bull: {_sig} | confidence={float(_bd.get("confidence", 0)):.2f} | reason={_rsn}')

                _brd = _parse_task_output(bear_task)
                if _brd:
                    _kf  = _brd.get('key_factors', [])
                    _kfl = _kf if isinstance(_kf, list) else [str(_kf)]
                    _sig = next((str(k) for k in _kfl if '/4' in str(k)), '?/4 signals')
                    _rsn = str(_brd.get('reasoning', ''))[:80].replace('\n', ' ')
                    print(f'🐻 {ticker} Bear: {_sig} | confidence={float(_brd.get("confidence", 0)):.2f} | reason={_rsn}')

                _rd = _parse_task_output(risk_task)
                if _rd:
                    _rm_verdict = 'APPROVE' if _rd.get('execute') else 'REJECT'
                    _rsn = str(_rd.get('risk_manager_reasoning', ''))[:80].replace('\n', ' ')
                    print(f'🛡️ {ticker} Risk: {_rm_verdict} | confidence={float(_rd.get("confidence", 0)):.2f} | reason={_rsn}')

                _pm_verdict = 'execute=True' if raw_dict.get('execute') else 'execute=False'
                _pm_rsn = str(raw_dict.get('risk_manager_reasoning', ''))[:80].replace('\n', ' ')
                print(f'📋 {ticker} PM: {_pm_verdict} | confidence={float(raw_dict.get("confidence", 0)):.2f} | reason={_pm_rsn}')
            except Exception:
                pass  # Agent chain logging is non-critical — never crash the cycle

            # ── Compact Decision Summary ──────────────────────────────────────
            # Replaces the verbose ╭──────╮ agent output (suppressed via verbose=False).
            # Shows the key fields needed for trade review without dumping full prompts.
            _reasoning = (decision.risk_manager_reasoning or decision.bull_reasoning or '')[:120]
            print(
                f'🤖 {ticker}: execute={decision.execute} | '
                f'confidence={decision.confidence:.2f} | '
                f'type={decision.trade_type or "none"} | '
                f'{_reasoning}'
            )
            _cycle_analyzed += 1  # Item 4: this ticker completed crew analysis

            # ── Decision Post-Processing ──────────────────────────────────────
            # If the agent omitted entry_price (returns null for market orders),
            # fall back to the current market price so the whole-share calculation
            # in trade_executor always has a price to work with. Alpaca rejects
            # bracket orders with fractional qty, so a price is always required.
            if decision.execute and not decision.entry_price:
                decision.entry_price = market_data.current_price

            # ── Position Sizing & Execution ───────────────────────────────────
            if decision.execute and decision.trade_type:
                trade_str = decision.trade_type.value if hasattr(decision.trade_type, 'value') else str(decision.trade_type)

                # ── Earnings Day Gate ─────────────────────────────────────────
                # Block all entries on earnings day regardless of signal strength.
                # next_earnings_date is fetched from yfinance calendar at data
                # collection time; comparison is date-string equality so no
                # timezone conversion is needed.
                _earnings_date = market_data.next_earnings_date
                if _earnings_date and _earnings_date == datetime.now(ZoneInfo('America/New_York')).date().isoformat():
                    print(
                        f'⏭️ {ticker} — earnings today ({_earnings_date}), '
                        f'skipping to avoid earnings volatility'
                    )
                    continue

                # ── Exposure Cap Gate ─────────────────────────────────────────
                # Hard block when total deployed capital >= 80% of portfolio.
                # Uses the snapshot of open_positions taken at cycle start;
                # abs() handles shorts whose market_value is negative in Alpaca.
                _total_exposure = sum(abs(p.get('market_value', 0)) for p in open_positions)
                _exposure_pct   = (_total_exposure / portfolio_value * 100) if portfolio_value else 0.0
                if _exposure_pct >= 80.0:
                    print(
                        f'⏭️ {ticker} — exposure cap reached '
                        f'({_exposure_pct:.1f}% of portfolio deployed, 80% max)'
                    )
                    continue

                # ── Re-entry Cooldown Gate ────────────────────────────────────
                # Block re-entering the same ticker in the same direction within
                # 10 minutes of the previous exit. Prevents immediately walking
                # back into the same conditions that just closed the position.
                _recent_exit = db.get_recent_closed_trade_by_direction(
                    ticker, trade_str, minutes=10
                )
                if _recent_exit:
                    try:
                        _exit_dt  = datetime.fromisoformat(_recent_exit['exit_time'])
                        _mins_ago = (datetime.now() - _exit_dt).total_seconds() / 60
                    except Exception:
                        _mins_ago = 0.0
                    print(
                        f'⏭️ {ticker} — re-entry cooldown active '
                        f'(exited {_mins_ago:.0f}min ago), skipping'
                    )
                    continue

                # Resolve hold period — default to SWING if the agent omitted it
                requested_hold = HoldPeriod(decision.hold_period) if decision.hold_period else HoldPeriod.SWING
                hold = sizer.get_hold_period_safe(requested_hold)
                decision.hold_period = hold.value  # Reflect any PDT upgrade in the trade record

                # Calculate dollar size, share count, stop-loss, and take-profit
                sizing = sizer.calculate(
                    portfolio_value, market_data.current_price, decision.confidence, hold
                )
                # ── High Conviction Override ──────────────────────────────────
                # Confidence >= 0.85 with a favorable regime (long in bull, short in
                # bear) recalculates using a 20% ceiling instead of the standard 15%.
                # Applied before regime/VIX multipliers so reductions still apply on top.
                _is_high_conviction = (
                    decision.confidence >= 0.85
                    and (
                        (trade_str == 'buy' and market_regime == 'bull')
                        or (trade_str in ('short', 'sell_short') and market_regime == 'bear')
                    )
                )
                if _is_high_conviction:
                    print(
                        f'📏 {ticker} — high conviction sizing: '
                        f'confidence {decision.confidence:.2f} + favorable regime '
                        f'→ max size 30%'
                    )
                    _hc_min_usd = portfolio_value * 0.10
                    _hc_max_usd = portfolio_value * 0.30
                    _conf_scalar = max(0.0, min(1.0, (decision.confidence - 0.75) / 0.25))
                    sizing = dict(sizing)
                    sizing['position_usd'] = round(
                        _hc_min_usd + (_hc_max_usd - _hc_min_usd) * _conf_scalar, 2
                    )

                # ── Regime + VIX Position Size Multipliers ───────────────────
                # Applied post-calculation so config min/max defaults are never
                # mutated at runtime. Multipliers compound: final = base * regime * vix.
                _regime_mult = 1.0
                if market_regime == 'bull' and trade_str in ('short', 'sell_short'):
                    _regime_mult = 0.85  # shorting against bull market
                elif market_regime in ('sideways',):
                    _regime_mult = 0.85  # neutral/uncertain direction
                elif market_regime == 'bear' and trade_str == 'buy':
                    _regime_mult = 0.85  # longing against bear market
                # bull+long and bear+short stay at 1.0 (full tailwind)

                _vix_mult = 1.0
                if vix_regime == 'HIGH VOLATILITY':
                    _vix_mult = 0.80
                # NORMAL and LOW VOLATILITY stay at 1.0

                _base_size = sizing['position_usd']
                _adjusted_size = round(_base_size * _regime_mult * _vix_mult, 2)
                if _regime_mult != 1.0 or _vix_mult != 1.0:
                    print(
                        f'📏 {ticker} — position size adjusted: '
                        f'${_base_size:.2f} → ${_adjusted_size:.2f} '
                        f'({market_regime} regime, VIX {vix_regime})'
                    )
                decision.position_size_usd  = _adjusted_size
                decision.stop_loss_price    = sizer.get_stop_loss(
                    market_data.current_price, decision.trade_type, hold,
                    atr_pct=market_data.atr_pct, ticker=ticker,
                )
                decision.take_profit_price  = sizer.get_take_profit(
                    market_data.current_price, decision.trade_type, hold,
                    atr_pct=market_data.atr_pct, ticker=ticker,
                )
                decision.max_hold_days      = sizer.get_max_hold_days(hold)

                # Log whether ATR-based or fixed stops were applied
                if sizer._last_atr_stop_pct is not None and sizer._last_atr_target_pct is not None:
                    print(f'🎯 ATR-based stops: {ticker} — stop {sizer._last_atr_stop_pct*100:.1f}% / target {sizer._last_atr_target_pct*100:.1f}% (ATR: {market_data.atr_pct:.1f}%)')
                else:
                    print(f'⚠️  ATR unavailable for {ticker} — using fixed stops')

                # ORB gate — block ALL new entries before 9:45 AM ET regardless
                # of trade_type. The Risk Manager prompt states this rule but the
                # LLM can ignore it; this hard gate enforces it unconditionally.
                if et_now.time() < orb_cutoff:
                    print(
                        f'⏰ {ticker} — ORB gate: no entries before 9:45 AM ET '
                        f'({et_now.strftime("%H:%M ET")}), skipping {decision.trade_type}'
                    )
                    continue

                # High-impact macro day gate — extends the entry blackout to 10:30 AM ET
                # on CPI/NFP/GDP/PPI/FOMC days to avoid gap-and-reverse fills.
                if is_high_impact and et_now.time() < high_impact_cutoff:
                    print(
                        f'⚠️  {ticker} — High-impact day ({macro_event}): entries blocked until '
                        f'10:30 AM ET ({et_now.strftime("%H:%M ET")}), skipping {decision.trade_type}'
                    )
                    continue

                # ORB breakout → market order ─────────────────────────────
                # orb_breakout_up/down live on MarketData, not TradeDecision,
                # so this override must happen here where both are in scope.
                # Highest-conviction intraday signal: accept any fill price.
                if (
                    decision.order_type == 'limit'
                    and decision.trade_type is not None
                ):
                    if trade_str == 'buy' and market_data.orb_breakout_up:
                        decision.order_type = 'market'
                        print(f'[crew] {ticker} — ORB breakout up → market order')
                    elif trade_str in ('short', 'sell_short') and market_data.orb_breakout_down:
                        decision.order_type = 'market'
                        print(f'[crew] {ticker} — ORB breakout down → market order')

                # ── 2-Bar Momentum Confirmation Gate ─────────────────────────
                # Require recent price movement to align with trade direction.
                # High-confidence entries (>= 0.85) may override minor opposing
                # momentum (<= 0.25%) but are still blocked on strong opposition.
                # Skips on data failure (None) so a bad fetch never blocks a trade.
                _momentum = _get_2bar_momentum(ticker)
                if _momentum is not None:
                    if trade_str == 'buy' and _momentum < 0:
                        if decision.confidence >= 0.85 and _momentum >= -0.25:
                            print(
                                f'⚠️ {ticker} — minor opposing momentum ({_momentum:+.2f}%) '
                                f'overridden by high confidence ({decision.confidence:.2f}), proceeding'
                            )
                        else:
                            print(
                                f'⏭️ {ticker} — LONG skipped: strong opposing momentum '
                                f'({_momentum:+.2f}%), confidence {decision.confidence:.2f} '
                                f'insufficient to override'
                            ) if decision.confidence >= 0.85 else print(
                                f'⏭️ {ticker} — LONG skipped: price pulling back on last 2 bars '
                                f'({_momentum:+.2f}%), waiting for momentum to confirm up'
                            )
                            continue
                    elif trade_str in ('short', 'sell_short') and _momentum > 0:
                        if decision.confidence >= 0.85 and _momentum <= 0.25:
                            print(
                                f'⚠️ {ticker} — minor opposing momentum ({_momentum:+.2f}%) '
                                f'overridden by high confidence ({decision.confidence:.2f}), proceeding'
                            )
                        else:
                            print(
                                f'⏭️ {ticker} — SHORT skipped: strong opposing momentum '
                                f'({_momentum:+.2f}%), confidence {decision.confidence:.2f} '
                                f'insufficient to override'
                            ) if decision.confidence >= 0.85 else print(
                                f'⏭️ {ticker} — SHORT skipped: price bouncing up on last 2 bars '
                                f'({_momentum:+.2f}%), waiting for momentum to resume down'
                            )
                            continue

                # ── SPY Momentum Confirmation Gate ────────────────────────────
                # Require SPY 2-bar momentum to align with trade direction.
                # Skips on data failure (None) so a bad fetch never blocks a trade.
                _spy_momentum = _get_2bar_momentum('SPY')
                if _spy_momentum is not None:
                    if trade_str == 'buy' and _spy_momentum < 0:
                        print(
                            f'⏭️ {ticker} — LONG skipped: SPY trending down '
                            f'({_spy_momentum:+.2f}%), no market tailwind'
                        )
                        continue
                    elif trade_str in ('short', 'sell_short') and _spy_momentum > 0:
                        print(
                            f'⏭️ {ticker} — SHORT skipped: SPY trending up '
                            f'({_spy_momentum:+.2f}%), no market tailwind'
                        )
                        continue

                # Submit the bracket order to Alpaca
                order_result = executor.execute_trade(decision)
                _order_status = order_result.get('status', 'unknown')
                _order_id = order_result.get('order_id', '')
                print(f'📋 {ticker} order: {_order_status}{f" | id={_order_id[:8]}" if _order_id else ""}')

                if order_result.get('status') == 'placed':
                    trades_executed += 1

                    # Resolve the actual fill price for entry_price.
                    # Priority: (1) Alpaca avg_entry_price from the live position
                    # after a brief settle — this is the true fill price for both
                    # market and limit orders and is always populated by Alpaca.
                    # (2) decision.entry_price (limit price or agent estimate).
                    # (3) market_data.current_price (pre-order snapshot).
                    # Storing a correct price here is critical — NULL entry_price
                    # disables ALL dynamic exits (loss/profit/time) in position_monitor.
                    actual_entry_price = decision.entry_price or market_data.current_price
                    try:
                        time.sleep(2)  # Allow market order to fill before querying
                        live_pos_list = executor.get_open_positions()
                        for _pos in live_pos_list:
                            if _pos['ticker'] == ticker and _pos.get('avg_entry_price'):
                                actual_entry_price = _pos['avg_entry_price']
                                print(f'[executor] {ticker} — actual fill price from Alpaca: ${actual_entry_price:.2f}')
                                break
                    except Exception as _e:
                        print(f'[executor] {ticker} — could not fetch Alpaca fill price, using estimate: {_e}')

                    # Build the full trade record for both SQLite and the JSON journal.
                    # trade_id is a UUID generated here rather than by the database so
                    # it can be referenced in logs before the DB write completes.
                    trade_record = {
                        'trade_id':               str(uuid.uuid4()),
                        'ticker':                 ticker,
                        'trade_type':             decision.trade_type,
                        'order_type':             decision.order_type,
                        'hold_period':            decision.hold_period,
                        'max_hold_days':          decision.max_hold_days,
                        'entry_price':            actual_entry_price,
                        'exit_price':             None,       # Populated at close
                        'shares':                 sizing['shares'],
                        'position_size_usd':      sizing['position_usd'],
                        'stop_loss_price':        decision.stop_loss_price,
                        'take_profit_price':      decision.take_profit_price,
                        'pnl':                    None,       # Populated at close
                        'pnl_pct':                None,       # Populated at close
                        'status':                 'open',
                        'exit_reason':            None,       # Set by position_monitor or executor
                        'confidence_at_entry':    decision.confidence,
                        'bull_reasoning':         decision.bull_reasoning,
                        'bear_reasoning':         decision.bear_reasoning,
                        'risk_manager_reasoning': decision.risk_manager_reasoning,
                        'hold_period_reasoning':  decision.hold_period_reasoning,
                        'data_sources_available': str(market_data.data_sources_used.model_dump()),
                        'atr_pct':                market_data.atr_pct,  # Stored for ATR-tiered exit logic
                        'entry_time':             datetime.now().isoformat(),
                        'exit_time':              None,       # Populated at close
                    }

                    # Write to both persistence layers — SQLite for querying,
                    # JSON journal for human-readable audit trail
                    try:
                        db.insert_trade(trade_record)
                        print(f'✅ Trade record saved to DB: {ticker}')
                    except Exception as e:
                        print(f'❌ DB insert failed for {ticker}: {e}')
                        log_error('database_insert', ticker, str(e))
                    log_trade(trade_record)

            else:
                # Decision was execute=False or no trade_type — normal outcome
                print(f'⏭️  {ticker} — no trade (confidence: {decision.confidence:.2f})')

        except Exception as e:
            # Log the error and continue to the next ticker — one bad ticker
            # should never abort the entire watchlist cycle
            try:
                db.conn.rollback()
            except Exception:
                pass
            log_error('crew', ticker, str(e))
            print(f'❌ Error analyzing {ticker}: {e}')
            _cycle_errors += 1  # Item 4
            continue

    # ── Cycle Summary ─────────────────────────────────────────────────────────
    run_log.trades_executed  = trades_executed
    run_log.duration_seconds = (datetime.now() - start_time).total_seconds()
    log_run(run_log)

    print(
        f'\n✅ Cycle complete in {run_log.duration_seconds:.1f}s | '
        f'analyzed: {_cycle_analyzed} | '
        f'errors: {_cycle_errors} | '
        f'high_vol_gated: {_cycle_high_vol_gated} | '
        f'strategies_skipped: {_cycle_strategies_skipped} | '
        f'trades: {trades_executed}'
    )


# ── News-Triggered Single-Ticker Analysis ─────────────────────────────────────

def run_single_ticker(ticker: str, headline: str, position_multiplier: float = 1.0):
    """
    Run a full 4-agent crew analysis on a single ticker triggered by breaking news.

    Called by the news monitor background thread in scheduler.py when a high-impact
    headline mentioning a known ticker is detected. Mirrors the per-ticker logic
    in run_trading_cycle() but is optimised for speed — no watchlist loop, no
    run log, and the headline is injected directly into the agent summary so the
    crew weights it heavily.

    Position multiplier:
        1.0 — ticker is in config.watchlist (full position size)
        0.5 — ticker is S&P 500 universe only (half size — less analytical context)

    Args:
        ticker:              Symbol to analyse.
        headline:            Breaking news headline that triggered this call.
        position_multiplier: Scaling factor applied to the calculated position size.
    """
    try:
        print(f'\n🚨 News-triggered analysis: {ticker}')
        print(f'   Headline: {headline[:80]}')

        # Collect market data — skip if Alpaca is unreachable (no price = can't size)
        market_data = collector.collect(ticker)
        if not market_data.data_sources_used.alpaca:
            print(f'⚠️  No price data for {ticker} — skipping')
            return

        # Circuit breaker check before placing any news-triggered order
        portfolio_value = executor.get_portfolio_value()
        if not cb.check(portfolio_value):
            print('🚨 Circuit breaker active — skipping news trade')
            return

        # Label shown to agents so they understand the reduced position context
        position_label = 'FULL' if position_multiplier == 1.0 else 'HALF (non-watchlist)'

        # Headline is surfaced prominently at the top of the summary and again
        # at the bottom with an explicit instruction to weight it heavily
        summary = f'''
            Ticker: {ticker}
            BREAKING NEWS TRIGGER: {headline}
            Price: ${market_data.current_price:.2f}
            Volume: {market_data.volume:,}
            RSI: {market_data.rsi if market_data.rsi else 'N/A'}
            MACD: {market_data.macd if market_data.macd else 'N/A'}
            News Headlines: {market_data.news_headlines[:3]}
            Macro Context: {market_data.macro_context or 'N/A'}
            Position Size: {position_label}
            Data Sources: {market_data.data_sources_used.model_dump()}
            This analysis was triggered by breaking news.
            Weight the news headline heavily in your decision.
        '''

        # Fresh agents per call — news triggers are infrequent enough that
        # the instantiation overhead is negligible
        bull_agent      = create_bull_agent()
        bear_agent      = create_bear_agent()
        risk_agent      = create_risk_manager()
        portfolio_agent = create_portfolio_manager()

        bull_task      = create_bull_task(bull_agent, ticker, summary)
        bear_task      = create_bear_task(bear_agent, ticker, summary)
        risk_task      = create_risk_manager_task(risk_agent, ticker, bull_task, bear_task)

        open_positions = executor.get_open_positions()
        portfolio_task = create_portfolio_task(portfolio_agent, ticker, risk_task, open_positions)

        crew = Crew(
            agents=[bull_agent, bear_agent, risk_agent, portfolio_agent],
            tasks=[bull_task, bear_task, risk_task, portfolio_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        # Parse decision — same dual-path fallback as run_trading_cycle()
        if hasattr(result, 'json_dict') and result.json_dict:
            raw_dict = result.json_dict
        else:
            raw = result.raw if hasattr(result, 'raw') else str(result)
            raw_dict = json.loads(raw)

        # Safety override — same hierarchy enforcement as run_trading_cycle()
        _VALID_TRADE_TYPES = {'buy', 'sell', 'short', 'cover'}
        if isinstance(raw_dict.get('trade_type'), str):
            if raw_dict['trade_type'] not in _VALID_TRADE_TYPES:
                raw_dict['trade_type'] = None

        risk_execute = None
        try:
            risk_out = risk_task.output
            if hasattr(risk_out, 'json_dict') and risk_out.json_dict:
                risk_execute = risk_out.json_dict.get('execute')
            elif hasattr(risk_out, 'raw') and risk_out.raw:
                _r = risk_out.raw.strip()
                if _r.startswith('```'):
                    _r = _r.split('\n', 1)[-1]
                if _r.endswith('```'):
                    _r = _r.rsplit('```', 1)[0]
                risk_execute = json.loads(_r.strip()).get('execute')
        except Exception:
            pass

        if risk_execute is False and raw_dict.get('execute') is True:
            print(f'⚠️  Safety override: Portfolio Manager attempted to flip execute=false to execute=true for {ticker} — blocked')
            raw_dict['execute']           = False
            raw_dict['trade_type']        = None
            raw_dict['entry_price']       = None
            raw_dict['stop_loss_price']   = None
            raw_dict['take_profit_price'] = None
            raw_dict['position_size_usd'] = None

        decision = TradeDecision(**raw_dict)

        # ── Position Sizing & Execution ───────────────────────────────────────
        if decision.execute and decision.trade_type:
            hold = HoldPeriod(decision.hold_period) if decision.hold_period else HoldPeriod.SWING
            sizing = sizer.calculate(
                portfolio_value, market_data.current_price, decision.confidence, hold
            )

            # Scale down position for non-watchlist universe stocks
            sizing['position_usd'] = sizing['position_usd'] * position_multiplier
            sizing['shares']       = round(sizing['position_usd'] / market_data.current_price, 2)

            decision.position_size_usd  = sizing['position_usd']
            decision.stop_loss_price    = sizer.get_stop_loss(
                market_data.current_price, decision.trade_type, hold,
                atr_pct=market_data.atr_pct, ticker=ticker,
            )
            decision.take_profit_price  = sizer.get_take_profit(
                market_data.current_price, decision.trade_type, hold,
                atr_pct=market_data.atr_pct, ticker=ticker,
            )
            if sizer._last_atr_stop_pct is not None and sizer._last_atr_target_pct is not None:
                print(f'🎯 ATR-based stops: {ticker} — stop {sizer._last_atr_stop_pct*100:.1f}% / target {sizer._last_atr_target_pct*100:.1f}% (ATR: {market_data.atr_pct:.1f}%)')
            else:
                print(f'⚠️  ATR unavailable for {ticker} — using fixed stops')
            decision.max_hold_days      = sizer.get_max_hold_days(hold)

            # ORB gate — applies to news-triggered trades as well as scheduled cycles.
            # Block ALL new entries before 9:45 AM ET regardless of trade_type.
            _et_now_news = datetime.now(ZoneInfo('America/New_York'))
            if _et_now_news.time() < time(9, 45):
                print(
                    f'⏰ {ticker} (news) — ORB gate: no entries before 9:45 AM ET '
                    f'({_et_now_news.strftime("%H:%M ET")}), skipping {decision.trade_type}'
                )
                return

            # High-impact macro day gate — same 10:30 AM ET cutoff as main cycle.
            # Also raises the confidence threshold for this news-triggered trade.
            _is_high_impact, _macro_event = check_high_impact_day()
            if _is_high_impact:
                config.confidence_threshold = max(config.confidence_threshold, 0.87)
                if _et_now_news.time() < time(10, 30):
                    print(
                        f'⚠️  {ticker} (news) — High-impact day ({_macro_event}): entries blocked until '
                        f'10:30 AM ET ({_et_now_news.strftime("%H:%M ET")}), skipping {decision.trade_type}'
                    )
                    return

            order_result = executor.execute_trade(decision)
            _order_status = order_result.get('status', 'unknown')
            _order_id = order_result.get('order_id', '')
            print(f'📋 {ticker} (news) order: {_order_status}{f" | id={_order_id[:8]}" if _order_id else ""}')

            if order_result.get('status') == 'placed':
                import uuid
                # Resolve actual fill price — same priority chain as main cycle
                actual_entry_price = decision.entry_price or market_data.current_price
                try:
                    time.sleep(2)
                    live_pos_list = executor.get_open_positions()
                    for _pos in live_pos_list:
                        if _pos['ticker'] == ticker and _pos.get('avg_entry_price'):
                            actual_entry_price = _pos['avg_entry_price']
                            print(f'[executor] {ticker} (news) — actual fill price from Alpaca: ${actual_entry_price:.2f}')
                            break
                except Exception as _e:
                    print(f'[executor] {ticker} (news) — could not fetch Alpaca fill price, using estimate: {_e}')

                trade_record = {
                    'trade_id':               str(uuid.uuid4()),
                    'ticker':                 ticker,
                    'trade_type':             decision.trade_type,
                    'order_type':             decision.order_type,
                    'hold_period':            decision.hold_period,
                    'max_hold_days':          decision.max_hold_days,
                    'entry_price':            actual_entry_price,
                    'exit_price':             None,
                    'shares':                 sizing['shares'],
                    'position_size_usd':      sizing['position_usd'],
                    'stop_loss_price':        decision.stop_loss_price,
                    'take_profit_price':      decision.take_profit_price,
                    'pnl':                    None,
                    'pnl_pct':                None,
                    'status':                 'open',
                    # exit_reason stores the triggering headline for audit trail
                    'exit_reason':            f'news_triggered: {headline[:50]}',
                    'confidence_at_entry':    decision.confidence,
                    'bull_reasoning':         decision.bull_reasoning,
                    'bear_reasoning':         decision.bear_reasoning,
                    'risk_manager_reasoning': decision.risk_manager_reasoning,
                    'hold_period_reasoning':  decision.hold_period_reasoning,
                    'data_sources_available': str(market_data.data_sources_used.model_dump()),
                    'atr_pct':                market_data.atr_pct,  # Stored for ATR-tiered exit logic
                    'entry_time':             datetime.now().isoformat(),
                    'exit_time':              None,
                }
                try:
                    db.insert_trade(trade_record)
                    print(f'✅ Trade record saved to DB: {ticker}')
                except Exception as e:
                    print(f'❌ DB insert failed for {ticker}: {e}')
                    log_error('database_insert', ticker, str(e))
                log_trade(trade_record)
                print(f'✅ News trade placed: {ticker} ${sizing["position_usd"]:.2f}')

        else:
            print(f'⏭️  {ticker} news analyzed — no trade (confidence: {decision.confidence:.2f})')

    except Exception as e:
        log_error('run_single_ticker', ticker, str(e))
        print(f'❌ Error in news-triggered analysis for {ticker}: {e}')
