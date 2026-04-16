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
)
from tasks import (
    create_bull_task, create_bear_task,
    create_risk_manager_task, create_portfolio_task,
    create_exit_bull_task, create_exit_bear_task,
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
    orb_cutoff        = time(10, 0)   # ORB formation window — no new entries before 10:00 AM ET
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

    # ── Gate 3: 1:00 PM Entry Cutoff ─────────────────────────────────────────
    # After 1:00 PM ET, all new entry evaluation is blocked. Position monitoring
    # has already completed above — this gate only prevents the entry loop below.
    if et_now.time() >= time(13, 0):
        print(f'⛔ Past 1:00 PM ET — entries closed, monitoring positions only')
        log_run(run_log)
        return

    # Snapshot of open positions after any expired ones have been closed.
    # Passed to the portfolio task to enforce max_positions and duplicate checks.
    open_positions = executor.get_open_positions()
    alpaca_held_tickers = {p['ticker'] for p in open_positions}
    db_open_tickers     = {t['ticker'] for t in db.get_open_trades()}
    trades_executed = 0

    # Direction counts — used by portfolio task to enforce same-direction cap
    open_longs  = sum(1 for p in open_positions if float(p.get('qty', 0)) > 0)
    open_shorts = sum(1 for p in open_positions if float(p.get('qty', 0)) < 0)

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
    print(f'📊 Open positions: {open_longs} longs, {open_shorts} shorts (max {config.max_same_direction_positions} per direction)')

    if market_regime == 'bear':
        print('🐻 Bear market detected — reducing position sizes and favoring shorts')
        config.min_position_pct = 0.067  # ~$2,000 on $29.9k
        config.max_position_pct = 0.10   # ~$3,000 on $29.9k
    elif market_regime == 'sideways':
        print('➡️  Sideways market — being selective')
        config.min_position_pct = 0.10   # ~$3,000 on $29.9k
        config.max_position_pct = 0.134  # ~$4,000 on $29.9k
    else:
        # Bull market — restore full config targets ($4,000–$6,000 on $29.9k)
        config.min_position_pct = 0.134
        config.max_position_pct = 0.201

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
    # 0.87 and block all new entries until 10:30 AM ET — the additional 30 minutes
    # beyond the standard ORB window reduces exposure to gap-and-reverse patterns
    # that are most common immediately after high-impact releases.
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
                            f'⏸️ {ticker} — 30min cooloff after loss exit '
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
                print(
                    f'⚠️ {ticker} ATR {_ticker_atr:.2f}% — high volatility, '
                    f'requiring 3/4 signals and 0.85 confidence'
                )

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
            portfolio_task = create_portfolio_task(portfolio_agent, ticker, risk_task, open_positions, open_longs, open_shorts)

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

            # ── Decision Post-Processing ──────────────────────────────────────
            # If the agent omitted entry_price (returns null for market orders),
            # fall back to the current market price so the whole-share calculation
            # in trade_executor always has a price to work with. Alpaca rejects
            # bracket orders with fractional qty, so a price is always required.
            if decision.execute and not decision.entry_price:
                decision.entry_price = market_data.current_price

            # ── Position Sizing & Execution ───────────────────────────────────
            if decision.execute and decision.trade_type:
                # Resolve hold period — default to SWING if the agent omitted it
                requested_hold = HoldPeriod(decision.hold_period) if decision.hold_period else HoldPeriod.SWING
                hold = sizer.get_hold_period_safe(requested_hold)
                decision.hold_period = hold.value  # Reflect any PDT upgrade in the trade record

                # Calculate dollar size, share count, stop-loss, and take-profit
                sizing = sizer.calculate(
                    portfolio_value, market_data.current_price, decision.confidence, hold
                )
                decision.position_size_usd  = sizing['position_usd']
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

                # ORB gate — block ALL new entries before 10:00 AM ET regardless
                # of trade_type. The Risk Manager prompt states this rule but the
                # LLM can ignore it; this hard gate enforces it unconditionally.
                if et_now.time() < orb_cutoff:
                    print(
                        f'⏰ {ticker} — ORB gate: no entries before 10:00 AM ET '
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
                    trade_str = decision.trade_type.value if hasattr(decision.trade_type, 'value') else str(decision.trade_type)
                    if trade_str == 'buy' and market_data.orb_breakout_up:
                        decision.order_type = 'market'
                        print(f'[crew] {ticker} — ORB breakout up → market order')
                    elif trade_str in ('short', 'sell_short') and market_data.orb_breakout_down:
                        decision.order_type = 'market'
                        print(f'[crew] {ticker} — ORB breakout down → market order')

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
            log_error('crew', ticker, str(e))
            print(f'❌ Error analyzing {ticker}: {e}')
            continue

    # ── Cycle Summary ─────────────────────────────────────────────────────────
    run_log.trades_executed  = trades_executed
    run_log.duration_seconds = (datetime.now() - start_time).total_seconds()
    log_run(run_log)

    print(
        f'\n✅ Cycle complete — {trades_executed} trades executed '
        f'in {run_log.duration_seconds:.1f}s'
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
        news_open_longs  = sum(1 for p in open_positions if float(p.get('qty', 0)) > 0)
        news_open_shorts = sum(1 for p in open_positions if float(p.get('qty', 0)) < 0)
        portfolio_task = create_portfolio_task(portfolio_agent, ticker, risk_task, open_positions, news_open_longs, news_open_shorts)

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
            # Block ALL new entries before 10:00 AM ET regardless of trade_type.
            _et_now_news = datetime.now(ZoneInfo('America/New_York'))
            if _et_now_news.time() < time(10, 0):
                print(
                    f'⏰ {ticker} (news) — ORB gate: no entries before 10:00 AM ET '
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
