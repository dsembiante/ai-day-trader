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
from datetime import datetime
import json
import uuid


# ── Module-Level Singletons ───────────────────────────────────────────────────
# Instantiated once at import time and reused for every ticker across all
# scheduler cycles within the process lifetime. This avoids opening new
# database connections and API clients on every call to run_trading_cycle().
collector = DataCollector()
sizer     = PositionSizer()
executor  = TradeExecutor()
db        = Database()


# ── Main Cycle ────────────────────────────────────────────────────────────────

def run_trading_cycle(circuit_breaker: CircuitBreaker):
    """
    Execute one full analysis and trading cycle across the entire watchlist.

    Called by scheduler.py at each scheduled interval. Steps:
        1. Circuit breaker check — halt immediately if portfolio is down ≥10%
        2. Position monitor — close any positions that have exceeded their hold period
        3. Per-ticker crew run — collect data → analyse → decide → size → execute
        4. Persist run summary to logs

    Args:
        circuit_breaker: Shared CircuitBreaker instance from scheduler.py.
                         Passed in (rather than instantiated here) so the peak
                         value high-water mark persists across cycles.
    """
    run_log = new_run_log(config.watchlist)
    start_time = datetime.now()

    # ── Gate 1: Circuit Breaker ───────────────────────────────────────────────
    # Fetch portfolio value first — required for both the breaker check and
    # position sizing later in the cycle.
    portfolio_value = executor.get_portfolio_value()

    if not circuit_breaker.check(portfolio_value):
        # Breaker has fired — log, record, and abort the entire cycle.
        # No new positions should be opened until manual review clears the breaker.
        print('🚨 Circuit breaker active — skipping trading cycle')
        run_log.circuit_breaker_triggered = True
        log_run(run_log)
        return

    # ── Gate 2: Hold Period Enforcement ──────────────────────────────────────
    # Check all open positions for time-based expiry before analysing new ones.
    # Closing stale positions first keeps max_positions headroom accurate for
    # the portfolio task later in the cycle.
    monitor = PositionMonitor(executor)
    monitor.check_all_positions()

    # Snapshot of open positions after any expired ones have been closed.
    # Passed to the portfolio task to enforce max_positions and duplicate checks.
    open_positions = executor.get_open_positions()
    trades_executed = 0

    # ── Agent Instantiation ───────────────────────────────────────────────────
    # Agents are created once per cycle (not per ticker) and reused.
    # Each agent holds the same shared LLM client from agents.py, so creating
    # them once avoids redundant LLM client setup across the watchlist.
    bull_agent      = create_bull_agent()
    bear_agent      = create_bear_agent()
    risk_agent      = create_risk_manager()
    portfolio_agent = create_portfolio_manager()

    # ── Per-Ticker Loop ───────────────────────────────────────────────────────
    for ticker in config.watchlist:
        try:
            print(f'\n📊 Analyzing {ticker}...')

            # ── Data Collection ───────────────────────────────────────────────
            # collect() returns partial data on source failure — DataSourceStatus
            # tracks which sources were reachable so agents can adjust confidence.
            market_data = collector.collect(ticker)

            # Without a price from Alpaca we cannot size a position — skip entirely
            if not market_data.data_sources_used.alpaca:
                print(f'⚠️  Skipping {ticker} — no price data available')
                continue

            # ── Market Data Summary ───────────────────────────────────────────
            # Pre-format all signals into a single string injected into each
            # agent prompt. Inline formatting handles None values gracefully
            # so the LLM never sees Python's 'None' string in the context.
            summary = f'''
                Ticker: {ticker}
                Price: ${market_data.current_price:.2f}
                Volume: {market_data.volume:,}
                RSI: {market_data.rsi:.1f if market_data.rsi else 'N/A'}
                MACD: {market_data.macd:.4f if market_data.macd else 'N/A'}
                50-day MA: {market_data.moving_avg_50:.2f if market_data.moving_avg_50 else 'N/A'}
                200-day MA: {market_data.moving_avg_200:.2f if market_data.moving_avg_200 else 'N/A'}
                News sentiment: {market_data.news_sentiment if market_data.news_sentiment else 'N/A'}
                Recent headlines: {market_data.news_headlines[:3]}
                Macro context: {market_data.macro_context or 'N/A'}
                Data sources available: {market_data.data_sources_used.model_dump()}
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
            if hasattr(result, 'json_dict') and result.json_dict:
                decision = TradeDecision(**result.json_dict)
            else:
                raw = result.raw if hasattr(result, 'raw') else str(result)
                decision = TradeDecision(**json.loads(raw))

            # ── Position Sizing & Execution ───────────────────────────────────
            if decision.execute and decision.trade_type:
                # Resolve hold period — default to SWING if the agent omitted it
                hold = HoldPeriod(decision.hold_period) if decision.hold_period else HoldPeriod.SWING

                # Calculate dollar size, share count, stop-loss, and take-profit
                sizing = sizer.calculate(
                    portfolio_value, market_data.current_price, decision.confidence, hold
                )
                decision.position_size_usd  = sizing['position_usd']
                decision.stop_loss_price    = sizer.get_stop_loss(
                    market_data.current_price, decision.trade_type, hold
                )
                decision.take_profit_price  = sizer.get_take_profit(
                    market_data.current_price, decision.trade_type, hold
                )
                decision.max_hold_days      = sizer.get_max_hold_days(hold)

                # Submit the bracket order to Alpaca
                order_result = executor.execute_trade(decision)

                if order_result.get('status') == 'placed':
                    trades_executed += 1

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
                        'entry_price':            market_data.current_price,
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
                        'entry_time':             datetime.now().isoformat(),
                        'exit_time':              None,       # Populated at close
                    }

                    # Write to both persistence layers — SQLite for querying,
                    # JSON journal for human-readable audit trail
                    db.insert_trade(trade_record)
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
