"""
tasks.py — CrewAI task definitions for the AI trading crew.

Each task wraps a natural-language prompt with a strict JSON output schema
(via output_json) so the LLM is forced to return structured, Pydantic-
validatable data rather than free-form text.

Task execution order per ticker:
    1. Bull Task     — independent, analyses long opportunity
    2. Bear Task     — independent, analyses downside/short opportunity
    3. Risk Task     — depends on (1) and (2); synthesises final decision
    4. Portfolio Task — depends on (3); validates against portfolio constraints

Tasks 1 and 2 can run in parallel; 3 and 4 are sequential gates.

The hold period guidance embedded in each prompt is intentionally repeated
across tasks — each agent must independently classify the timeframe so the
risk manager can evaluate consistency between bull and bear views.

Usage:
    from tasks import create_bull_task, create_bear_task
    from tasks import create_risk_manager_task, create_portfolio_task
"""

from crewai import Task
from config import config


# ── Analyst Tasks ─────────────────────────────────────────────────────────────

def create_bull_task(agent, ticker: str, market_data_summary: str) -> Task:
    """
    Task for the Bull Analyst agent to evaluate a long opportunity.

    The prompt embeds the full market data summary so the LLM has all signals
    in context without requiring tool calls. output_json=AgentAnalysis forces
    the response through Pydantic validation — any missing or malformed field
    raises a validation error that CrewAI will retry before returning.

    Hold period guidance is included in the prompt to ensure the bull agent
    classifies timeframe based on signal strength, not just direction.

    Args:
        agent:               The bull CrewAI agent instance from agents.py.
        ticker:              Symbol being analysed.
        market_data_summary: Pre-formatted string from crew.py containing
                             price, technicals, sentiment, and macro context.

    Returns:
        A configured Task ready to be added to the Crew.
    """
    return Task(
        description=f'''
            Analyze {ticker} from a BULLISH INTRADAY perspective using this market data:
            {market_data_summary}

            You are looking for same-day long opportunities with MULTI-SIGNAL CONFLUENCE.
            Evaluate each of these four bullish signals and count how many are present:

            Signal 1 — VWAP (primary):
            - Price above VWAP = BULLISH ✓  |  Price below VWAP = NOT bullish ✗

            Signal 2 — Opening Range Breakout:
            - ORB breakout up = True = BULLISH ✓  |  No breakout = NOT bullish ✗

            Signal 3 — Pre-market gap:
            - Gap % above +0.5% = BULLISH ✓  |  Gap below +0.5% = NOT bullish ✗

            Signal 4 — Volume confirmation:
            - Volume ratio above 1.20x = BULLISH ✓  |  Below 1.20x = NOT bullish ✗

            Count how many of the 4 signals are bullish (0–4).
            IMPORTANT: Only recommend 'buy' if at least 2 signals are bullish.
            If fewer than 2 signals are bullish, recommend 'sell' (no trade) with low confidence.

            You MUST return a JSON object with these exact fields:
            - ticker: '{ticker}'
            - recommendation: one of 'buy', 'sell', 'short', 'cover'
            - confidence: float between 0.0 and 1.0
            - reasoning: your full analysis including the signal count (minimum 50 characters)
            - key_factors: list the bullish signals that fired AND the signal count e.g. "2/4 bullish signals"
            - recommended_hold_period: always 'intraday' for this task
            - hold_period_reasoning: why the intraday setup is valid right now

            IMPORTANT: If orb_breakout_down is True OR price_above_vwap is False OR gap_is_bearish is True, you should reduce your bullish conviction significantly. These intraday signals override general technical analysis for day trading purposes. A stock can have good fundamentals but still be a short candidate on a specific day based on intraday momentum.

            Be concise — keep reasoning under 3 sentences, key_factors to 2-4 items.
        ''',
        expected_output='JSON object with ticker, recommendation, confidence, reasoning, key_factors, recommended_hold_period, hold_period_reasoning',
        agent=agent,
    )


def create_bear_task(agent, ticker: str, market_data_summary: str) -> Task:
    """
    Task for the Bear Analyst agent to evaluate downside risk and short setups.

    Mirrors the bull task structure but prompts for bearish framing.
    The hold_period_reasoning field captures whether the risk is short-lived
    (intraday catalyst reversal) or structural (multi-week deterioration),
    giving the risk manager a timeframe dimension to compare against the bull.

    Args:
        agent:               The bear CrewAI agent instance from agents.py.
        ticker:              Symbol being analysed.
        market_data_summary: Same pre-formatted data string passed to the bull task.

    Returns:
        A configured Task ready to be added to the Crew.
    """
    return Task(
        description=f'''
            Analyze {ticker} from a BEARISH INTRADAY perspective using this market data:
            {market_data_summary}

            You are looking for same-day short opportunities with MULTI-SIGNAL CONFLUENCE.
            Evaluate each of these four bearish signals and count how many are present:

            Signal 1 — VWAP (primary):
            - Price below VWAP = BEARISH ✓  |  Price above VWAP = NOT bearish ✗

            Signal 2 — Opening Range Breakdown:
            - ORB breakdown = True = BEARISH ✓  |  No breakdown = NOT bearish ✗

            Signal 3 — Pre-market gap:
            - Gap % below -0.5% = BEARISH ✓  |  Gap above -0.5% = NOT bearish ✗

            Signal 4 — Volume on down move:
            - Volume ratio above 1.20x while price is declining = BEARISH ✓  |  Low volume = NOT bearish ✗

            Count how many of the 4 signals are bearish (0–4).
            IMPORTANT: Only recommend 'short' if at least 2 signals are bearish.
            If fewer than 2 signals are bearish, recommend 'buy' (no short trade) with low confidence.

            You MUST return a JSON object with these exact fields:
            - ticker: '{ticker}'
            - recommendation: one of 'buy', 'sell', 'short', 'cover'
            - confidence: float between 0.0 and 1.0
            - reasoning: your full risk analysis including the signal count (minimum 50 characters)
            - key_factors: list the bearish signals that fired AND the signal count e.g. "2/4 bearish signals"
            - recommended_hold_period: always 'intraday' for this task
            - hold_period_reasoning: why the bearish intraday setup is valid right now

            Be concise — keep reasoning under 3 sentences, key_factors to 2-4 items.
        ''',
        expected_output='JSON object with ticker, recommendation, confidence, reasoning, key_factors, recommended_hold_period, hold_period_reasoning',
        agent=agent,
    )


# ── Decision Tasks ────────────────────────────────────────────────────────────

def create_risk_manager_task(agent, ticker: str, bull_task: Task, bear_task: Task) -> Task:
    """
    Task for the Risk Manager agent to synthesise bull and bear analyses
    into a single actionable TradeDecision.

    context=[bull_task, bear_task] causes CrewAI to pass the outputs of both
    analyst tasks into this task's context window automatically, so the risk
    manager sees both JSON responses without any manual wiring.

    The prompt explicitly instructs the agent to leave position_size_usd,
    stop_loss_price, and take_profit_price as null — those are computed by
    position_sizer.py after the decision is returned, using the entry_price
    and hold_period chosen here.

    The confidence threshold is interpolated from config so the prompt always
    reflects the live setting without hardcoding 0.75 in the task string.

    Args:
        agent:     The risk manager CrewAI agent instance.
        ticker:    Symbol being evaluated.
        bull_task: Completed bull analyst task (passed as context).
        bear_task: Completed bear analyst task (passed as context).

    Returns:
        A configured Task that receives both analyst outputs as context.
    """
    return Task(
        description=f'''
            CRITICAL: This is a day trading system. You MUST always set hold_period to intraday and max_hold_days to 1. Never recommend swing or position hold periods under any circumstances. All positions must be closed within the same trading day by the end-of-day position monitor.

            You are the Risk Manager for {ticker}.
            Review BOTH the bull and bear analyses provided in context.

            Make the final trade decision. You MUST return a JSON object with:
            - ticker: '{ticker}'
            - execute: true or false
            - trade_type: 'buy', 'sell', 'short', or 'cover' (required if execute=true)
            - order_type: 'limit' (preferred) or 'market'
            - hold_period: 'intraday', 'swing', or 'position'
            - confidence: float between 0.0 and 1.0
            - position_size_usd: null (position sizer calculates this)
            - entry_price: suggested entry price (null for market orders)
            - stop_loss_price: null (position sizer calculates this)
            - take_profit_price: null (position sizer calculates this)
            - max_hold_days: maximum days to hold (1 for intraday, 5 for swing, 20 for position)
            - bull_reasoning: summary of bull argument
            - bear_reasoning: summary of bear argument
            - risk_manager_reasoning: your final decision reasoning (minimum 50 chars)
            - hold_period_reasoning: why you chose this hold period

            STRICT INTRADAY RULES — this is a DAY TRADING system:
            - hold_period MUST always be 'intraday'
            - max_hold_days MUST always be 1
            - All positions are force-closed at end of day — no overnight holds ever
            - Do NOT approve any trade before 10:00 AM EST — the opening range must fully form first
            - Only approve a trade if the analyst identified at least 2 confirming signals (look for "X/4 signals" in key_factors)
            - If only 1 signal is present, set confidence below {config.confidence_threshold} so the trade does not execute
            - Only set execute=true if confidence >= {config.confidence_threshold}
            - Favor HIGH confidence trades only — intraday has no time to recover from bad entries
            - When bull and bear signals conflict, prefer execute=false over a low-conviction trade
            - When in doubt, do nothing — execute=false is always the safe choice

            INTRADAY DIRECTION SIGNALS — weight these heavily:
            - If price_above_vwap is False, the intraday trend is DOWN — strongly favor SHORT over BUY.
            - If price_above_vwap is True, the intraday trend is UP — favor BUY over SHORT.
            - If orb_breakout_up is True, this is a confirmed bullish breakout — execute BUY with high confidence.
            - If orb_breakout_down is True, this is a confirmed bearish breakdown — execute SHORT with high confidence.
            - If gap_is_bearish is True, the day opened with bearish bias — favor SHORT for the entire session unless orb_breakout_up confirms a reversal.

            MARKET DIRECTION BIAS:
            - If market_regime is bear, strongly favor SHORT and COVER trade types over BUY.
            - In a bear market regime, only execute BUY trades if there are at least 3 strong confirming bullish signals.
            - When in doubt in a bear regime, prefer SHORT over BUY.

            DATA QUALITY GUIDANCE — evaluate these conditions before setting your confidence score:
            - If alpaca is False in data_sources_available: this ticker should have been skipped upstream — set execute=false.
            - If yfinance is False in data_sources_available: you are missing all technical indicators and fundamentals. Unless price action and news provide extremely compelling signals, set confidence below 0.78 or execute=false.
            - If finnhub is False in data_sources_available: this is expected on the current Finnhub plan — do NOT penalize confidence for missing Finnhub data. Proceed normally with available signals.
            - If finnhub is True AND news_sentiment is negative (below -0.3): treat this as a meaningful bearish signal and weight it in your confidence score.
            - If fred is False in data_sources_available: you are missing macro context. Apply extra caution on interest-rate-sensitive stocks such as utilities, REITs, and consumer staples.
            - If 2 or more sources other than finnhub show False in data_sources_available: set execute=false — insufficient data to make a reliable decision.

            VIX GUIDANCE — adjust sizing based on current volatility environment:
            - VIX above 25 (HIGH): Large intraday moves available. Reduce confidence threshold to 0.80 — more opportunities are valid. The position sizer will automatically increase size.
            - VIX below 15 (LOW): Small intraday moves expected. Raise confidence threshold to 0.87 — only the strongest setups are worth trading.
            - VIX 15-25 (NORMAL): Proceed with standard confidence threshold of 0.82.

            TIME-OF-DAY MOMENTUM FILTER — morning momentum window closes at 11 AM ET:
            - If session_phase is 'morning': no additional restrictions — proceed with standard signal evaluation.
            - If session_phase is 'midday' or 'afternoon':
              - Only execute BUY if vwap_margin_pct > 0.30% (price clearly above VWAP, not just hovering).
              - Only execute BUY if price_vs_orb_high % is within 1% above the ORB high (near breakout) OR price is making new intraday highs (orb_breakout_up is True).
              - If neither VWAP nor ORB condition is met, set execute=false.
              - Morning momentum window is closing — only the strongest setups are worth entering after 11 AM.
              - Valid exceptions: pullback-to-VWAP entries (price just crossed back above VWAP with volume confirmation) and genuine ORB breakouts are still valid at any time of day.

            Be concise — keep all reasoning fields under 2 sentences each.
        ''',
        expected_output='JSON object with ticker, execute, trade_type, order_type, hold_period, confidence, position_size_usd, entry_price, stop_loss_price, take_profit_price, max_hold_days, bull_reasoning, bear_reasoning, risk_manager_reasoning, hold_period_reasoning',
        agent=agent,
        context=[bull_task, bear_task],  # CrewAI injects both analyst outputs automatically
    )


def create_portfolio_task(agent, ticker: str, risk_task: Task, open_positions: list) -> Task:
    """
    Final portfolio-level gate before a trade decision reaches the executor.

    This task acts as a circuit-breaker for portfolio concentration:
        1. Max positions cap — blocks entry if the portfolio is already at max
        2. Duplicate position check — prevents doubling up on the same ticker
        3. Sector concentration — flags over-exposure to a single sector
        4. Multi-signal confirmation — requires at least 2 confirming signals

    Direction concentration is enforced as a hard code gate in crew.py via
    an 80% portfolio exposure cap, not here in the LLM prompt.

    The agent is instructed to pass the TradeDecision through unchanged if
    all checks pass, or flip execute=false with an explanation in
    risk_manager_reasoning if any check fails. Using the same output schema
    (TradeDecision) means crew.py can treat the portfolio task output
    identically to the risk manager output with no conditional logic.

    Args:
        agent:          The portfolio manager CrewAI agent instance.
        ticker:         Symbol being evaluated.
        risk_task:      The risk manager task whose output is passed as context.
        open_positions: Current open positions list from TradeExecutor.get_open_positions(),
                        used to populate the prompt with live portfolio state.

    Returns:
        A configured Task that receives the risk manager output as context.
    """
    return Task(
        description=f'''
            You are the Portfolio Manager reviewing the trade decision for {ticker}.
            Current open positions: {len(open_positions)} of {config.max_positions} maximum.
            Open tickers: {[p['ticker'] for p in open_positions]}

            CRITICAL PASSTHROUGH RULE: If the Risk Manager's decision has execute=false, you MUST return that exact JSON unchanged. You are absolutely forbidden from changing execute from false to true. You are absolutely forbidden from inventing entry_price, stop_loss_price, take_profit_price, or position_size_usd values. Your only job when execute=false is to pass the decision through unmodified.

            VALID trade_type values are ONLY: 'buy', 'sell', 'short', 'cover'. Never use 'long', 'hold', or any other value.

            Review the risk manager decision from context.
            If execute=true, verify ALL of the following:
            1. We have not exceeded max positions ({config.max_positions})
            2. We do not already have an open position in {ticker}
            3. Adding this position does not over-concentrate in one sector
            4. Multi-signal confirmation: the analyst reasoning mentions at least 2 confirming signals
               (look for "2/4", "3/4", or "4/4" in key_factors — if only "1/4" is present, set execute=false)

            If condition 4 fails, set execute=false and note "insufficient signal confirmation" in risk_manager_reasoning.
            Return the same TradeDecision JSON from context, but set execute=false
            if any of the above conditions are violated. Otherwise pass it through unchanged.
            Always include your reasoning in risk_manager_reasoning. Be concise — 1-2 sentences.
        ''',
        expected_output='JSON matching TradeDecision schema, approved or rejected',
        agent=agent,
        context=[risk_task],  # Portfolio manager sees only the risk manager's final decision
    )


# ── Exit Evaluation Tasks ─────────────────────────────────────────────────────

def create_exit_bull_task(agent, ticker: str, market_data_summary: str, entry_price: float) -> Task:
    """
    Task for evaluating whether to EXIT an existing long position.

    Asks the bull agent whether the original bullish thesis still holds.
    Returns a simple exit/hold decision — not a new entry decision.

    Args:
        agent:               The bull CrewAI agent instance.
        ticker:              Symbol of the held position.
        market_data_summary: Pre-formatted market data string from crew.py.
        entry_price:         Original entry price for unrealized P&L context.

    Returns:
        A configured Task producing a JSON exit decision.
    """
    return Task(
        description=f'''
            You are evaluating whether to EXIT an existing LONG position in {ticker}.
            Original entry price: {f'${entry_price:.2f}' if entry_price is not None else 'unknown (use current price for P&L context)'}

            Current market data:
            {market_data_summary}

            Evaluate whether the BULLISH thesis that justified this long position still holds.
            Check these exit signals:
            - Price below VWAP: intraday momentum has turned bearish → EXIT signal
            - ORB breakdown (orb_breakout_down=True): confirmed bearish break → EXIT signal
            - Bearish gap: day opened with bearish bias → EXIT signal
            - Volume on down move (volume_ratio > 1.20 while price declining) → EXIT signal

            Count how many exit signals are present (0-4).
            IMPORTANT: Recommend EXIT (sell) if 2 or more exit signals are present.
            Recommend HOLD if fewer than 2 exit signals are present.

            You MUST return a JSON object with these exact fields:
            - ticker: '{ticker}'
            - exit: true or false
            - trade_type: 'sell' (if exit=true) or 'hold' (if exit=false)
            - confidence: float between 0.0 and 1.0
            - reasoning: your analysis including signal count (minimum 50 characters)
            - key_factors: list of exit signals that fired

            Only recommend exit=true if confidence >= 0.75.
            Be concise — keep reasoning under 3 sentences.
        ''',
        expected_output='JSON object with ticker, exit, trade_type, confidence, reasoning, key_factors',
        agent=agent,
    )


def create_exit_bear_task(agent, ticker: str, market_data_summary: str, entry_price: float) -> Task:
    """
    Task for evaluating whether to EXIT an existing short position.

    Asks the bear agent whether the original bearish thesis still holds.
    Returns a simple exit/hold decision — not a new entry decision.

    Args:
        agent:               The bear CrewAI agent instance.
        ticker:              Symbol of the held position.
        market_data_summary: Pre-formatted market data string from crew.py.
        entry_price:         Original entry price for unrealized P&L context.

    Returns:
        A configured Task producing a JSON exit decision.
    """
    return Task(
        description=f'''
            You are evaluating whether to EXIT an existing SHORT position in {ticker}.
            Original entry price: {f'${entry_price:.2f}' if entry_price is not None else 'unknown (use current price for P&L context)'}

            Current market data:
            {market_data_summary}

            Evaluate whether the BEARISH thesis that justified this short position still holds.
            Check these cover signals (reasons the short thesis has broken down):
            - Price above VWAP: intraday momentum has turned bullish → COVER signal
            - ORB breakout up (orb_breakout_up=True): confirmed bullish break → COVER signal
            - Bullish gap: day opened with bullish bias → COVER signal
            - Volume on up move (volume_ratio > 1.20 while price rising) → COVER signal

            Count how many cover signals are present (0-4).
            IMPORTANT: Recommend COVER (exit short) if 2 or more cover signals are present.
            Recommend HOLD if fewer than 2 cover signals are present.

            You MUST return a JSON object with these exact fields:
            - ticker: '{ticker}'
            - exit: true or false
            - trade_type: 'cover' (if exit=true) or 'hold' (if exit=false)
            - confidence: float between 0.0 and 1.0
            - reasoning: your analysis including signal count (minimum 50 characters)
            - key_factors: list of cover signals that fired

            Only recommend exit=true if confidence >= 0.75.
            Be concise — keep reasoning under 3 sentences.
        ''',
        expected_output='JSON object with ticker, exit, trade_type, confidence, reasoning, key_factors',
        agent=agent,
    )
