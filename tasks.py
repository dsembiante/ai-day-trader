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
from models import AgentAnalysis, TradeDecision
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
            Analyze {ticker} from a BULLISH perspective using this market data:
            {market_data_summary}

            You MUST return a JSON object with these exact fields:
            - ticker: '{ticker}'
            - recommendation: one of 'buy', 'sell', 'short', 'cover'
            - confidence: float between 0.0 and 1.0
            - reasoning: your full analysis (minimum 50 characters)
            - key_factors: list of at least 2 specific factors supporting your view
            - recommended_hold_period: one of 'intraday', 'swing', 'position'
            - hold_period_reasoning: why you chose this hold period

            For hold period guidance:
            - intraday:  strong short-term momentum, news catalyst, tight setup
            - swing:     solid multi-day technical setup, moderate conviction
            - position:  strong fundamental thesis, high conviction, longer timeframe
        ''',
        expected_output='JSON matching AgentAnalysis schema with bullish perspective',
        output_json=AgentAnalysis,  # Pydantic validation enforced by CrewAI
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
            Analyze {ticker} from a BEARISH perspective using this market data:
            {market_data_summary}

            You MUST return a JSON object with these exact fields:
            - ticker: '{ticker}'
            - recommendation: one of 'buy', 'sell', 'short', 'cover'
            - confidence: float between 0.0 and 1.0
            - reasoning: your full risk analysis (minimum 50 characters)
            - key_factors: list of at least 2 specific risks or bearish factors
            - recommended_hold_period: one of 'intraday', 'swing', 'position'
            - hold_period_reasoning: timeframe of the risk you see
        ''',
        expected_output='JSON matching AgentAnalysis schema with bearish perspective',
        output_json=AgentAnalysis,
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

            IMPORTANT: Only set execute=true if confidence >= {config.confidence_threshold}.
            When in doubt, do nothing — execute=false is always valid.
        ''',
        expected_output='JSON matching TradeDecision schema',
        output_json=TradeDecision,
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

            Review the risk manager decision from context.
            If execute=true, verify:
            1. We have not exceeded max positions ({config.max_positions})
            2. We do not already have an open position in {ticker}
            3. Adding this position does not over-concentrate in one sector

            Return the same TradeDecision JSON from context, but set execute=false
            if any of the above conditions are violated. Otherwise pass it through unchanged.
            Always include your reasoning in risk_manager_reasoning.
        ''',
        expected_output='JSON matching TradeDecision schema, approved or rejected',
        output_json=TradeDecision,
        agent=agent,
        context=[risk_task],  # Portfolio manager sees only the risk manager's final decision
    )
