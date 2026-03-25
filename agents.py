"""
agents.py — CrewAI agent definitions for the AI trading crew.

Four specialist agents collaborate on every ticker analysis cycle:

    Bull Analyst      — Identifies long opportunities and classifies hold period
    Bear Analyst      — Surfaces risks and short setups across timeframes
    Risk Manager      — Arbitrates bull/bear debate, sets final hold period,
                        and gates execution at ≥0.75 confidence
    Portfolio Manager — Ensures balanced exposure across hold period tiers

Each agent shares the same underlying Groq LLM instance, created once at
module load time via create_llm_with_retry(). Factory functions are provided
rather than module-level agent instances so the crew can recreate agents
between runs without re-initialising the LLM.

Usage:
    from agents import create_bull_agent, create_bear_agent
    from agents import create_risk_manager, create_portfolio_manager
"""

from crewai import Agent, LLM
from config import config


# ── LLM Setup ────────────────────────────────────────────────────────────────
# Groq exposes an OpenAI-compatible REST API at api.groq.com/openai/v1.
# CrewAI's native OpenAI provider is used here with base_url overridden to
# point at Groq — no LiteLLM package required. The model string uses the
# "openai/" prefix so CrewAI routes through its built-in OpenAI client.

llm = LLM(
    model=config.groq_model,                    # e.g. llama-3.3-70b-versatile
    provider='openai',                          # Explicit provider bypasses model validation;
                                                # CrewAI uses its native OpenAI client routed
                                                # to Groq's OpenAI-compatible endpoint.
    base_url='https://api.groq.com/openai/v1',  # Groq's OpenAI-compatible REST endpoint
    api_key=config.groq_api_key,               # Groq key — passed directly, no env var needed
    temperature=config.temperature,             # Low (0.2) for deterministic decisions
    max_tokens=config.max_tokens,               # 2048 — sufficient for structured JSON
)


# ── Agent Factories ───────────────────────────────────────────────────────────
# Each function returns a fresh Agent instance. Factory pattern (rather than
# module-level singletons) allows crew.py to reconstruct agents between runs
# while still sharing the same underlying LLM client.

def create_bull_agent() -> Agent:
    """
    Optimistic analyst focused on long entry opportunities.

    The agent is explicitly instructed to classify each opportunity by hold
    period (intraday / swing / position) so the risk manager has a concrete
    timeframe recommendation to evaluate — not just a directional view.
    """
    return Agent(
        role='Intraday Momentum Analyst',
        goal=(
            'Identify intraday long opportunities driven by momentum, volume spikes, '
            'and short-term technical breakouts. All recommendations are same-day trades.'
        ),
        backstory=(
            'You are an intraday momentum specialist with a tape-reading background. '
            'You focus exclusively on same-day price action — volume surges, VWAP reclaims, '
            'breakouts from intraday consolidation, and news-driven momentum. '
            'You never hold overnight and size into high-conviction setups with tight stops.'
        ),
        llm=llm,
        verbose=True,
    )


def create_bear_agent() -> Agent:
    """
    Risk-focused analyst specialising in downside scenarios and short setups.

    Framing the goal around timeframe classification ensures the bear agent
    produces structured risk assessments the risk manager can directly compare
    against the bull agent's output.
    """
    return Agent(
        role='Intraday Short Specialist',
        goal=(
            'Identify intraday short opportunities and same-day reversal setups. '
            'Focus on breakdown signals, volume distribution, and momentum exhaustion '
            'that play out within the current trading session.'
        ),
        backstory=(
            'You are an intraday short-selling specialist. You look for stocks losing '
            'key intraday levels like VWAP and LOD, failed breakouts with high-volume '
            'rejection, and RSI exhaustion on short timeframes. '
            'You only trade same-day setups and cut losses immediately when wrong.'
        ),
        llm=llm,
        verbose=True,
    )


def create_risk_manager() -> Agent:
    """
    Senior arbitrator who synthesises bull and bear analyses into a final decision.

    Key responsibilities:
    - Weigh bull vs. bear arguments and resolve conflicts
    - Assign the final hold period (intraday / swing / position) based on
      signal quality rather than just direction
    - Enforce the 0.75 confidence threshold as a hard gate before approving execution
    """
    return Agent(
        role='Intraday Risk Manager',
        goal=(
            'Evaluate intraday bull and bear arguments and make the final same-day '
            'trade decision. Enforce tight stops, quick exits, and never allow '
            f'positions to carry overnight. Only approve trades with confidence above {config.confidence_threshold}.'
        ),
        backstory=(
            'You are a senior intraday risk manager specializing in day trading discipline. '
            'All trades you approve are intraday only — hold_period is always intraday, '
            'max_hold_days is always 1. You enforce tight stop losses (1.5%) and quick '
            'take profits (2.5%) to lock in gains before momentum fades. '
            'You cut losing trades fast and never let a winner turn into a loser.'
        ),
        llm=llm,
        verbose=True,
    )


def create_portfolio_manager() -> Agent:
    """
    Portfolio-level oversight agent that evaluates decisions in the context
    of existing holdings and hold period distribution.

    Prevents the portfolio from becoming over-concentrated in a single hold
    period tier (e.g. all intraday positions on a volatile day) which would
    distort the risk profile relative to the strategy's design.
    """
    return Agent(
        role='Portfolio Manager',
        goal=(
            'Ensure portfolio balance across positions and hold periods. '
            'Avoid over-concentration in any single hold period category.'
        ),
        backstory=(
            'You manage overall portfolio health and ensure a balanced mix '
            'of intraday, swing, and position trades.'
        ),
        llm=llm,
        verbose=True,
    )
