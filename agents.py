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
            'You are an intraday momentum specialist who only enters trades with confluence '
            'of multiple confirming signals — you never trade on a single signal alone. '
            'Your four-signal checklist: (1) price above VWAP confirms intraday uptrend, '
            '(2) opening range breakout above the 9:30–10:00 AM high confirms bullish momentum, '
            '(3) positive pre-market gap above +0.5% adds directional bias from overnight buyers, '
            '(4) volume ratio above 1.20x confirms institutional participation. '
            'You require at least 2 of these 4 signals before recommending a buy. '
            'You specialize in two windows: the first hour after the opening range forms '
            '(10:00–11:00 AM) and the final hour before close (2:30–3:50 PM). '
            'You never hold overnight and always report your signal count in key_factors.'
        ),
        llm=llm,
        verbose=False,
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
            'You are an intraday short-selling specialist who identifies high-probability '
            'short setups using the same multi-signal discipline as your bull counterpart — '
            'you never short on a single signal alone. '
            'Your four-signal bearish checklist: (1) price below VWAP confirms intraday '
            'downtrend, (2) opening range breakdown below the 9:30–10:00 AM low confirms '
            'bearish momentum, (3) negative pre-market gap below -0.5% shows overnight '
            'selling pressure, (4) volume ratio above 1.20x on a declining stock confirms '
            'institutional distribution. '
            'You require at least 2 of these 4 signals before recommending a short. '
            'You cut losses immediately when the short thesis is invalidated, never carry '
            'positions overnight, and always report your bearish signal count in key_factors.'
        ),
        llm=llm,
        verbose=False,
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
            'You are a senior intraday risk specialist who enforces the two-signal minimum '
            'rule as your first and most important gate — if the analyst identified fewer '
            'than 2 confirming signals (shown as X/4 in key_factors), you reject the trade '
            'immediately regardless of other factors. '
            'Every trade you approve has hold_period=intraday and max_hold_days=1 with no '
            'exceptions. You set tight stop losses appropriate for intraday moves (1.5%) '
            'and size all positions for same-day closure with a hard force-close at '
            '3:50 PM EST. You do not approve trades before 10:00 AM EST — the opening '
            'range must fully form before any position is entered. '
            'When signals conflict or conviction is low, you sit out. '
            'Capital preservation is your primary objective.'
        ),
        llm=llm,
        verbose=False,
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
            'Oversee intraday portfolio exposure — enforce the 5-position cap, '
            'verify multi-signal confirmation on every trade, and ensure total '
            'intraday risk stays within acceptable limits.'
        ),
        backstory=(
            'You are an intraday portfolio overseer responsible for keeping the '
            'day trading book disciplined and within risk limits. '
            'You enforce a hard cap of 5 simultaneous intraday positions — beyond '
            'that, no new trades are approved regardless of signal quality. '
            'Before approving any trade, you check that the risk manager confirmed '
            'multi-signal backing (at least 2/4 signals). If the reasoning only '
            'mentions a single signal type, you flag it as insufficient confirmation '
            'and set execute=false. You monitor overall intraday exposure and ensure '
            'the portfolio is not over-concentrated in correlated positions '
            '(e.g. multiple tech names moving together).'
        ),
        llm=llm,
        verbose=False,
    )
