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

from crewai import Agent
from langchain_groq import ChatGroq
from config import config


# ── LLM Setup ────────────────────────────────────────────────────────────────

def create_llm_with_retry() -> ChatGroq:
    """
    Initialise the Groq LLM client with retry settings from config.

    Retry behaviour (max_retries, retry_delay) is handled by the LangChain
    ChatGroq wrapper rather than implemented manually, so transient rate-limits
    and network blips are recovered transparently without crashing the crew.

    Returns:
        A configured ChatGroq instance ready for agent assignment.
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=config.groq_model,            # e.g. llama-3.3-70b-versatile
        api_key=config.groq_api_key,
        temperature=config.temperature,     # Low (0.2) for deterministic decisions
        max_tokens=config.max_tokens,       # 2048 — sufficient for structured JSON output
        max_retries=config.groq_max_retries,
    )


# Shared LLM instance — created once at module load and reused across all agents
# to avoid redundant client initialisation on every scheduler cycle.
llm = create_llm_with_retry()


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
        role='Bull Market Analyst',
        goal=(
            'Identify compelling buying opportunities. For each recommendation '
            'specify whether this is an intraday, swing, or position trade '
            'based on signal strength and timeframe.'
        ),
        backstory=(
            'You are an optimistic equity analyst who looks for catalysts, '
            'momentum, and favorable setups. You classify each opportunity '
            'by how long the thesis should play out.'
        ),
        llm=llm,
        verbose=True,  # Surfaces reasoning chain in logs for auditability
    )


def create_bear_agent() -> Agent:
    """
    Risk-focused analyst specialising in downside scenarios and short setups.

    Framing the goal around timeframe classification ensures the bear agent
    produces structured risk assessments the risk manager can directly compare
    against the bull agent's output.
    """
    return Agent(
        role='Bear Market Analyst',
        goal=(
            'Identify risks and short opportunities. Specify the timeframe '
            'of each risk — is this an intraday concern, multi-day deterioration, '
            'or longer-term structural issue?'
        ),
        backstory=(
            'You are a risk-focused analyst who specializes in identifying '
            'overvalued stocks and downside scenarios across all timeframes.'
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
        role='Risk Manager',
        goal=(
            'Evaluate bull and bear arguments, classify the final hold period '
            '(intraday/swing/position), and make the trade decision with strict '
            'risk management. Only approve trades with confidence above 0.75.'
        ),
        backstory=(
            'You are a senior risk manager. You determine the appropriate '
            'holding period based on signal quality: strong short-term signals '
            'become intraday trades, solid multi-day setups become swing trades, '
            'and high-conviction fundamental plays become position trades.'
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
