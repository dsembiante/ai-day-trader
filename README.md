# AI Day Trading Agent

An autonomous, multi-agent day trading system built with **CrewAI**, **Groq LLM**, and **Alpaca Markets**. Four specialist AI agents collaborate every 10 minutes during market hours to identify intraday long and short opportunities, enforce tight risk controls, and force-close all positions before market close.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Multi-Agent Orchestration** | CrewAI |
| **LLM Inference** | Groq (llama-3.3-70b-versatile) |
| **Brokerage & Order Execution** | Alpaca Markets |
| **Market Data** | yfinance (Yahoo Finance) |
| **News & Sentiment** | Finnhub |
| **Macro Data** | FRED (Federal Reserve) |
| **Dashboard** | Streamlit |
| **Deployment** | Railway |

---

## How It Works

The scheduler fires every **10 minutes from 9:30 AM to 3:40 PM EST** on weekdays. At **3:50 PM**, all open intraday positions are force-closed before market close. An end-of-day PDF report is generated at 4:00 PM.

Each cycle runs a 4-agent CrewAI crew for every ticker on the watchlist:

```
┌─────────────────────────────────────────────────────────────────┐
│                        scheduler.py                             │
│          Fires every 10 minutes — 9:30 AM to 3:40 PM EST       │
│          Force-closes all intraday positions at 3:50 PM         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          crew.py                                │
│                   Per-ticker orchestration                      │
│                                                                 │
│  data_collector.py → [MarketData] → CrewAI Crew                │
│                                          │                      │
│                          ┌───────────────┴───────────────┐      │
│                          ▼               ▼               │      │
│               Intraday Momentum    Intraday Short         │      │
│                  Analyst           Specialist    (parallel)│     │
│                          └───────────────┬───────────────┘      │
│                                          ▼                      │
│                               Intraday Risk Manager             │
│                                          │                      │
│                                          ▼                      │
│                                  Portfolio Manager              │
│                                          │                      │
│                          [TradeDecision + position sizing]      │
│                                          │                      │
│                     trade_executor.py → Alpaca API              │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Specialization | Output |
|---|---|---|
| **Intraday Momentum Analyst** | Volume spikes, VWAP breakouts, short-term momentum | `AgentAnalysis` |
| **Intraday Short Specialist** | Breakdowns, LOD breaks, failed breakouts, RSI exhaustion | `AgentAnalysis` |
| **Intraday Risk Manager** | Arbitrates bull/bear, enforces tight stops, gates at ≥0.80 confidence | `TradeDecision` |
| **Portfolio Manager** | Validates position limits, duplicate detection, concentration | `TradeDecision` |

---

## Risk Controls

| Layer | Control |
|---|---|
| **Agent** | Confidence threshold ≥ 0.80 required to execute |
| **Position** | Max 2% of portfolio per trade; sized by confidence |
| **Intraday** | Stop loss 1.5%, take profit 2.5%; all positions force-closed at 3:50 PM |
| **Circuit Breaker** | Trading halts at 10% portfolio drawdown |

---

## Run Modes

| Mode | Schedule |
|---|---|
| `intraday_10min` | Every 10 min, 9:30 AM–3:40 PM + force-close 3:50 PM **(default)** |
| `intraday_30min` | Every 30 min, 9:30 AM–3:30 PM + flush 3:45 PM |
| `fixed_6x` | 6 fixed times per day |

---

## Project Structure

```
ai-day-trader/
├── app.py                  # Streamlit dashboard
├── scheduler.py            # Automated cycle runner (Railway)
├── crew.py                 # Per-ticker orchestration loop
├── agents.py               # CrewAI agent definitions
├── tasks.py                # Agent task prompts and schemas
├── models.py               # Pydantic data contracts
├── config.py               # Centralised settings (loaded from .env)
├── data_collector.py       # Multi-source data aggregation
├── position_sizer.py       # Risk-based position sizing
├── circuit_breaker.py      # Portfolio drawdown hard stop
├── trade_executor.py       # Alpaca order placement
├── position_monitor.py     # Hold period enforcement + force-close
├── database.py             # Trade journal (PostgreSQL)
├── report_generator.py     # PDF report generation
├── notifier.py             # Email alerts
├── backtester.py           # RSI-based strategy validation
├── logger.py               # Structured observability
├── .env                    # API keys and runtime settings
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- Alpaca Markets account (paper trading)
- Groq API key
- Finnhub API key
- FRED API key

### Installation

```bash
git clone https://github.com/dsembiante/ai-day-trader.git
cd ai-day-trader

python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
# Alpaca
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Data Sources
FINNHUB_API_KEY=your_key
FRED_API_KEY=your_key

# LLM
GROQ_API_KEY=your_key

# Trading Mode — keep paper until validated
TRADING_MODE=paper

# Run Mode
RUN_MODE=intraday_10min

# Alerts (optional)
ALERT_EMAIL=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_gmail_app_password
```

### Validate & Run

```bash
# Run historical backtest first
python backtester.py

# Launch dashboard
streamlit run app.py

# Start trading scheduler
python scheduler.py
```

---

## Required Accounts

| Service | Purpose | Cost |
|---|---|---|
| [Alpaca Markets](https://alpaca.markets) | Order execution + market data | Free (paper) |
| [Groq](https://console.groq.com) | LLM inference | Free tier |
| [Finnhub](https://finnhub.io) | News headlines | Free tier |
| [FRED](https://fred.stlouisfed.org) | Macro data | Free |
| [Railway](https://railway.app) | Cloud deployment | ~$5/month |

---

## Deployment (Railway)

1. Push repository to GitHub
2. Create a new Railway project from the repo
3. Set all `.env` variables as Railway environment variables
4. Set start command: `python scheduler.py`
5. Set `TZ=America/New_York` to ensure correct market hours

---

## Pre-Live Checklist

Before switching `TRADING_MODE=live`:

- [ ] Backtest win rate ≥ 50% and Sharpe ≥ 0.5 (`python backtester.py`)
- [ ] Minimum 30 days of paper trading with consistent results
- [ ] Circuit breaker tested and confirmed functional
- [ ] Email alerts configured and tested
- [ ] All data sources returning valid data
- [ ] Dashboard showing accurate positions and P&L
- [ ] `ALPACA_BASE_URL` updated to `https://api.alpaca.markets`

---

## Dependencies

| Package | Purpose |
|---|---|
| `crewai` | Multi-agent orchestration |
| `groq` + `langchain-groq` | LLM inference via Groq |
| `yfinance` | Market data, technicals, fundamentals |
| `finnhub-python` | News headlines |
| `fredapi` | Macro economic data |
| `pandas` + `pandas-ta` | OHLCV processing and indicators |
| `pydantic` | Data validation |
| `streamlit` + `plotly` | Dashboard |
| `reportlab` | PDF reports |
| `sqlalchemy` + `psycopg2-binary` | PostgreSQL database |
| `schedule` | Job scheduling |
| `python-dotenv` | Environment variable loading |
