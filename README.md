# AI Trading Agent

An autonomous, multi-agent trading system built with CrewAI, Groq LLM, and Alpaca Markets. Four specialised AI agents collaborate on every analysis cycle to produce structured, risk-gated trade decisions across intraday, swing, and position holding periods.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        scheduler.py                             │
│          Drives the trading loop (fixed_6x / intraday_30min)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          crew.py                                │
│                   Per-ticker orchestration                      │
│                                                                 │
│  data_collector.py → [MarketData] → CrewAI Crew                │
│                                          │                      │
│                          ┌───────────────┼───────────────┐      │
│                          ▼               ▼               ▼      │
│                     Bull Agent      Bear Agent     (parallel)   │
│                          └───────────────┴───────────────┘      │
│                                          │                      │
│                                          ▼                      │
│                                    Risk Manager                 │
│                                          │                      │
│                                          ▼                      │
│                                  Portfolio Manager              │
│                                          │                      │
│                          [TradeDecision + position sizing]      │
│                                          │                      │
│                     trade_executor.py → Alpaca API              │
│                     database.py + logger.py → persistence       │
└─────────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          app.py          circuit_breaker  position_monitor
      (Streamlit UI)      (10% hard stop)  (hold period exits)
```

### Agent Collaboration Model

Each ticker analysis runs a sequential 4-agent crew:

| Agent | Role | Output |
|---|---|---|
| **Bull Analyst** | Identifies long opportunities, classifies hold period | `AgentAnalysis` |
| **Bear Analyst** | Surfaces downside risks and short setups | `AgentAnalysis` |
| **Risk Manager** | Arbitrates bull/bear debate, sets final hold period, gates at ≥0.75 confidence | `TradeDecision` |
| **Portfolio Manager** | Validates against max positions, duplicate detection, sector concentration | `TradeDecision` |

---

## Hold Period System

The system classifies every trade into one of three holding tiers at the point of decision. Each tier has independent risk parameters enforced by `position_sizer.py` and `position_monitor.py`:

| Tier | Max Hold | Stop Loss | Take Profit | Use Case |
|---|---|---|---|---|
| **Intraday** | 1 day | 2% | 4% | News catalysts, strong short-term momentum |
| **Swing** | 5 days | 5% | 10% | Multi-day technical setups, moderate conviction |
| **Position** | 20 days | 8% | 20% | High-conviction fundamental plays |

All three tiers maintain a **2:1 reward-to-risk ratio** by design.

---

## Data Sources

| Source | Data | Tier Required |
|---|---|---|
| **Alpaca** | OHLCV bars, account state, order placement | Free |
| **Finnhub** | News headlines | Free |
| **Finnhub** | Sentiment score | Paid |
| **Alpha Vantage** | Company fundamentals (cached daily) | Free (25 req/day) |
| **FRED** | Fed Funds Rate, CPI inflation (cached daily) | Free |
| **Groq** | LLM inference (llama-3.3-70b-versatile) | Free tier available |

Data collection is designed for graceful degradation — each source is wrapped in an independent try/except. The `DataSourceStatus` model tracks source availability and is stored with every trade record for post-analysis.

---

## Default Watchlist

The system analyses 13 symbols on every cycle, covering three market segments:

| Segment | Tickers |
|---|---|
| **Technology** | AAPL, MSFT, GOOGL, AMZN, NVDA |
| **Financials** | JPM, BAC, GS, MS, V |
| **ETFs / Market Proxies** | SPY, QQQ, IWM |

The watchlist is configurable via the `WATCHLIST` field in `config.py`. ETFs provide broad market context that informs the macro framing of individual stock decisions.

---

## Risk Controls

Three independent layers of risk management operate in the system:

**1. Confidence Threshold (Agent Layer)**
The Risk Manager agent only approves trades with a confidence score ≥ 0.75. This is enforced at both the agent prompt level and as a Pydantic field validator on `TradeDecision`.

**2. Position Sizing (Trade Layer)**
Maximum 2% of portfolio per position. Size scales with confidence (0.75→50% of budget, 1.0→100%) and hold period tier (intraday: 0.7×, swing: 1.0×, position: 1.3×). Hard-capped at 2.6% regardless of inputs.

**3. Circuit Breaker (Portfolio Layer)**
Monitors rolling drawdown from the portfolio's all-time peak. At ≥10% drawdown, all trading halts immediately and the event is logged, emailed, and flagged in the dashboard. **No auto-liquidation** — resumption requires manual review and deletion of `data/peak_value.json`.

---

## Project Structure

```
ai-trading-agent/
├── app.py                  # Streamlit dashboard (8 tabs)
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
├── position_monitor.py     # Hold period enforcement
├── database.py             # SQLite trade journal
├── report_generator.py     # PDF report generation
├── notifier.py             # Email alerts
├── backtester.py           # RSI-based strategy validation
├── logger.py               # Structured observability
├── data/
│   ├── trading.db          # SQLite database
│   └── cache/              # Daily API response cache
├── reports/
│   ├── daily/
│   ├── weekly/
│   └── monthly/
├── logs/
│   ├── trade_journal.json
│   ├── run_logs/
│   └── errors.log
├── .env                    # API keys and runtime settings
├── .gitignore
└── requirements.txt
```

---

## Required Accounts

All services below have free tiers sufficient to run this project in paper trading mode.

### Alpaca Markets — Brokerage & Market Data
- Sign up at [alpaca.markets](https://alpaca.markets)
- Create a **Paper Trading** account (no real money required)
- Navigate to **Paper Trading → API Keys** to generate your `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
- Keep `ALPACA_BASE_URL=https://paper-api.alpaca.markets` until you are ready for live trading

### Groq — LLM Inference
- Sign up at [console.groq.com](https://console.groq.com)
- Navigate to **API Keys** and create a new key
- Free tier is sufficient — the system uses `llama-3.3-70b-versatile`

### Finnhub — News Headlines
- Sign up at [finnhub.io](https://finnhub.io)
- Your API key is displayed on the dashboard immediately after registration
- Free tier provides news headlines; sentiment scoring requires a paid plan (optional)

### yfinance — Technical Indicators & Fundamentals
- No API key required — yfinance fetches data from Yahoo Finance publicly
- Provides RSI, MACD, moving averages, P/E ratio, EPS, revenue growth, earnings dates, and analyst recommendations

### FRED — Macro Economic Data
- Sign up at [fred.stlouisfed.org](https://fredaccount.stlouisfed.org/login/secure/)
- Navigate to **My Account → API Keys** to request a free key
- Provides Fed Funds Rate and CPI inflation data used in agent macro context

### Railway — Cloud Deployment (Optional)
- Sign up at [railway.app](https://railway.app)
- Connect your GitHub account and create a new project from your repository
- Set all `.env` variables as Railway environment variables
- Set start command to `python scheduler.py` and `TZ=America/New_York`
- Free tier available; the scheduler runs continuously so a paid hobby plan (~$5/month) is recommended for production use

### Gmail — Email Alerts (Optional)
- Use an existing Gmail account
- Enable 2-factor authentication
- Generate an **App Password** at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
- Use the App Password (not your regular password) for `ALERT_EMAIL_PASSWORD`

---

## Setup

### Prerequisites
- Python 3.10+
- Accounts set up as described in the Required Accounts section above

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-agent.git
cd ai-trading-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy and populate the `.env` file:

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

# Trading Mode — DO NOT change to live until 6+ months paper trading
TRADING_MODE=paper

# Run Mode
RUN_MODE=fixed_6x          # or: intraday_30min

# Alerts (optional)
ALERT_EMAIL=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_gmail_app_password
```

### Validate Setup

```bash
# Test data collection
python -c "from data_collector import DataCollector; dc = DataCollector(); print(dc.collect('AAPL'))"

# Run historical backtest before going live
python backtester.py

# Launch dashboard
streamlit run app.py

# Start trading scheduler
python scheduler.py
```

---

## Run Modes

### `fixed_6x` (Recommended for beginners)
Six analysis cycles per trading day at fixed times:

| Time | Action |
|---|---|
| 09:30 AM | Market open cycle |
| 11:00 AM | Mid-morning cycle |
| 01:00 PM | Post-lunch cycle |
| 02:30 PM | Early afternoon cycle |
| 03:45 PM | Pre-close cycle + intraday flush |
| 04:00 PM | End-of-day report |

### `intraday_30min`
Analysis cycle every 30 minutes from 9:30 AM to 3:30 PM, plus the standard 3:45 PM and 4:00 PM jobs.

---

## Dashboard

Run `streamlit run app.py` to access the 8-tab dashboard:

| Tab | Contents |
|---|---|
| **Overview** | Total P&L, win rate, trade count, profit factor |
| **Performance vs S&P 500** | Portfolio equity curve |
| **Active Positions** | Open trades with entry, stop, and target prices |
| **Trade History** | Filterable full log with agent reasoning drill-down |
| **Hold Period Analysis** | P&L breakdown by intraday / swing / position tier |
| **Risk Monitor** | Circuit breaker status, risk parameter summary |
| **Reports** | On-demand PDF generation and download |
| **Settings** | Read-only view of active configuration |

---

## Deployment (Railway)

1. Push repository to GitHub
2. Create a new Railway project from the GitHub repo
3. Set all `.env` variables as Railway environment variables
4. Set the start command to: `python scheduler.py`
5. Set `TZ=America/New_York` in Railway environment variables to ensure market hours are evaluated correctly

---

## Pre-Live Checklist

Before switching `TRADING_MODE=live`, verify all of the following:

- [ ] Backtest win rate ≥ 50% and Sharpe ratio ≥ 0.5 (`python backtester.py`)
- [ ] Minimum 6 months of paper trading with consistent results
- [ ] Circuit breaker has been tested and confirmed functional
- [ ] Email alerts are configured and tested
- [ ] All 4 data sources returning valid data
- [ ] Dashboard displaying accurate open positions and P&L
- [ ] `ALPACA_BASE_URL` updated to `https://api.alpaca.markets`

---

## Backtesting

Before deploying to paper trading, run the built-in backtester to validate the strategy has directional edge on your watchlist:

```bash
python backtester.py
```

The backtester uses a simplified RSI mean-reversion strategy as a proxy (not the full LLM crew, which cannot be replayed historically). It fetches 180 days of Alpaca OHLCV data and simulates entries on RSI < 30 with price above SMA-50, exiting on stop-loss (-5%), take-profit (+10%), or RSI > 70.

**Acceptance thresholds:**
- Win rate ≥ 50%
- Sharpe ratio ≥ 0.5 (annualised)

Results below these thresholds suggest the underlying technicals lack sufficient edge on the current watchlist before the LLM layer is applied.

---

## Reporting

Three PDF report types are generated automatically and available for download from the dashboard:

| Report | Schedule | Location |
|---|---|---|
| **Daily** | Generated at 4:00 PM by the scheduler | `reports/daily/daily_YYYY-MM-DD.pdf` |
| **Weekly** | On-demand or schedulable | `reports/weekly/weekly_YYYY-W##.pdf` |
| **Monthly** | On-demand or schedulable | `reports/monthly/monthly_YYYY-MM.pdf` |

Each report contains: total P&L, win rate, open position count, P&L breakdown by hold period tier, and a trade table for the period.

---

## Logging

Three independent logging surfaces provide full observability:

| File | Contents | Used By |
|---|---|---|
| `logs/errors.log` | Timestamped error entries from all components | Dashboard Risk tab, alerts |
| `logs/run_logs/run_*.json` | Full cycle summary per scheduler run (tickers, trades, duration, API status) | Post-run diagnostics |
| `logs/trade_journal.json` | Append-only JSON array of every trade event | Human-readable audit trail outside SQLite |

---

## Email Alerts

Configure `ALERT_EMAIL` and `ALERT_EMAIL_PASSWORD` in `.env` to receive alerts for:

| Event | Trigger |
|---|---|
| **Circuit Breaker** | Portfolio drawdown reaches 10% — immediate alert, trading halted |
| **Trade Placed** | Every successful order submission |
| **Daily Summary** | End-of-day P&L and trade count at 4:00 PM |
| **API Failure** | Any data source becomes unreachable during collection |

Gmail App Passwords are required (not your regular Gmail password). Generate one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).

---

## Key Design Decisions

**Why SQLite?**
No external infrastructure required. The system is designed to run on a single Railway dyno or local machine. SQLite serialises writes internally, making it safe for the Streamlit and scheduler threads to share a connection with `check_same_thread=False`.

**Why Groq over OpenAI?**
Groq provides significantly lower inference latency on llama-3.3-70b-versatile at a fraction of the cost. Fast inference is important for intraday mode where 13 tickers must be analysed within a 30-minute window.

**Why bracket orders?**
Stop-loss and take-profit legs are submitted atomically with the entry order via Alpaca's bracket order type. This eliminates the race condition where a fill occurs before protective orders are placed.

**Why no auto-liquidation on circuit breaker?**
Forced liquidation during a drawdown event can lock in losses that would otherwise recover. The circuit breaker halts new entries and alerts the operator — position management during a drawdown event is a human decision.

---

## Dependencies

| Package | Purpose |
|---|---|
| `crewai` | Multi-agent orchestration |
| `langchain-groq` | Groq LLM integration for CrewAI |
| `groq` | Base Groq SDK |
| `alpaca-py` | Market data and order execution |
| `finnhub-python` | News headlines |
| `yfinance` | Technical indicators & company fundamentals |
| `fredapi` | Macro economic data |
| `pandas` + `pandas-ta` | OHLCV processing and technical indicators |
| `pydantic` | Data validation and schema enforcement |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `reportlab` | PDF report generation |
| `sqlalchemy` | Database ORM utilities |
| `schedule` | Lightweight job scheduler |
| `python-dotenv` | Environment variable loading |

