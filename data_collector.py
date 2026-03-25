"""
data_collector.py — Aggregates market data from all external sources.

Each source is wrapped in an independent try/except block so a single API
outage or rate-limit does not abort the collection cycle. Failures are
recorded in DataSourceStatus and logged via logger.py; downstream agents
receive the best available data and adjust their confidence accordingly.

Sources:
    1. Alpaca      — Current price and volume
    2. yfinance    — Technical indicators (RSI, MACD, MA50, MA200) and
                     fundamentals (P/E, forward P/E, EPS, revenue growth,
                     next earnings date, analyst recommendation)
    3. Finnhub     — Recent news headlines + company sentiment score
    4. FRED        — Macro series: Fed Funds Rate + trailing CPI inflation

Usage:
    from data_collector import DataCollector
    data = DataCollector().collect('AAPL')
"""

import finnhub
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import json, os
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from fredapi import Fred
from models import MarketData, DataSourceStatus
from config import config
from logger import log_error


class DataCollector:
    """
    Stateful collector that holds authenticated API clients for the lifetime
    of a scheduler run. Instantiate once per cycle and call collect() per ticker.
    """

    def __init__(self):
        # Alpaca historical client — used for current price and volume
        self.alpaca = StockHistoricalDataClient(
            config.alpaca_api_key, config.alpaca_secret_key)

        # Finnhub client — news and sentiment
        self.finnhub = finnhub.Client(api_key=config.finnhub_api_key)

        # FRED client — macro economic series
        self.fred = Fred(api_key=config.fred_api_key)

        # Ensure cache directory exists before any source tries to write to it
        os.makedirs(config.cache_dir, exist_ok=True)

    def collect(self, ticker: str) -> MarketData:
        """
        Fetch and aggregate all available signals for a single ticker.

        Returns a MarketData object populated with whatever data was reachable.
        Fields are None when their source was unavailable — agents must handle
        this via reduced-signal analysis rather than raising exceptions.
        """
        status = DataSourceStatus()

        price = volume = rsi = macd = ma50 = ma200 = None
        pe_ratio = forward_pe = revenue_growth = eps = None
        next_earnings_date = analyst_recommendation = None
        news_sentiment = None
        headlines = []
        macro_context = None

        # ── 1. Alpaca — Current Price & Volume ────────────────────────────────
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=5)
            ))
            df = bars.df.reset_index()
            price = float(df['close'].iloc[-1])
            volume = int(df['volume'].iloc[-1])

        except Exception as e:
            status.alpaca = False
            log_error('alpaca', ticker, str(e))

        # ── 2. yfinance — Technicals & Fundamentals ───────────────────────────
        # Pull 1 year of daily history for indicator calculation.
        # Fundamentals come from yf.Ticker.info — no separate API key required.
        try:
            yf_ticker = yf.Ticker(ticker)

            # ── Technical indicators ──────────────────────────────────────────
            hist = yf_ticker.history(period='1y')
            if not hist.empty:
                close = hist['Close']

                rsi_series = ta.rsi(close, length=14)
                if rsi_series is not None and not rsi_series.empty:
                    val = rsi_series.iloc[-1]
                    rsi = float(val) if not pd.isna(val) else None

                macd_df = ta.macd(close)
                if macd_df is not None and 'MACD_12_26_9' in macd_df.columns:
                    val = macd_df['MACD_12_26_9'].iloc[-1]
                    macd = float(val) if not pd.isna(val) else None

                ma50_series = close.rolling(50).mean()
                ma200_series = close.rolling(200).mean()
                val50 = ma50_series.iloc[-1]
                val200 = ma200_series.iloc[-1]
                ma50 = float(val50) if not pd.isna(val50) else None
                ma200 = float(val200) if not pd.isna(val200) else None

            # ── Fundamentals ──────────────────────────────────────────────────
            info = yf_ticker.info
            pe_ratio             = info.get('trailingPE', None)
            forward_pe           = info.get('forwardPE', None)
            revenue_growth       = info.get('revenueGrowth', None)   # decimal e.g. 0.12
            eps                  = info.get('trailingEps', None)
            analyst_recommendation = info.get('recommendationKey', None)  # 'buy','hold','sell'

            # ── Next earnings date ────────────────────────────────────────────
            try:
                cal = yf_ticker.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    dates = cal['Earnings Date']
                    if dates:
                        next_earnings_date = str(dates[0].date())
            except Exception:
                pass  # Earnings date is best-effort

        except Exception as e:
            status.yfinance = False
            log_error('yfinance', ticker, str(e))

        # ── 3. Finnhub — News Headlines & Sentiment ───────────────────────────
        try:
            news = self.finnhub.company_news(
                ticker,
                _from=(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d')
            )
            if news:
                headlines = [n['headline'] for n in news[:5]]
        except Exception as e:
            status.finnhub = False
            log_error('finnhub_news', ticker, str(e))

        try:
            sd = self.finnhub.news_sentiment(ticker)
            news_sentiment = sd.get('companyNewsScore', None)
        except Exception as e:
            log_error('finnhub_sentiment', ticker, str(e))

        # ── 4. FRED — Macro Economic Context ─────────────────────────────────
        try:
            macro_cache = f"{config.cache_dir}/macro_{datetime.now().strftime('%Y%m%d')}.json"
            if os.path.exists(macro_cache):
                with open(macro_cache) as f:
                    macro = json.load(f)
            else:
                fed_rate  = self.fred.get_series('FEDFUNDS').iloc[-1]
                inflation = self.fred.get_series('CPIAUCSL').pct_change(12, fill_method=None).iloc[-1] * 100
                macro = {'fed_rate': float(fed_rate), 'inflation': float(inflation)}
                with open(macro_cache, 'w') as f:
                    json.dump(macro, f)

            macro_context = (
                f"Fed Rate: {macro['fed_rate']:.2f}%, "
                f"Inflation: {macro['inflation']:.2f}%"
            )

        except Exception as e:
            status.fred = False
            log_error('fred', ticker, str(e))

        # ── Assemble & Return ─────────────────────────────────────────────────
        return MarketData(
            ticker=ticker,
            current_price=price or 0.0,
            volume=volume or 0,
            rsi=rsi,
            macd=macd,
            moving_avg_50=ma50,
            moving_avg_200=ma200,
            pe_ratio=pe_ratio,
            forward_pe=forward_pe,
            revenue_growth=revenue_growth,
            eps=eps,
            next_earnings_date=next_earnings_date,
            analyst_recommendation=analyst_recommendation,
            news_sentiment=news_sentiment,
            news_headlines=headlines,
            macro_context=macro_context,
            data_sources_used=status,
        )

    def get_market_regime(self) -> str:
        """
        Determine the current broad market regime using SPY price vs. moving averages.

        Uses the classic golden cross / death cross framework:
            Bull:     Price > SMA-50 > SMA-200 — uptrend confirmed on both timeframes
            Bear:     Price < SMA-50 < SMA-200 — downtrend confirmed on both timeframes
            Sideways: Neither condition met — mixed or transitioning market

        Returns:
            'bull', 'bear', or 'sideways'
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols='SPY',
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=300),
            ))
            df = bars.df.reset_index()
            df['SMA_50']  = df['close'].rolling(50).mean()
            df['SMA_200'] = df['close'].rolling(200).mean()

            current_price = float(df['close'].iloc[-1])
            sma_50        = float(df['SMA_50'].iloc[-1])
            sma_200       = float(df['SMA_200'].iloc[-1])

            if current_price > sma_50 and sma_50 > sma_200:
                return 'bull'
            elif current_price < sma_50 and sma_50 < sma_200:
                return 'bear'
            else:
                return 'sideways'

        except Exception as e:
            log_error('market_regime', 'SPY', str(e))
            return 'sideways'
