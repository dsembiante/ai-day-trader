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
        vwap = price_above_vwap = atr_pct = None
        opening_range_high = opening_range_low = None
        orb_breakout_up = orb_breakout_down = None
        gap_pct = gap_is_bullish = gap_is_bearish = None
        volume_ratio = volume_confirmed = None

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

        # ── 5. Intraday Indicators ────────────────────────────────────────────
        current_price = price or 0.0
        current_volume = volume or 0
        vwap, price_above_vwap                               = self.get_vwap(ticker)
        opening_range_high, opening_range_low, _, orb_breakout_up, orb_breakout_down = self.get_opening_range(ticker)
        gap_pct, gap_is_bullish, gap_is_bearish              = self.get_premarket_gap(ticker)
        volume_ratio, volume_confirmed                       = self.get_volume_confirmation(ticker)
        atr_pct                                              = self.get_atr(ticker, current_price)

        # ── Assemble & Return ─────────────────────────────────────────────────
        return MarketData(
            ticker=ticker,
            current_price=current_price,
            volume=current_volume,
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
            vwap=vwap,
            price_above_vwap=price_above_vwap,
            atr_pct=atr_pct,
            opening_range_high=opening_range_high,
            opening_range_low=opening_range_low,
            orb_breakout_up=orb_breakout_up,
            orb_breakout_down=orb_breakout_down,
            gap_pct=gap_pct,
            gap_is_bullish=gap_is_bullish,
            gap_is_bearish=gap_is_bearish,
            volume_ratio=volume_ratio,
            volume_confirmed=volume_confirmed,
            data_sources_used=status,
        )

    def get_vwap(self, ticker: str) -> tuple:
        """
        Calculate today's VWAP from 1-minute intraday bars using close * volume.
        Returns (vwap, price_above_vwap) or (None, None) on failure.
        """
        try:
            df = yf.Ticker(ticker).history(period='1d', interval='1m')
            if df.empty:
                return None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            vwap_series = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            vwap_val = float(vwap_series.iloc[-1])
            current_price = float(df['Close'].iloc[-1])
            return vwap_val, current_price > vwap_val
        except Exception as e:
            log_error('vwap', ticker, str(e))
            return None, None

    def get_opening_range(self, ticker: str) -> tuple:
        """
        Calculate the opening range (9:30–10:00 AM EST) from 1-minute bars.
        Returns (orh, orl, orm, orb_breakout_up, orb_breakout_down) or all Nones.
        """
        try:
            df = yf.Ticker(ticker).history(period='1d', interval='1m')
            if df.empty:
                return None, None, None, None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Ensure index is timezone-aware in EST
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')
            open_bars = df.between_time('09:30', '09:59')
            if open_bars.empty:
                return None, None, None, None, None
            orh = float(open_bars['High'].max())
            orl = float(open_bars['Low'].min())
            orm = (orh + orl) / 2
            current_price = float(df['Close'].iloc[-1])
            return orh, orl, orm, current_price > orh, current_price < orl
        except Exception as e:
            log_error('opening_range', ticker, str(e))
            return None, None, None, None, None

    def get_premarket_gap(self, ticker: str) -> tuple:
        """
        Calculate pre-market gap using fast_info (last_price vs previous_close).
        Returns (gap_pct, gap_is_bullish, gap_is_bearish) or (None, None, None).
        """
        try:
            fi = yf.Ticker(ticker).fast_info
            last_price = fi.last_price
            prev_close = fi.previous_close
            if not last_price or not prev_close or prev_close == 0:
                return None, None, None
            gap_pct = float((last_price - prev_close) / prev_close * 100)
            return gap_pct, gap_pct > 0.5, gap_pct < -0.5
        except Exception as e:
            log_error('premarket_gap', ticker, str(e))
            return None, None, None

    def get_volume_confirmation(self, ticker: str) -> tuple:
        """
        Compare today's volume against the 20-day average.
        Returns (volume_ratio, volume_confirmed) or (None, None).
        volume_confirmed is True when volume_ratio > 1.20.
        """
        try:
            df = yf.Ticker(ticker).history(period='25d', interval='1d')
            if df.empty or len(df) < 21:
                return None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            avg_volume = float(df['Volume'].iloc[-21:-1].mean())  # 20-day avg, excluding today
            today_volume = float(df['Volume'].iloc[-1])
            if avg_volume == 0:
                return None, None
            volume_ratio = today_volume / avg_volume
            return float(volume_ratio), volume_ratio > 1.20
        except Exception as e:
            log_error('volume_confirmation', ticker, str(e))
            return None, None

    def get_atr(self, ticker: str, current_price: float) -> 'Optional[float]':
        """
        Calculate 14-day Average True Range as a percentage of current price.
        ATR% = ATR / current_price * 100
        """
        try:
            df = yf.download(ticker, period='30d', interval='1d', progress=False, auto_adjust=True)
            if df.empty or len(df) < 15:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            prev_close = df['Close'].shift(1)
            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - prev_close).abs(),
                (df['Low'] - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            if pd.isna(atr) or current_price == 0:
                return None
            return float(atr / current_price * 100)
        except Exception as e:
            log_error('atr', ticker, str(e))
            return None

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
