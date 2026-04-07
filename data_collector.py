"""
data_collector.py — Aggregates market data from all external sources.

Each source is wrapped in an independent try/except block so a single API
outage or rate-limit does not abort the collection cycle. Failures are
recorded in DataSourceStatus and logged via logger.py; downstream agents
receive the best available data and adjust their confidence accordingly.

Sources:
    1. Alpaca      — Current price and volume, all intraday indicators
                     (VWAP, opening range, volume confirmation, ATR,
                     premarket gap) — used for all time-sensitive data
                     to avoid yfinance rate-limit bans on server IPs
    2. yfinance    — Technical indicators (RSI, MACD, MA50, MA200),
                     fundamentals (P/E, forward P/E, EPS, revenue growth,
                     next earnings date, analyst recommendation), and
                     recent news headlines. Results are cached per ticker
                     per day to reduce rate-limit exposure; one retry with
                     a 2-second sleep on 429/rate-limit errors.
    3. FRED        — Macro series: Fed Funds Rate + trailing CPI inflation

Usage:
    from data_collector import DataCollector
    data = DataCollector().collect('AAPL')
"""

import time
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import json, os
from datetime import datetime, timedelta
from typing import Optional
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
        # Alpaca historical client — used for current price, volume, and all
        # intraday indicators (avoids yfinance rate-limit bans on server IPs)
        self.alpaca = StockHistoricalDataClient(
            config.alpaca_api_key, config.alpaca_secret_key)

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
        vix = None
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
        # Results are cached per ticker per day. One retry with a 2-second sleep
        # on rate-limit errors so the source degrades gracefully rather than
        # immediately returning False. status.yfinance is only set False when
        # both attempts fail.
        yf_cache = f"{config.cache_dir}/yf_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
        _yf_loaded = False

        if os.path.exists(yf_cache):
            try:
                with open(yf_cache) as f:
                    cached = json.load(f)
                rsi                    = cached.get('rsi')
                macd                   = cached.get('macd')
                ma50                   = cached.get('ma50')
                ma200                  = cached.get('ma200')
                pe_ratio               = cached.get('pe_ratio')
                forward_pe             = cached.get('forward_pe')
                revenue_growth         = cached.get('revenue_growth')
                eps                    = cached.get('eps')
                analyst_recommendation = cached.get('analyst_recommendation')
                next_earnings_date     = cached.get('next_earnings_date')
                _yf_loaded = True
            except Exception:
                pass  # Fall through to live fetch

        if not _yf_loaded:
            for _attempt in range(2):
                try:
                    yf_ticker = yf.Ticker(ticker)

                    # ── Technical indicators ──────────────────────────────────
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

                    # ── Fundamentals ──────────────────────────────────────────
                    info = yf_ticker.info
                    pe_ratio               = info.get('trailingPE', None)
                    forward_pe             = info.get('forwardPE', None)
                    revenue_growth         = info.get('revenueGrowth', None)   # decimal e.g. 0.12
                    eps                    = info.get('trailingEps', None)
                    analyst_recommendation = info.get('recommendationKey', None)  # 'buy','hold','sell'

                    # ── Next earnings date ────────────────────────────────────
                    try:
                        cal = yf_ticker.calendar
                        if isinstance(cal, dict) and 'Earnings Date' in cal:
                            dates = cal['Earnings Date']
                            if dates:
                                next_earnings_date = str(dates[0].date())
                    except Exception:
                        pass  # Earnings date is best-effort

                    # ── Cache results for the rest of the day ─────────────────
                    try:
                        with open(yf_cache, 'w') as f:
                            json.dump({
                                'rsi': rsi, 'macd': macd, 'ma50': ma50, 'ma200': ma200,
                                'pe_ratio': pe_ratio, 'forward_pe': forward_pe,
                                'revenue_growth': revenue_growth, 'eps': eps,
                                'analyst_recommendation': analyst_recommendation,
                                'next_earnings_date': next_earnings_date,
                            }, f)
                    except Exception:
                        pass

                    break  # Success — exit retry loop

                except Exception as e:
                    err_str = str(e).lower()
                    if _attempt == 0 and ('rate' in err_str or '429' in err_str or 'too many' in err_str):
                        time.sleep(2)
                        continue
                    status.yfinance = False
                    log_error('yfinance', ticker, str(e))

        # ── 3. yfinance — Recent News Headlines ──────────────────────────────
        # Finnhub is no longer used — status stays False so tasks.py data
        # quality guidance handles it correctly.
        status.finnhub = False
        news_sentiment = None  # No free yfinance equivalent for sentiment score
        try:
            news = yf.Ticker(ticker).news
            if news:
                def _extract_title(item):
                    if 'title' in item and item['title']:
                        return item['title']
                    if 'content' in item and isinstance(item['content'], dict):
                        if 'title' in item['content']:
                            return item['content']['title']
                    if 'headline' in item and item['headline']:
                        return item['headline']
                    return None
                headlines = [t for t in [_extract_title(n) for n in news[:5]] if t]
        except Exception as e:
            log_error('yfinance_news', ticker, str(e))

        # ── 4. VIX — Volatility Index ─────────────────────────────────────────
        vix = self.get_vix()

        # ── 5. FRED — Macro Economic Context ─────────────────────────────────
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

        # ── 6. Intraday Indicators (all via Alpaca) ───────────────────────────
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
            vix=vix,
            data_sources_used=status,
        )

    def get_vwap(self, ticker: str) -> tuple:
        """
        Calculate today's VWAP from 1-minute Alpaca bars using close × volume.
        Returns (vwap, price_above_vwap) or (None, None) on failure.
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=8),
            ))
            df = bars.df.reset_index()
            if df.empty:
                return None, None
            vwap_series = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            vwap_val = float(vwap_series.iloc[-1])
            current_price = float(df['close'].iloc[-1])
            return vwap_val, current_price > vwap_val
        except Exception as e:
            log_error('vwap', ticker, str(e))
            return None, None

    def get_opening_range(self, ticker: str) -> tuple:
        """
        Calculate the opening range (9:30–10:00 AM ET) from 1-minute Alpaca bars.
        Returns (orh, orl, orm, orb_breakout_up, orb_breakout_down) or all Nones.
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=8),
            ))
            df = bars.df.reset_index()
            if df.empty:
                return None, None, None, None, None
            # Convert Alpaca UTC timestamps to ET for time-based filtering
            ts = pd.to_datetime(df['timestamp'])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize('UTC')
            df['timestamp'] = ts.dt.tz_convert('America/New_York')
            df = df.set_index('timestamp')
            open_bars = df.between_time('09:30', '09:59')
            if open_bars.empty:
                return None, None, None, None, None
            orh = float(open_bars['high'].max())
            orl = float(open_bars['low'].min())
            orm = (orh + orl) / 2
            current_price = float(df['close'].iloc[-1])
            return orh, orl, orm, current_price > orh, current_price < orl
        except Exception as e:
            log_error('opening_range', ticker, str(e))
            return None, None, None, None, None

    def get_premarket_gap(self, ticker: str) -> tuple:
        """
        Calculate pre-market gap using Alpaca daily bars.
        Uses yesterday's close as prev_close and today's open as last_price.
        Returns (gap_pct, gap_is_bullish, gap_is_bearish) or (None, None, None).
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=5),
            ))
            df = bars.df.reset_index()
            if df.empty or len(df) < 2:
                return None, None, None
            prev_close = float(df['close'].iloc[-2])
            today_open = float(df['open'].iloc[-1])
            if prev_close == 0:
                return None, None, None
            gap_pct = float((today_open - prev_close) / prev_close * 100)
            return gap_pct, gap_pct > 0.5, gap_pct < -0.5
        except Exception as e:
            log_error('premarket_gap', ticker, str(e))
            return None, None, None

    def get_volume_confirmation(self, ticker: str) -> tuple:
        """
        Compare today's volume against the 20-day average using Alpaca daily bars.
        Returns (volume_ratio, volume_confirmed) or (None, None).
        volume_confirmed is True when volume_ratio > 1.20.
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=45),
            ))
            df = bars.df.reset_index()
            print(f'[volume_confirmation] {ticker} — {len(df)} bars returned from Alpaca')
            if df.empty or len(df) < 21:
                return None, None
            avg_volume = float(df['volume'].iloc[-21:-1].mean())  # 20-day avg, excluding today
            today_volume = float(df['volume'].iloc[-1])
            if avg_volume == 0:
                return None, None
            volume_ratio = today_volume / avg_volume
            return float(volume_ratio), volume_ratio > 1.20
        except Exception as e:
            log_error('volume_confirmation', ticker, str(e))
            return None, None

    def get_atr(self, ticker: str, current_price: float) -> Optional[float]:
        """
        Calculate 14-day Average True Range as a percentage of current price.
        ATR% = ATR / current_price * 100
        Uses Alpaca daily bars for the past 35 days to ensure 15+ rows after any
        holiday gaps.
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=35),
            ))
            df = bars.df.reset_index()
            if df.empty or len(df) < 15:
                return None
            prev_close = df['close'].shift(1)
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            if pd.isna(atr) or current_price == 0:
                return None
            return float(atr / current_price * 100)
        except Exception as e:
            log_error('atr', ticker, str(e))
            return None

    def get_vix(self) -> Optional[float]:
        """
        Fetch the current VIX level from yfinance (^VIX), cached daily.

        Uses the same pattern as the FRED macro cache: check for an existing
        daily cache file first, fetch from yfinance on a miss, write on success.
        Returns the VIX close as a float, or None on any failure — callers must
        default to NORMAL regime and 0.82 confidence threshold when None.
        """
        vix_cache = f"{config.cache_dir}/vix_{datetime.now().strftime('%Y%m%d')}.json"
        if os.path.exists(vix_cache):
            try:
                with open(vix_cache) as f:
                    return float(json.load(f)['vix'])
            except Exception:
                pass  # Fall through to live fetch

        try:
            hist = yf.Ticker('^VIX').history(period='1d')
            if not hist.empty:
                vix_val = float(hist['Close'].iloc[-1])
                try:
                    with open(vix_cache, 'w') as f:
                        json.dump({'vix': vix_val}, f)
                except Exception:
                    pass
                return vix_val
        except Exception as e:
            log_error('vix', '^VIX', str(e))

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
