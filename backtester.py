"""
backtester.py — Historical strategy validation against Alpaca OHLCV data.

Provides a lightweight RSI-based backtest to validate that the core signal
logic (oversold entries, overbought/stop exits) produces acceptable results
BEFORE deploying the full LLM crew in paper trading mode.

Important design note:
    The simple_rsi_strategy() used here is a proxy — not the live system.
    The live agent uses multi-source LLM reasoning (bull/bear/risk crew) which
    cannot be replicated in a backtest without replaying historical news and
    macro data. This backtest validates directional signal quality only.

Acceptance thresholds (checked in run()):
    Win rate  ≥ 50%
    Sharpe ratio ≥ 0.5 (annualised, 252-day basis)

Run directly:
    python backtester.py

Or from another module:
    from backtester import Backtester
    results = Backtester().run(days=180)
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from config import config
from logger import log_error
import numpy as np


class Backtester:
    """
    Fetches historical data, simulates trades, and computes performance metrics
    across the full watchlist. Stateless between tickers — results are aggregated
    in run() rather than stored on the instance.
    """

    def __init__(self):
        # Historical data client — read-only, no trading permissions needed
        self.client = StockHistoricalDataClient(
            config.alpaca_api_key, config.alpaca_secret_key
        )

    # ── Data Fetching ─────────────────────────────────────────────────────────

    def get_historical_data(self, ticker: str, days: int = 180) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a ticker and compute technical indicators.

        250 bars are requested (not `days`) for the SMA-200 calculation to have
        enough history even when `days` is only 180. The caller's `days` parameter
        controls the backtest window; indicator lookbacks need extra runway.

        Returns an empty DataFrame on failure so callers can check df.empty
        without catching exceptions.

        Args:
            ticker: Symbol to fetch (e.g. 'AAPL').
            days:   Number of calendar days of history to retrieve.

        Returns:
            DataFrame with OHLCV columns plus RSI_14, MACD_12_26_9, SMA_50, SMA_200.
            Empty DataFrame on API error.
        """
        try:
            bars = self.client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=days),
            ))
            df = bars.df.reset_index()

            # Compute indicators in-place using pandas-ta
            df.ta.rsi(append=True)
            df.ta.macd(append=True)
            df['SMA_50']  = df['close'].rolling(50).mean()
            df['SMA_200'] = df['close'].rolling(200).mean()

            return df

        except Exception as e:
            log_error('backtester', ticker, str(e))
            return pd.DataFrame()  # Empty frame signals failure to caller

    # ── Strategy Simulation ───────────────────────────────────────────────────

    def simple_rsi_strategy(self, df: pd.DataFrame) -> list:
        """
        Simulate an RSI mean-reversion strategy on historical bars.

        Entry conditions (all must be true):
            - No current open position
            - RSI_14 < 30 (oversold)
            - Close price > SMA_50 (long-term uptrend still intact)

        Exit conditions (first triggered wins):
            1. Price falls to stop-loss (entry × 0.95) → -5% loss cap
            2. Price rises to take-profit (entry × 1.10) → +10% target
            3. RSI_14 > 70 (overbought signal regardless of price)

        Note: This is a simplified proxy for the live LLM-driven strategy.
        It validates that the underlying technicals have directional edge, but
        cannot replicate the sentiment and macro reasoning of the full crew.

        Args:
            df: DataFrame from get_historical_data() with indicator columns.

        Returns:
            List of trade dicts, each with entry/exit date, prices, P&L %, and exit reason.
        """
        trades = []
        position = None  # None = flat; dict = holding a position

        # Start at index 1 so we can safely reference prior bars if needed later
        for i in range(1, len(df)):
            row   = df.iloc[i]
            rsi   = row.get('RSI_14')
            price = row['close']
            date  = str(row.get('timestamp', i))

            # Skip bars where the indicator hasn't warmed up yet (NaN during lookback)
            if pd.isna(rsi):
                continue

            if position is None:
                # ── Entry Logic ───────────────────────────────────────────────
                # RSI oversold + price above 50-day MA confirms we're not in freefall
                if rsi < 30:
                    if not pd.isna(row.get('SMA_50')) and price > row['SMA_50']:
                        position = {
                            'entry_price': price,
                            'entry_date':  date,
                            'stop_loss':   price * 0.95,   # -5% hard stop
                            'take_profit': price * 1.10,   # +10% target (2:1 R/R)
                        }

            else:
                # ── Exit Logic ────────────────────────────────────────────────
                # Evaluate all three exit conditions; earliest trigger wins
                exit_price  = None
                exit_reason = None

                if price <= position['stop_loss']:
                    # Price hit the stop — use the stop price, not close,
                    # to avoid underestimating losses (assumes fill at stop level)
                    exit_price  = position['stop_loss']
                    exit_reason = 'stop_loss'

                elif price >= position['take_profit']:
                    exit_price  = position['take_profit']
                    exit_reason = 'take_profit'

                elif rsi > 70:
                    # RSI overbought — exit at close regardless of P&L
                    exit_price  = price
                    exit_reason = 'rsi_overbought'

                if exit_price:
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    trades.append({
                        'entry_date':  position['entry_date'],
                        'exit_date':   date,
                        'entry_price': position['entry_price'],
                        'exit_price':  exit_price,
                        'pnl_pct':     pnl_pct,
                        'exit_reason': exit_reason,
                    })
                    position = None  # Reset to flat for next entry signal

        return trades

    # ── Metrics ───────────────────────────────────────────────────────────────

    def calculate_metrics(self, trades: list) -> dict:
        """
        Compute standard strategy performance metrics from a list of closed trades.

        Metrics returned:
            total_trades:     Number of completed round-trips.
            win_rate:         Fraction of trades with positive P&L.
            total_return_pct: Sum of all P&L percentages (simple, not compounded).
            avg_win_pct:      Average P&L of winning trades.
            avg_loss_pct:     Average P&L of losing trades (negative value).
            profit_factor:    |avg_win| / |avg_loss| — values > 1 are profitable.
            sharpe_ratio:     Annualised Sharpe using trade P&L as the return series.
                              Assumes 252 trading days per year.
            max_drawdown_pct: Maximum peak-to-trough decline in cumulative equity.

        Args:
            trades: List of trade dicts from simple_rsi_strategy().

        Returns:
            Dict of metric name → value, or {'error': '...'} if no trades exist.
        """
        if not trades:
            return {'error': 'No trades to analyze'}

        pnls   = [t['pnl_pct'] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate     = len(wins) / len(trades)
        avg_win      = np.mean(wins)   if wins   else 0
        avg_loss     = np.mean(losses) if losses else 0
        total_return = sum(pnls)

        # Annualised Sharpe: mean / std * sqrt(252)
        # Guard against zero std (all trades same P&L) or single-trade sets
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown: compute cumulative equity, find rolling peak, measure drop
        cumulative   = np.cumprod([1 + p for p in pnls])
        peak         = np.maximum.accumulate(cumulative)
        drawdown     = (peak - cumulative) / peak
        max_drawdown = np.max(drawdown)

        return {
            'total_trades':      len(trades),
            'win_rate':          win_rate,
            'total_return_pct':  total_return * 100,
            'avg_win_pct':       avg_win  * 100,
            'avg_loss_pct':      avg_loss * 100,
            # profit_factor = 0 when no losses occurred (avoid division by zero)
            'profit_factor':     abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio':      sharpe,
            'max_drawdown_pct':  max_drawdown * 100,
        }

    # ── Runner ────────────────────────────────────────────────────────────────

    def run(self, days: int = 180):
        """
        Execute the full backtest across every ticker in config.watchlist.

        Per-ticker results are printed inline for progress visibility.
        Overall aggregated metrics are printed at the end with a pass/fail
        recommendation based on minimum acceptance thresholds.

        Acceptance criteria:
            Win rate  ≥ 50% — strategy wins more than it loses
            Sharpe    ≥ 0.5 — sufficient risk-adjusted return to justify live use

        Args:
            days: Number of calendar days of historical data to test against.
        """
        print(f'\n📊 Running backtest over last {days} days...')
        print(f'Tickers: {config.watchlist}\n')

        all_trades = []

        for ticker in config.watchlist:
            print(f'Testing {ticker}...')
            df = self.get_historical_data(ticker, days)

            if df.empty:
                print(f'  ⚠️  No data for {ticker}')
                continue

            trades = self.simple_rsi_strategy(df)
            all_trades.extend(trades)  # Accumulate for overall metrics

            if trades:
                metrics = self.calculate_metrics(trades)
                print(
                    f'  Trades: {metrics["total_trades"]} | '
                    f'Win Rate: {metrics["win_rate"]:.1%} | '
                    f'Return: {metrics["total_return_pct"]:.1f}%'
                )

        # ── Overall Summary ───────────────────────────────────────────────────
        print('\n' + '=' * 50)
        print('OVERALL BACKTEST RESULTS')
        print('=' * 50)

        overall = self.calculate_metrics(all_trades)
        for key, value in overall.items():
            if isinstance(value, float):
                print(f'{key}: {value:.2f}')
            else:
                print(f'{key}: {value}')

        # ── Pass / Fail Recommendation ────────────────────────────────────────
        print('\n📋 RECOMMENDATION:')
        if overall.get('win_rate', 0) >= 0.50 and overall.get('sharpe_ratio', 0) >= 0.5:
            print('✅ Backtest results acceptable — proceed to paper trading')
        else:
            print('⚠️  Backtest results below threshold — review strategy before paper trading')
            print('   Win rate should be >= 50% and Sharpe ratio >= 0.5')


# ── Script Entrypoint ─────────────────────────────────────────────────────────
# Run directly with: python backtester.py
if __name__ == '__main__':
    backtester = Backtester()
    backtester.run(days=180)
