"""
trade_executor.py — Alpaca order placement and position management.

Responsible for all interactions with the Alpaca Trading API:
    - Reading account and position state
    - Placing bracket orders (entry + take-profit + stop-loss in a single request)
    - Closing individual positions and emergency closing all positions

Bracket orders are used for every entry so that take-profit and stop-loss
levels computed by position_sizer.py are submitted atomically with the entry.
This eliminates the race condition where a fill occurs but the protective
orders haven't been placed yet.

Paper vs live mode is controlled entirely by config.trading_mode — no other
code change is required to switch environments.

Usage:
    from trade_executor import TradeExecutor
    executor = TradeExecutor()
    result = executor.execute_trade(decision)
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    StopOrderRequest, TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from config import config
from logger import log_error
from models import TradeDecision
import time


class TradeExecutor:
    """
    Thin wrapper around the Alpaca TradingClient that enforces pre-flight
    checks (confidence threshold, execute flag) before any order is sent.
    """

    def __init__(self):
        # paper=True routes all orders to the Alpaca paper trading environment.
        # Comparing to the string 'paper' (not the enum) because config.trading_mode
        # is a Pydantic field that may be stored as a string at runtime.
        self.client = TradingClient(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            paper=config.trading_mode == 'paper',
        )

    # ── Account State ─────────────────────────────────────────────────────────

    def get_portfolio_value(self) -> float:
        """
        Fetch the current total liquidation value of the account.

        Called by the circuit breaker and crew at the start of each cycle
        to establish the baseline for drawdown calculation and position sizing.

        Returns:
            Portfolio value in USD, or 0.0 on API failure (logged).
        """
        try:
            account = self.client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            log_error('alpaca_account', 'portfolio', str(e))
            return 0.0  # Safe default — crew will skip sizing if value is 0

    def get_open_positions(self) -> list:
        """
        Return a normalised list of all currently open positions.

        Converts Alpaca position objects to plain dicts so callers don't
        need to import Alpaca types. Used by the crew to check concentration
        before approving new entries.

        Returns:
            List of position dicts with keys:
                ticker, qty, market_value, unrealized_pl, side
            Returns an empty list on API failure (logged).
        """
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    'ticker':        p.symbol,
                    'qty':           float(p.qty),
                    'market_value':  float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl),
                    'side':          p.side,
                }
                for p in positions
            ]
        except Exception as e:
            log_error('alpaca_positions', 'all', str(e))
            return []

    # ── Order Placement ───────────────────────────────────────────────────────

    def execute_trade(self, decision: TradeDecision) -> dict:
        """
        Translate a TradeDecision into an Alpaca bracket order.

        Pre-flight checks:
            1. decision.execute must be True — False means the crew passed
               on this ticker and no order should be placed.
            2. Confidence must meet or exceed config.confidence_threshold (0.75).
               This is a second gate after the Pydantic validator in TradeDecision
               to guard against any object constructed outside normal crew flow.

        Order type selection:
            LIMIT — preferred; uses entry_price from position_sizer.py to
                    control slippage on the fill.
            MARKET — fallback when no entry_price was computed; uses notional
                     (dollar amount) rather than share qty for fractional support.

        Both order types are submitted as bracket orders so take-profit and
        stop-loss legs are placed atomically with the entry order.

        Args:
            decision: Validated TradeDecision from the risk manager agent.

        Returns:
            Dict with 'status' key:
                'placed'  — order submitted successfully; includes order_id.
                'skipped' — pre-flight check failed; no order sent.
                'error'   — Alpaca API error; details in 'error' key.
        """
        # Gate 1: crew explicitly flagged this as a no-trade cycle
        if not decision.execute:
            print(f'[executor] {decision.ticker} — skipped: execute=False')
            return {'status': 'skipped', 'reason': 'execute=False'}

        # Gate 2: confidence below minimum threshold
        if decision.confidence < config.confidence_threshold:
            print(f'[executor] {decision.ticker} — skipped: confidence {decision.confidence} below threshold {config.confidence_threshold}')
            return {
                'status': 'skipped',
                'reason': f'confidence {decision.confidence} below threshold',
            }

        try:
            # Map trade_type string to Alpaca's OrderSide enum
            side = OrderSide.BUY if decision.trade_type in ['buy'] else OrderSide.SELL

            if decision.order_type == 'limit' and decision.entry_price:
                # Limit bracket order — preferred path for controlled entry.
                # Alpaca rejects fractional shares on bracket orders, so qty is
                # floored to the nearest whole share. This slightly undersizes the
                # position but guarantees the bracket order is accepted.
                whole_shares = int(decision.position_size_usd / decision.entry_price)
                if whole_shares < 1:
                    print(f'[executor] {decision.ticker} — skipped: position too small ({whole_shares} shares at ${decision.entry_price})')
                    return {'status': 'skipped', 'reason': 'position too small for 1 whole share'}
                order_data = LimitOrderRequest(
                    symbol=decision.ticker,
                    qty=whole_shares,
                    side=side,
                    time_in_force=TimeInForce.GTC,   # Unfilled limit expires at close
                    limit_price=decision.entry_price,
                    order_class='bracket',
                    take_profit=TakeProfitRequest(limit_price=decision.take_profit_price),
                    stop_loss=StopLossRequest(stop_price=decision.stop_loss_price),
                )
            else:
                # Market bracket order — fallback; uses notional for fractional share support
                order_data = MarketOrderRequest(
                    symbol=decision.ticker,
                    notional=decision.position_size_usd,  # Dollar amount, not share qty
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    order_class='bracket',
                    take_profit=TakeProfitRequest(limit_price=decision.take_profit_price),
                    stop_loss=StopLossRequest(stop_price=decision.stop_loss_price),
                )

            print(f'[executor] {decision.ticker} — stop: ${decision.stop_loss_price}, target: ${decision.take_profit_price}, entry: ${decision.entry_price}')
            order = self.client.submit_order(order_data)

            print(f'[executor] {decision.ticker} — placed successfully: {whole_shares if decision.order_type == "limit" and decision.entry_price else "notional"} shares at ${decision.entry_price}')
            print(
                f'✅ Order placed: {decision.trade_type} {decision.ticker} '
                f'| ${decision.position_size_usd:.2f}'
            )
            return {
                'status':     'placed',
                'order_id':   str(order.id),
                'ticker':     decision.ticker,
                'trade_type': decision.trade_type,
                'notional':   decision.position_size_usd,
            }

        except Exception as e:
            print(f'[executor] {decision.ticker} — ERROR: {str(e)}')
            log_error('trade_executor', decision.ticker, str(e))
            return {'status': 'error', 'error': str(e)}

    # ── Position Closing ──────────────────────────────────────────────────────

    def close_position(self, ticker: str, trade_type: str):
        """
        Close a single open position at market price.

        Called by position_monitor.py for hold period expiry and intraday
        forced closes. Alpaca automatically cancels any open bracket legs
        (take-profit / stop-loss) when the position is closed this way.

        Args:
            ticker:     Symbol of the position to close.
            trade_type: Included for logging context; not used by Alpaca directly.
        """
        try:
            self.client.close_position(ticker)
            print(f'✅ Position closed: {ticker}')
        except Exception as e:
            log_error('close_position', ticker, str(e))

    def get_filled_exit_price(self, ticker: str) -> float | None:
        """
        Return the average fill price of the most recent closed order for ticker.

        Queries the last 10 orders for the symbol and returns the filled_avg_price
        of the most recent filled order. Returns None if no filled order is found.

        Args:
            ticker: Symbol to look up.

        Returns:
            Fill price as a float, or None if unavailable.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(symbol=ticker, status=QueryOrderStatus.CLOSED, limit=10)
            orders = self.client.get_orders(req)
            for order in orders:
                if order.symbol == ticker and order.filled_avg_price is not None:
                    return float(order.filled_avg_price)
        except Exception as e:
            log_error('get_filled_exit_price', ticker, str(e))
        return None

    def close_all_positions(self):
        """
        Emergency close of all open positions.

        Intended for use by the circuit breaker after a 10% drawdown trigger.
        cancel_orders=True ensures all pending bracket and limit orders are
        also cancelled so no new fills can occur after the emergency close.

        Note: This is a last-resort method. Normal exits go through
        close_position() so each closure is individually logged and recorded
        in the database by position_monitor.py.
        """
        try:
            self.client.close_all_positions(cancel_orders=True)
            print('🚨 All positions closed by circuit breaker')
        except Exception as e:
            log_error('close_all_positions', 'ALL', str(e))
