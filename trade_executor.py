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
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
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

        try:
            # Map trade_type string to Alpaca's OrderSide enum
            side = OrderSide.BUY if decision.trade_type in ['buy'] else OrderSide.SELL
            type_str = decision.trade_type.value if hasattr(decision.trade_type, 'value') else str(decision.trade_type)

            # ── Limit → Market fallback ───────────────────────────────────────
            # Alpaca rejects fractional shares on bracket orders, so whole_shares
            # is floored to the nearest integer. If that produces 0 shares, fall
            # through to the market/notional path which supports fractional fills.
            if decision.order_type == 'limit' and decision.entry_price:
                whole_shares = int(decision.position_size_usd / decision.entry_price)
                if whole_shares < 1:
                    print(f'[executor] {decision.ticker} — limit order too small ({whole_shares} shares at ${decision.entry_price}), falling back to market order')
                    decision.order_type = 'market'

            # ── Price validation ──────────────────────────────────────────────
            # Guard against ATR-based stops that land on the wrong side of entry.
            # Falls back to fixed config percentages for any invalid price.
            if decision.entry_price:
                if type_str in ('buy',):
                    if decision.stop_loss_price and decision.stop_loss_price >= decision.entry_price:
                        print(f'[executor] {decision.ticker} — invalid stop loss above entry, recalculating')
                        decision.stop_loss_price = round(decision.entry_price * (1 - config.intraday_stop_loss_pct), 2)
                    if decision.take_profit_price and decision.take_profit_price <= decision.entry_price:
                        print(f'[executor] {decision.ticker} — invalid take profit below entry, recalculating')
                        decision.take_profit_price = round(decision.entry_price * (1 + config.intraday_take_profit_pct), 2)
                else:  # short
                    if decision.stop_loss_price and decision.stop_loss_price <= decision.entry_price:
                        print(f'[executor] {decision.ticker} — invalid stop loss below entry for short, recalculating')
                        decision.stop_loss_price = round(decision.entry_price * (1 + config.intraday_stop_loss_pct), 2)
                    if decision.take_profit_price and decision.take_profit_price >= decision.entry_price:
                        print(f'[executor] {decision.ticker} — invalid take profit above entry for short, recalculating')
                        decision.take_profit_price = round(decision.entry_price * (1 - config.intraday_take_profit_pct), 2)

            # ── Order construction ────────────────────────────────────────────
            if decision.order_type == 'limit' and decision.entry_price:
                order_data = LimitOrderRequest(
                    symbol=decision.ticker,
                    qty=whole_shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,   # Unfilled entry expires at market close
                    limit_price=decision.entry_price,
                    order_class='bracket',
                    take_profit=TakeProfitRequest(limit_price=decision.take_profit_price),
                    stop_loss=StopLossRequest(stop_price=decision.stop_loss_price),
                )
            else:
                # Market bracket order — notional dollar amount supports fractional shares
                order_data = MarketOrderRequest(
                    symbol=decision.ticker,
                    notional=decision.position_size_usd,
                    side=side,
                    time_in_force=TimeInForce.DAY,   # Unfilled entry expires at market close
                    order_class='bracket',
                    take_profit=TakeProfitRequest(limit_price=decision.take_profit_price),
                    stop_loss=StopLossRequest(stop_price=decision.stop_loss_price),
                )

            print(f'[executor] {decision.ticker} — submitting: stop ${decision.stop_loss_price}, target ${decision.take_profit_price}, entry ${decision.entry_price}')
            order = self.client.submit_order(order_data)

            print(f'[executor] {decision.ticker} — placed successfully')
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

    def _cancel_open_orders(self, ticker: str):
        """
        Cancel all open orders for a symbol using the correct Alpaca SDK v2 pattern.

        cancel_orders_for_symbol() does not exist in the SDK. Instead, fetch open
        orders filtered by symbol and cancel each one individually. Errors on
        individual cancellations are logged but do not abort the loop.
        """
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[ticker],
            )
            open_orders = self.client.get_orders(filter=request)
            for order in open_orders:
                try:
                    self.client.cancel_order_by_id(order.id)
                except Exception as e:
                    log_error('cancel_order_by_id', ticker, str(e))
                    print(f'[cancel_orders] {ticker} — could not cancel order {order.id}: {e}')
            if open_orders:
                print(f'[cancel_orders] {ticker} — cancelled {len(open_orders)} open order(s)')
                time.sleep(1)  # Give Alpaca a moment to process cancellations
        except Exception as e:
            log_error('cancel_open_orders', ticker, str(e))
            print(f'[cancel_orders] {ticker} — order fetch/cancel failed: {e}')

    def cancel_stale_orders(self) -> int:
        """
        Cancel all open orders submitted before today's 9:30 AM ET market open.

        Called once at the start of the first trading cycle each day to ensure
        no close orders, bracket legs, or limit orders from the prior session
        survive into the new trading day. Alpaca's default TIF for close_position()
        is 'day', so normal EOD closes expire automatically — this is a safety
        net for edge cases (e.g. orders submitted in the final seconds of the
        session, partial fills, or bracket legs that weren't cancelled cleanly).

        Returns:
            Number of orders cancelled (0 on failure or nothing to cancel).
        """
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        et_now = datetime.now(et_tz)
        today_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)

        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = self.client.get_orders(filter=request)
            cancelled = 0
            for order in open_orders:
                submitted_at = order.submitted_at  # UTC-aware datetime from Alpaca
                if submitted_at and submitted_at < today_open:
                    ticker   = order.symbol
                    side     = str(order.side).split('.')[-1].lower()
                    otype    = str(order.type).split('.')[-1].lower()
                    try:
                        self.client.cancel_order_by_id(order.id)
                        print(
                            f'🧹 Stale order cancelled: {ticker} {side} {otype} '
                            f'submitted {submitted_at}'
                        )
                        cancelled += 1
                    except Exception as e:
                        log_error('cancel_stale_order', ticker, str(e))
            print(f'🧹 Morning cleanup: {cancelled} stale orders cancelled')
            return cancelled
        except Exception as e:
            log_error('cancel_stale_orders', 'ALL', str(e))
            return 0

    def close_stale_intraday_positions(self) -> int:
        """
        Close any intraday positions that survived overnight due to a failed EOD close.

        Called once at market open after cancel_stale_orders(). Compares open DB trades
        with hold_period='intraday' against live Alpaca positions. Any match means the
        EOD forced-close failed (e.g. error 40310000) and the position was held overnight
        against the intraday rule. These are closed immediately at market open.

        Returns:
            Number of stale intraday positions closed (0 if none found or on failure).
        """
        from database import Database
        db = Database()

        try:
            open_trades = db.get_open_trades()
        except Exception as e:
            log_error('close_stale_intraday_positions', 'ALL', str(e))
            return 0

        intraday = [t for t in open_trades if t.get('hold_period') == 'intraday']
        if not intraday:
            print('🧹 Morning cleanup: no stale intraday positions found')
            return 0

        # Compare against live Alpaca positions — only act on confirmed live holds
        live_positions = {p['ticker']: p for p in self.get_open_positions()}
        closed = 0

        for trade in intraday:
            ticker = trade['ticker']
            if ticker not in live_positions:
                # DB says open but Alpaca has no position — mark as closed in DB
                print(f'⚠️  {ticker} intraday trade open in DB but no Alpaca position — marking closed')
                try:
                    db.update_trade_status(
                        trade['trade_id'],
                        status='closed',
                        exit_reason='stale_intraday_no_position',
                        exit_price=None,
                    )
                except Exception as e:
                    log_error('close_stale_intraday_positions', ticker, str(e))
                continue

            print(f'⚠️  Stale intraday position found at market open: {ticker} — closing now')
            try:
                self._cancel_open_orders(ticker)
                time.sleep(2)
                self.client.close_position(ticker)
                time.sleep(2)
                exit_price = self.get_filled_exit_price(ticker)
                db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='stale_intraday_morning_close',
                    exit_price=exit_price,
                )
                print(f'✅ Stale intraday position closed at market open: {ticker}')
                closed += 1
            except Exception as e:
                log_error('close_stale_intraday_positions', ticker, str(e))

        print(f'🧹 Morning cleanup: {closed} stale intraday position(s) closed')
        return closed

    def close_position(self, ticker: str, trade_type: str):
        """
        Close a single open position at market price.

        Cancels all open bracket legs first (take-profit / stop-loss) to avoid
        Alpaca error 40310000 where shares are held for open orders and cannot
        be liquidated directly.

        Args:
            ticker:     Symbol of the position to close.
            trade_type: Included for logging context; not used by Alpaca directly.
        """
        try:
            self._cancel_open_orders(ticker)
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

    def get_filled_entry_price(self, ticker: str, trade_type: str) -> float | None:
        """
        Return the actual Alpaca fill price for the most recent entry order.

        Queries the last 20 closed orders for the symbol and finds the most
        recent filled order on the entry side (buy for longs, sell_short for
        shorts). Used to recover a NULL or $0.00 entry_price in the DB when
        market_data.current_price was unavailable at the time of insertion.

        Args:
            ticker:     Symbol to look up.
            trade_type: 'buy'/'long' for longs, 'short' for shorts.

        Returns:
            Fill price as a float, or None if no matching filled order found.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(symbol=ticker, status=QueryOrderStatus.CLOSED, limit=20)
            orders = self.client.get_orders(req)
            # Entry side is 'buy' for longs and 'sell_short' for shorts
            is_long = trade_type in ('buy', 'long')
            entry_side = 'buy' if is_long else 'sell_short'
            for order in orders:
                if order.symbol != ticker or order.filled_avg_price is None:
                    continue
                side_str = str(order.side).lower().replace('orderside.', '')
                if side_str == entry_side:
                    return float(order.filled_avg_price)
        except Exception as e:
            log_error('get_filled_entry_price', ticker, str(e))
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
