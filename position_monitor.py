"""
position_monitor.py — Enforces hold period rules and intraday exit discipline.

Runs on every scheduler cycle to audit all open positions against their
time-based exit constraints. Two exit mechanisms are managed here:

1. Hold Period Expiry — any position that has been held longer than its
   max_hold_days budget (set at entry time by position_sizer.py) is closed
   regardless of P&L. This prevents trades from drifting outside their
   intended risk window.

2. Intraday Forced Close — all positions classified as 'intraday' are closed
   before market close (3:45 PM cycle) to eliminate overnight gap risk.

Note: Stop-loss and take-profit exits are handled by Alpaca bracket orders
placed at entry time via trade_executor.py — not by this module.

Usage:
    from position_monitor import PositionMonitor
    monitor = PositionMonitor(trade_executor=executor)
    monitor.check_all_positions()
"""

import time
from database import Database
from config import config, HoldPeriod
from datetime import datetime, timedelta
from logger import log_error


def get_profit_threshold(atr_pct, minutes_held: float) -> float:
    """
    Return the minimum gain % (as a plain percentage, e.g. 1.5 means 1.5%)
    required to trigger a dynamic take-profit exit.

    Two-axis logic:
        - Fresh positions (< 10 min): use full ATR-tiered targets so the
          position has room to reach its original bracket take-profit.
        - Aging positions (10 min+): lower the bar progressively so
          unrealised gains don't reverse back to breakeven or a loss.

    Args:
        atr_pct:      ATR as a percentage of price (e.g. 2.3 means 2.3%).
                      Pass 0 or None to receive the medium-volatility default.
        minutes_held: Minutes elapsed since entry.

    Returns:
        Profit threshold as a percentage float.
    """
    if minutes_held < 10:
        # Fresh position — let it run to the full ATR-calibrated target
        if not atr_pct or atr_pct < 2.0:
            return 1.5   # Low volatility: AAPL, MSFT, JPM
        elif atr_pct < 3.5:
            return 2.0   # Medium volatility: META, AMZN, NVDA
        else:
            return 2.5   # High volatility: TSLA, AMD, COIN
    elif minutes_held < 40:
        return 0.35      # 10–40 min held — take any gain above 0.35%
    elif minutes_held < 70:
        return 0.25      # 40–70 min held — take any gain above 0.25%
    else:
        return 0.20      # 70+ min held — take any gain above 0.20%


class PositionMonitor:
    """
    Audits open positions on every cycle and triggers time-based exits.
    Requires a live TradeExecutor instance to place closing orders.
    """

    def __init__(self, trade_executor):
        self.db = Database()
        # Executor is injected rather than instantiated here to share the
        # same authenticated Alpaca client used by the rest of the crew
        self.executor = trade_executor
        # In-memory price history for momentum fade detection.
        # Keyed by trade_id; each value is [older_price, last_price] (oldest first).
        # Reset on process restart — only needs to persist across scheduler cycles.
        self._price_history: dict = {}
        # In-memory peak gain tracking for trailing profit protection.
        # Keyed by trade_id; value is the highest gain_pct seen since entry (as a fraction).
        self._peak_gain_pct: dict = {}

    # ── Public Interface ──────────────────────────────────────────────────────

    def reconcile_bracket_exits(self):
        """
        Detect positions closed by Alpaca bracket fills and record them in the DB.

        When a bracket take-profit or stop-loss leg fires, Alpaca closes the
        position with no callback to our code — the trade row stays status='open'
        forever and never appears in P&L or win-rate metrics.

        This method compares every DB-open trade against live Alpaca positions.
        Any DB-open trade whose ticker has no live Alpaca position is presumed
        closed by a bracket fill; the exit price is fetched from Alpaca's order
        history and the row is written to closed with pnl computed.

        Called at the start of every scheduler cycle (before check_all_positions)
        so the DB stays in sync with actual Alpaca state within one cycle.
        """
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return

        live_positions = {p['ticker'] for p in self.executor.get_open_positions()}

        for trade in open_trades:
            ticker = trade['ticker']
            if ticker in live_positions:
                continue  # Position still open — nothing to reconcile

            # Position gone from Alpaca — bracket leg fired or external close
            exit_price  = self.executor.get_filled_exit_price(ticker)
            entry_price = trade.get('entry_price')
            take_profit = trade.get('take_profit_price')
            stop_loss   = trade.get('stop_loss_price')
            is_long     = trade.get('trade_type', 'buy') in ('buy', 'long')

            # Classify the exit by which bracket level the fill price is closest to
            exit_reason = 'bracket_fill'
            if exit_price and take_profit and stop_loss:
                dist_tp = abs(exit_price - take_profit)
                dist_sl = abs(exit_price - stop_loss)
                exit_reason = 'bracket_take_profit' if dist_tp < dist_sl else 'bracket_stop_loss'

            print(
                f'🔄 Reconciling {ticker}: no live Alpaca position — '
                f'recording {exit_reason} at ${exit_price}'
            )
            try:
                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason=exit_reason,
                    exit_price=exit_price,
                )
            except Exception as e:
                log_error('reconcile_bracket_exits', ticker, str(e))

    def check_all_positions(self):
        """
        Retrieve all open trades from the database and evaluate each one
        against its hold period constraint. Called at the start of every
        scheduler cycle before new signals are analysed.
        """
        open_trades = self.db.get_open_trades()
        for trade in open_trades:
            self._check_hold_expiry(trade)

    def check_dynamic_exits(self):
        """
        Evaluate open positions against dynamic intraday exit conditions.

        Runs on every cycle immediately after check_all_positions(). Closes a
        position when any of the following are true:

        1. Unrealized gain > 2.5%            — take profit early rather than
                                               waiting for the fixed bracket
        2. Open > 2 hours AND gain > 1%      — good enough; free the capital
        3. Price drops below VWAP (long)     — intraday momentum has reversed
        4. SPY drops > 1.5% from today's open — broad market reversal; exit
                                               all longs immediately

        The stop-loss bracket placed at entry is not modified — it remains the
        hard floor. Only the take-profit side is replaced by this dynamic logic.
        """
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return

        # Live unrealized P&L from Alpaca, keyed by ticker
        alpaca_positions = {p['ticker']: p for p in self.executor.get_open_positions()}

        # SPY intraday change from today's open — shared signal for all positions
        spy_reversal = False
        try:
            import yfinance as yf
            spy_info = yf.Ticker('SPY').fast_info
            if spy_info.open and spy_info.last_price and spy_info.open > 0:
                spy_drop = (spy_info.last_price - spy_info.open) / spy_info.open
                if spy_drop <= -0.015:
                    spy_reversal = True
                    print(f'🚨 SPY down {spy_drop*100:.2f}% from open — dynamic exit triggered for all longs')
        except Exception as e:
            log_error('dynamic_exit_spy', 'SPY', str(e))

        # DataCollector provides get_vwap() — imported lazily to avoid a
        # circular import at module level (data_collector imports config)
        from data_collector import DataCollector
        collector = DataCollector()

        for trade in open_trades:
            ticker = trade['ticker']
            trade_type = trade.get('trade_type', 'buy')
            is_long = trade_type in ('buy', 'long')

            alpaca_pos = alpaca_positions.get(ticker)
            if not alpaca_pos:
                continue  # Position not yet reflected in Alpaca — skip this cycle

            unrealized_pl = alpaca_pos['unrealized_pl']
            entry_price = trade.get('entry_price')
            shares = trade.get('shares') or 0

            # Derive current price from Alpaca position — market_value / qty works
            # for both longs (both positive) and shorts (both negative)
            raw_qty = alpaca_pos.get('qty') or 0
            current_price = abs(alpaca_pos['market_value']) / abs(raw_qty) if raw_qty != 0 else None

            # Unrealized gain % from entry; None when entry data is missing
            gain_pct = None
            if entry_price and entry_price > 0 and shares > 0:
                gain_pct = unrealized_pl / (entry_price * shares)

            # Update peak gain — track the highest gain% seen for this trade
            trade_id = trade['trade_id']
            if gain_pct is not None:
                if gain_pct > self._peak_gain_pct.get(trade_id, float('-inf')):
                    self._peak_gain_pct[trade_id] = gain_pct

            # Time-and-ATR-tiered profit threshold — combines ATR volatility regime
            # with how long the position has been open. Fresh positions must hit
            # the full ATR target; aging positions lower the bar to lock in gains
            # before they reverse.
            atr_pct = trade.get('atr_pct')
            entry_time_str = trade.get('entry_time')
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    minutes_held = (datetime.now() - entry_dt).total_seconds() / 60
                except Exception:
                    minutes_held = 0.0
            else:
                minutes_held = 0.0

            profit_threshold = get_profit_threshold(atr_pct, minutes_held)

            exit_reason = None

            # Momentum fade detection — two consecutive declining cycles while below entry.
            # For longs: fade = price dropped each of the last 2 cycles AND below entry.
            # For shorts: fade = price rose each of the last 2 cycles AND above entry.
            if exit_reason is None and current_price is not None and entry_price and minutes_held >= 10:
                history = self._price_history.get(trade_id, [])
                if len(history) >= 2:
                    older_price, last_price = history[0], history[1]
                    below_entry = (is_long and current_price < entry_price) or (not is_long and current_price > entry_price)
                    fading      = (is_long and current_price < last_price < older_price) or (not is_long and current_price > last_price > older_price)
                    if below_entry and fading:
                        exit_reason = 'momentum_fade'
                        print(
                            f'📉 {ticker} confirmed gradual fade — '
                            f'price lower 2 consecutive cycles while below entry — exiting'
                        )

            # Update price history for this trade — keep last 2 prices only
            if current_price is not None:
                history = self._price_history.get(trade_id, [])
                self._price_history[trade_id] = (history + [current_price])[-2:]

            # Time-based loss exit — cut losing positions early before the bracket fires.
            # Triggers only after 20 min held AND position is down 0.35%+.
            # gain_pct is already directionally correct for both longs and shorts
            # (Alpaca unrealized_pl is negative when losing regardless of side).
            if exit_reason is None and gain_pct is not None and minutes_held >= 20 and gain_pct <= -0.0035:
                exit_reason = 'time_loss_exit'
                print(
                    f'⏱️ {ticker} held {minutes_held:.0f}min at {gain_pct*100:.2f}% '
                    f'— time-based loss exit triggered, cutting position'
                )

            # Peak profit trailing exit — protect gains that have pulled back significantly.
            # Only activates once peak has reached 0.25%; exits if pullback from peak >= 0.15%.
            # Runs before the tiered threshold so a reversing winner exits immediately.
            if exit_reason is None and gain_pct is not None:
                peak = self._peak_gain_pct.get(trade_id, 0.0)
                if peak >= 0.0025:
                    pullback = peak - gain_pct
                    if pullback >= 0.0015:
                        exit_reason = 'peak_pullback_exit'
                        print(
                            f'📈 {ticker} peak was +{peak*100:.2f}% now +{gain_pct*100:.2f}% '
                            f'— pulled back {pullback*100:.2f}% from peak — exiting to protect profits'
                        )

            # Condition 1: gain exceeds time-and-ATR-tiered threshold — dynamic take-profit
            if exit_reason is None and gain_pct is not None:
                gain_display = f'{gain_pct * 100:+.2f}%'
                if gain_pct > profit_threshold / 100:
                    exit_reason = 'dynamic_take_profit'
                    print(
                        f'⏱️  {ticker} held {minutes_held:.0f}min — profit threshold: {profit_threshold:.2f}% '
                        f'(current gain: {gain_display}) — ABOVE threshold, exiting'
                    )
                else:
                    print(
                        f'⏱️  {ticker} held {minutes_held:.0f}min — profit threshold: {profit_threshold:.2f}% '
                        f'(current gain: {gain_display}) — BELOW threshold, holding'
                    )

            # Condition 2: open > 2 hours AND gain > 1% — free the capital
            if exit_reason is None and gain_pct is not None and gain_pct > 0.01:
                entry_time = datetime.fromisoformat(trade['entry_time'])
                hours_open = (datetime.now() - entry_time).total_seconds() / 3600
                if hours_open >= 2.0:
                    exit_reason = 'dynamic_time_profit'
                    print(f'⏱️ {ticker} 2h+ with {gain_pct*100:.2f}% gain — freeing capital')

            # Fetch VWAP once per ticker — used by conditions 3 and 5
            vwap_val = price_above_vwap = None
            try:
                vwap_val, price_above_vwap = collector.get_vwap(ticker)
            except Exception as e:
                log_error('dynamic_exit_vwap', ticker, str(e))

            # Condition 3: VWAP cross against long position
            if exit_reason is None and is_long:
                if vwap_val is not None and price_above_vwap is False:
                    exit_reason = 'vwap_cross_exit'
                    print(f'📉 {ticker} dropped below VWAP ({vwap_val:.2f}) — exiting long')

            # Condition 4: SPY broad market reversal — exit all longs
            if exit_reason is None and is_long and spy_reversal:
                exit_reason = 'spy_reversal_exit'
                print(f'🚨 {ticker} closed: SPY intraday reversal > 1.5%')

            # Condition 5: loss > 1.5% + VWAP momentum reversal against position
            if exit_reason is None and vwap_val is not None:
                market_value = alpaca_pos.get('market_value')
                if market_value and abs(market_value) > 0:
                    loss_pct = unrealized_pl / abs(market_value)
                    if is_long and loss_pct < -0.015 and price_above_vwap is False:
                        exit_reason = 'loss_vwap_reversal'
                        print(f'🛑 {ticker} loss-based exit: {loss_pct*100:.1f}% loss + VWAP reversal — cutting position')
                    elif not is_long and loss_pct < -0.015 and price_above_vwap is True:
                        exit_reason = 'loss_vwap_reversal'
                        print(f'🛑 {ticker} loss-based exit: {loss_pct*100:.1f}% loss + VWAP reversal — cutting position')

            if exit_reason:
                try:
                    self.executor.close_position(ticker, trade_type)
                    time.sleep(2)
                    exit_price = self.executor.get_filled_exit_price(ticker)
                    self.db.update_trade_status(
                        trade['trade_id'],
                        status='closed',
                        exit_reason=exit_reason,
                        exit_price=exit_price,
                    )
                except Exception as e:
                    log_error('dynamic_exit', ticker, str(e))

        # Clean up in-memory dicts for positions that are no longer open
        active_ids = {t['trade_id'] for t in open_trades}
        self._price_history  = {k: v for k, v in self._price_history.items()  if k in active_ids}
        self._peak_gain_pct  = {k: v for k, v in self._peak_gain_pct.items()  if k in active_ids}

    def check_market_reversal(self) -> 'str | None':
        """
        Detect sharp intraday SPY moves and signal which side needs to be covered.

        Uses today's opening bar (first 1-min bar after 9:30 AM ET) as the
        reference price, comparing it to the most recent bar's close. Fetches
        data via the DataCollector's Alpaca client — same pattern as get_vwap().

        Returns:
            'cover_shorts' — SPY up > 2% from open and short positions are open
            'cover_longs'  — SPY down > 2% from open and long positions are open
            None           — no reversal, or fetch failed
        """
        try:
            from data_collector import DataCollector
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            import pandas as pd

            dc = DataCollector()
            bars = dc.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols='SPY',
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=8),
            ))
            df = bars.df.reset_index()
            if df.empty:
                return None

            # Convert to ET and isolate bars from today's regular session open
            ts = pd.to_datetime(df['timestamp'])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize('UTC')
            df['timestamp'] = ts.dt.tz_convert('America/New_York')
            df = df.set_index('timestamp')
            session_bars = df.between_time('09:30', '16:00')
            if session_bars.empty:
                return None

            open_price    = float(session_bars['open'].iloc[0])
            current_price = float(session_bars['close'].iloc[-1])
            if open_price == 0:
                return None

            # Sanity check: if open_price is more than 10% from current_price,
            # the bar data is stale, pre-market, or from a different session.
            # Log and bail rather than firing a false reversal signal.
            if abs(open_price - current_price) / current_price > 0.10:
                print(f'[market_reversal] SPY open_price ${open_price:.2f} is >10% from current ${current_price:.2f} — likely bad bar data, skipping')
                log_error('check_market_reversal', 'SPY', f'open_price sanity check failed: open={open_price}, current={current_price}')
                return None

            spy_move = (current_price - open_price) / open_price

            open_trades = self.db.get_open_trades()
            has_longs  = any(t.get('trade_type') in ('buy', 'long')  for t in open_trades)
            has_shorts = any(t.get('trade_type') in ('short',)        for t in open_trades)

            if spy_move >= 0.02 and has_shorts:
                print(f'🚨 Market reversal detected: SPY up {spy_move*100:.1f}% from open — covering all shorts')
                return 'cover_shorts'
            if spy_move <= -0.02 and has_longs:
                print(f'🚨 Market reversal detected: SPY down {abs(spy_move)*100:.1f}% from open — covering all longs')
                return 'cover_longs'

        except Exception as e:
            log_error('check_market_reversal', 'SPY', str(e))

        return None

    # ── Hold Period Enforcement ───────────────────────────────────────────────

    def _check_hold_expiry(self, trade: dict):
        """
        Close a position if it has exceeded its maximum allowed hold duration.

        max_hold_days is stored on the trade record at entry time (sourced from
        config: 1 for intraday, 5 for swing, 20 for position trades). Using the
        per-record value rather than re-reading config means the rule applied
        at entry is always the rule enforced at exit, even if config changes.

        Args:
            trade: A trade record dict as returned by Database.get_open_trades().
        """
        entry_time = datetime.fromisoformat(trade['entry_time'])
        days_held = (datetime.now() - entry_time).days

        # Fall back to 5 days (swing default) if the field is missing from
        # legacy records written before max_hold_days was added to the schema
        max_days = trade.get('max_hold_days', 5)

        if days_held >= max_days:
            print(
                f"Position {trade['ticker']} exceeded max hold period "
                f"({days_held} days). Closing."
            )
            try:
                # Place a market close order via Alpaca
                self.executor.close_position(trade['ticker'], trade['trade_type'])

                # Wait for Alpaca to record the fill before querying order history
                time.sleep(2)
                exit_price = self.executor.get_filled_exit_price(trade['ticker'])

                # Update the database record so the dashboard and reports
                # reflect the correct exit reason for post-trade analysis
                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='hold_period_expired',
                    exit_price=exit_price,
                )
            except Exception as e:
                # Log and continue — a failure on one position should not
                # prevent the monitor from checking the remaining positions
                log_error('position_monitor', trade['ticker'], str(e))

    # ── Intraday Exit Logic ───────────────────────────────────────────────────

    def is_intraday_close_time(self) -> bool:
        """
        Returns True after 3:30 PM local time — the window during which all
        intraday positions must be closed before the 4:00 PM market close.

        The 3:45 PM scheduler cycle checks this flag before calling
        close_all_intraday(), giving a 15-minute execution buffer.
        """
        now = datetime.now()
        return now.hour >= 15 and now.minute >= 30

    def close_all_intraday(self):
        """
        Force-close every open intraday position before market close.

        Called by the scheduler on the 3:45 PM cycle when is_intraday_close_time()
        returns True. Intraday positions must never be held overnight — the
        wider gap risk is outside the risk budget defined for this hold tier.

        Hybrid overnight rule: if an intraday position has > 3% unrealized gain
        at this point AND the market regime is BULL, it is upgraded to a swing
        trade (hold_period=swing, max_hold_days=5) instead of being closed. The
        position stays open with its existing stop-loss bracket intact.

        Guard: if config.allow_intraday is False, no intraday positions should
        exist (get_hold_period_safe() upgrades them all to swing at entry).
        The early return here is a safety net for that case.

        Failures are logged individually; remaining intraday positions continue
        to be processed so a single bad close does not leave others open.
        """
        # No intraday positions can exist when PDT protection is active
        if not config.allow_intraday:
            print('⚠️  Intraday disabled — no intraday positions to close')
            return

        open_trades = self.db.get_open_trades()

        # Filter to intraday tier only — swing and position trades are unaffected
        intraday = [t for t in open_trades if t.get('hold_period') == 'intraday']

        # Pre-fetch Alpaca positions and market regime once for the entire batch
        alpaca_positions = {p['ticker']: p for p in self.executor.get_open_positions()}
        from data_collector import DataCollector
        market_regime = DataCollector().get_market_regime()

        for trade in intraday:
            ticker = trade['ticker']

            # Hybrid overnight rule: > 3% gain in bull regime → upgrade to swing
            if market_regime == 'bull':
                alpaca_pos = alpaca_positions.get(ticker)
                entry_price = trade.get('entry_price')
                shares = trade.get('shares') or 0
                if alpaca_pos and entry_price and entry_price > 0 and shares > 0:
                    gain_pct = alpaca_pos['unrealized_pl'] / (entry_price * shares)
                    if gain_pct > 0.03:
                        print(
                            f'🌙 {ticker} upgraded to swing: {gain_pct*100:.2f}% gain '
                            f'in bull regime — holding overnight'
                        )
                        try:
                            self.db.upgrade_trade_to_swing(trade['trade_id'])
                        except Exception as e:
                            log_error('upgrade_to_swing', ticker, str(e))
                        continue  # Skip force-close — position stays open as swing

            try:
                # Determine closing direction from the live Alpaca position side,
                # not the trade record — bracket fills can change the effective side.
                alpaca_pos = alpaca_positions.get(ticker)
                position_side = 'long'  # Default; overridden below when data is available
                if alpaca_pos:
                    side_val = alpaca_pos.get('side')
                    # Alpaca SDK returns PositionSide enum; compare string repr
                    if side_val and str(side_val).lower() in ('short', 'positionside.short'):
                        position_side = 'short'
                    elif float(alpaca_pos.get('qty', 0)) < 0:
                        position_side = 'short'

                print(
                    f'🔴 EOD closing {position_side} position: {ticker} — '
                    f"placing {'sell' if position_side == 'long' else 'buy-to-cover'}"
                )

                # Cancel open bracket legs first to avoid Alpaca error 40310000,
                # then close_position() — Alpaca handles both long (sell) and
                # short (buy-to-cover) correctly from the same API call.
                self.executor._cancel_open_orders(ticker)
                time.sleep(2)  # Wait for cancellation to propagate before placing close
                self.executor.client.close_position(ticker)

                # Wait for Alpaca to record the fill before querying order history
                time.sleep(2)
                exit_price = self.executor.get_filled_exit_price(ticker)

                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='intraday_forced_close',  # Distinct from hold_period_expired
                    exit_price=exit_price,
                )
            except Exception as e:
                log_error('intraday_close', ticker, str(e))
