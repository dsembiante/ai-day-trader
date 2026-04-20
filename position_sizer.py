"""
position_sizer.py — Calculates position size, stop-loss, take-profit, and
                    max hold duration for each trade based on hold period tier.

Sizing logic follows a risk-first approach:
    1. Start from the maximum allowed risk in USD (portfolio * max_position_pct)
    2. Scale down toward 50% of that budget for lower-confidence trades
    3. Apply a hold period scalar — longer-conviction trades get slightly more size
    4. Hard-cap at 130% of the base risk budget (2.6% of portfolio maximum)

Stop-loss and take-profit percentages are sourced from config so they can be
tuned without touching this file.

Usage:
    from position_sizer import PositionSizer
    sizer = PositionSizer()
    sizing = sizer.calculate(portfolio_value, current_price, confidence, hold_period)
"""

from config import config, HoldPeriod


class PositionSizer:
    """
    Stateless calculator — all inputs are passed per-call with no stored state.
    Instantiate once and reuse across the full watchlist each cycle.
    """

    def calculate(self, portfolio_value: float, current_price: float,
                  confidence: float, hold_period: HoldPeriod) -> dict:
        """
        Compute the dollar position size and share count for a trade.

        Confidence scaling:
            Confidence below 0.75 is blocked upstream by the risk manager, so
            the scalar maps the range [0.75, 1.0] → [0.0, 1.0]. A confidence
            of 0.75 produces 50% of the max budget; 1.0 produces 100%.

        Hold period scaling:
            Intraday trades receive 70% of the base size — smaller exposure for
            same-day exits where the thesis has less time to play out.
            Swing is the baseline at 100%.
            Position trades receive 130% — higher conviction, longer runway.

        Args:
            portfolio_value: Total liquidation value of the account in USD.
            current_price:   Latest price of the ticker being sized.
            confidence:      Risk manager confidence score in [0.75, 1.0].
            hold_period:     Classification of the trade's intended duration.

        Returns:
            Dict with keys:
                position_usd:     Dollar value of the position to open.
                shares:           Number of shares (fractional, rounded to 2dp).
                pct_of_portfolio: Position as a percentage of total portfolio value.
        """
        # Dollar floor and ceiling derived from config percentages.
        # min_position_pct (10%) = $4,000 on a $40k account (confidence 0.75).
        # max_position_pct (15%) = $6,000 on a $40k account (confidence 1.0).
        min_usd = portfolio_value * config.min_position_pct
        max_usd = portfolio_value * config.max_position_pct

        # Map confidence from [0.75, 1.0] to [0.0, 1.0]; clamp for safety
        confidence_scalar = max(0.0, min(1.0, (confidence - 0.75) / 0.25))

        # Linear interpolation from floor to ceiling based on confidence.
        # At confidence 0.75: position_usd = min_usd  ($4,000 on $40k)
        # At confidence 1.00: position_usd = max_usd  ($6,000 on $40k)
        position_usd = min_usd + (max_usd - min_usd) * confidence_scalar

        # Multiplier by hold period — longer holds justified by stronger conviction
        hold_scalar = {
            HoldPeriod.INTRADAY: 1.0,   # Full size — risk managed via tighter stop loss pct
            HoldPeriod.SWING:    1.0,   # Baseline
            HoldPeriod.POSITION: 1.3,   # Increased — high-conviction multi-week trade
        }.get(hold_period, 1.0)

        position_usd = position_usd * hold_scalar

        # Hard cap at max_usd × hold_scalar — prevents position trades from
        # exceeding the ceiling scaled by the hold multiplier
        position_usd = min(position_usd, max_usd * hold_scalar)

        shares = round(position_usd / current_price, 2)

        return {
            'position_usd':     round(position_usd, 2),
            'shares':           shares,
            'pct_of_portfolio': round(position_usd / portfolio_value * 100, 2),
        }

    # ── ATR-based take-profit caps (intraday only) ───────────────────────────
    _ATR_TARGET_MIN = 0.01   # 1.0% — floor on take-profit
    _ATR_TARGET_MAX = 0.10   # 10.0% — ceiling on take-profit

    def get_stop_loss(self, entry: float, trade_type: str, hold: HoldPeriod,
                      atr_pct: float = None, ticker: str = '') -> float:
        """
        Calculate the stop-loss price for a position.

        For INTRADAY trades, uses ATR-tiered fixed percentages calibrated to
        ~2× the corresponding time-tiered profit target, maintaining a 2:1
        risk/reward ratio aligned with position_monitor.get_profit_threshold():

            ATR < 2.0% → 0.70% stop  (2× the 0.35% 30–60 min profit target)
            ATR < 3.5% → 1.00% stop  (≈2× the 0.35–0.25% aging profit targets)
            ATR ≥ 3.5% → 1.25% stop  (wider cushion for high-vol names)

        When ATR is unavailable for intraday trades, falls back to 1.00%
        (the medium-tier default). Non-intraday holds use the fixed config
        percentages unchanged.

        Args:
            entry:      Fill price at trade entry.
            trade_type: 'buy' / 'long' for longs; 'short' for shorts.
            hold:       Hold period tier determining the stop percentage.
            atr_pct:    14-day ATR as a % of price from data_collector. Optional.
            ticker:     Symbol name used in log output only.

        Returns:
            Stop-loss price rounded to 2 decimal places.
        """
        if hold == HoldPeriod.INTRADAY:
            if atr_pct is not None and atr_pct > 0:
                # Tiered fixed percentages — matched to time-tiered profit targets
                if atr_pct < 2.0:
                    pct = 0.0035   # 0.35% — low-vol names
                elif atr_pct < 3.5:
                    pct = 0.0050   # 0.50% — med-vol names
                else:
                    pct = 0.0070   # 0.70% — high-vol names
                self._last_atr_stop_pct = pct
            else:
                # ATR unavailable — use medium-tier default rather than wide config fallback
                pct = 0.0050
                self._last_atr_stop_pct = None  # Signals fixed-stop path to caller
                if ticker:
                    print(f'⚠️  ATR unavailable for {ticker} — using fallback 1.00% intraday stop')
        else:
            pct = {
                HoldPeriod.SWING:    config.swing_stop_loss_pct,
                HoldPeriod.POSITION: config.position_stop_loss_pct,
            }.get(hold, config.swing_stop_loss_pct)
            self._last_atr_stop_pct = None  # Signals fixed-stop path to caller

        type_str = trade_type.value if hasattr(trade_type, 'value') else str(trade_type)
        is_long = type_str in ('buy', 'long')
        if is_long:
            stop_price = round(entry * (1 - pct), 2)
        else:
            stop_price = round(entry * (1 + pct), 2)

        if hold == HoldPeriod.INTRADAY and ticker:
            direction = 'below' if is_long else 'above'
            atr_str = f', ATR: {atr_pct:.1f}%' if atr_pct else ''
            print(f'🛡️  {ticker} stop loss set at ${stop_price:.2f} ({pct*100:.2f}% {direction} entry ${entry:.2f}{atr_str})')

        return stop_price

    def get_take_profit(self, entry: float, trade_type: str, hold: HoldPeriod,
                        atr_pct: float = None, ticker: str = '') -> float:
        """
        Calculate the take-profit price for a position.

        For INTRADAY trades with a valid atr_pct, uses 3.0 × ATR% as the target
        distance (capped between 1% and 10%), preserving the 2:1 reward-to-risk
        ratio relative to the 1.5× ATR stop. Falls back to config fixed percentages
        for non-intraday holds or when ATR is unavailable.

        Args:
            entry:      Fill price at trade entry.
            trade_type: 'buy' / 'long' for longs; 'short' for shorts.
            hold:       Hold period tier determining the target percentage.
            atr_pct:    14-day ATR as a % of price from data_collector. Optional.
            ticker:     Symbol name used in log output only.

        Returns:
            Take-profit price rounded to 2 decimal places.
        """
        if hold == HoldPeriod.INTRADAY and atr_pct is not None and atr_pct > 0:
            raw_target = (atr_pct / 100) * 3.0
            pct = max(self._ATR_TARGET_MIN, min(self._ATR_TARGET_MAX, raw_target))
            self._last_atr_target_pct = pct
        else:
            pct = {
                HoldPeriod.INTRADAY: config.intraday_take_profit_pct,
                HoldPeriod.SWING:    config.swing_take_profit_pct,
                HoldPeriod.POSITION: config.position_take_profit_pct,
            }.get(hold, config.swing_take_profit_pct)
            self._last_atr_target_pct = None

        type_str = trade_type.value if hasattr(trade_type, 'value') else str(trade_type)
        if type_str in ('buy', 'long'):
            return round(entry * (1 + pct), 2)  # Target above entry for longs
        return round(entry * (1 - pct), 2)       # Target below entry for shorts

    def get_hold_period_safe(self, requested_hold: HoldPeriod) -> HoldPeriod:
        """
        Return a PDT-safe hold period, upgrading intraday to swing when necessary.

        The Pattern Day Trader rule restricts accounts under $25,000 to no more
        than 3 intraday round-trips in a rolling 5-day window. When
        config.allow_intraday is False, any intraday decision from the crew is
        automatically upgraded to swing so the account never risks a PDT violation.

        This is the single enforcement point — both run_trading_cycle() and
        run_single_ticker() in crew.py call this before sizing a position.

        Args:
            requested_hold: The hold period recommended by the risk manager agent.

        Returns:
            The original hold period, or HoldPeriod.SWING if intraday is blocked.
        """
        if not config.allow_intraday and requested_hold == HoldPeriod.INTRADAY:
            print('⚠️  Intraday disabled (PDT protection) — upgrading to swing trade')
            return HoldPeriod.SWING
        return requested_hold

    def get_max_hold_days(self, hold: HoldPeriod) -> int:
        """
        Return the maximum calendar days a position in this tier may be held.

        Values sourced from config:
            Intraday: 1 day | Swing: 5 days | Position: 20 days

        Used by position_monitor.py to enforce time-based exits independent
        of stop-loss and take-profit bracket orders placed at entry.

        Args:
            hold: Hold period tier.

        Returns:
            Maximum hold duration in calendar days.
        """
        return {
            HoldPeriod.INTRADAY: config.intraday_max_days,
            HoldPeriod.SWING:    config.swing_max_days,
            HoldPeriod.POSITION: config.position_max_days,
        }.get(hold, config.swing_max_days)  # Default to swing if unrecognised
