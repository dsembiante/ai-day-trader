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
        # Base risk budget — the most we would ever allocate to a single position
        max_risk_usd = portfolio_value * config.max_position_pct

        # Map confidence from [0.75, 1.0] to [0.0, 1.0]; clamp for safety
        confidence_scalar = max(0.0, min(1.0, (confidence - 0.75) / 0.25))

        # Multiplier by hold period — longer holds justified by stronger conviction
        hold_scalar = {
            HoldPeriod.INTRADAY: 1.0,   # Full size — risk managed via tighter stop loss pct
            HoldPeriod.SWING:    1.0,   # Baseline
            HoldPeriod.POSITION: 1.3,   # Increased — high-conviction multi-week trade
        }.get(hold_period, 1.0)

        # Final size: slides from 50% (min confidence) to 100% (max confidence)
        # of the base budget, then scaled by hold period
        position_usd = max_risk_usd * (0.5 + 0.5 * confidence_scalar) * hold_scalar

        # Hard cap at 130% of base budget — prevents position trades from ever
        # exceeding 2.6% of portfolio regardless of confidence score
        position_usd = min(position_usd, max_risk_usd * 1.3)

        shares = round(position_usd / current_price, 2)

        return {
            'position_usd':     round(position_usd, 2),
            'shares':           shares,
            'pct_of_portfolio': round(position_usd / portfolio_value * 100, 2),
        }

    def get_stop_loss(self, entry: float, trade_type: str, hold: HoldPeriod) -> float:
        """
        Calculate the stop-loss price for a position.

        Stop percentages are read from config per hold tier:
            Intraday: 2% | Swing: 5% | Position: 8%

        For long trades the stop is below entry; for short trades the stop is
        above entry to limit loss on an adverse upward price move.

        Args:
            entry:      Fill price at trade entry.
            trade_type: 'buy' / 'long' for longs; 'short' for shorts.
            hold:       Hold period tier determining the stop percentage.

        Returns:
            Stop-loss price rounded to 2 decimal places.
        """
        pct = {
            HoldPeriod.INTRADAY: config.intraday_stop_loss_pct,
            HoldPeriod.SWING:    config.swing_stop_loss_pct,
            HoldPeriod.POSITION: config.position_stop_loss_pct,
        }.get(hold, config.swing_stop_loss_pct)  # Default to swing if unrecognised

        type_str = trade_type.value if hasattr(trade_type, 'value') else str(trade_type)
        if type_str in ('buy', 'long'):
            return round(entry * (1 - pct), 2)  # Stop below entry for longs
        return round(entry * (1 + pct), 2)       # Stop above entry for shorts

    def get_take_profit(self, entry: float, trade_type: str, hold: HoldPeriod) -> float:
        """
        Calculate the take-profit price for a position.

        Take-profit percentages per hold tier:
            Intraday: 4% | Swing: 10% | Position: 20%

        Reward-to-risk is always 2:1 across all tiers by design
        (take-profit % is exactly double the stop-loss %).

        Args:
            entry:      Fill price at trade entry.
            trade_type: 'buy' / 'long' for longs; 'short' for shorts.
            hold:       Hold period tier determining the target percentage.

        Returns:
            Take-profit price rounded to 2 decimal places.
        """
        pct = {
            HoldPeriod.INTRADAY: config.intraday_take_profit_pct,
            HoldPeriod.SWING:    config.swing_take_profit_pct,
            HoldPeriod.POSITION: config.position_take_profit_pct,
        }.get(hold, config.swing_take_profit_pct)

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
