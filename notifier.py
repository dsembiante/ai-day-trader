"""
notifier.py — Email alert system for critical trading events.

Sends Gmail SMTP alerts for four event types:
    - Circuit breaker trigger (trading halted)
    - Trade placed (entry confirmation)
    - Daily end-of-day P&L summary
    - Data source API failure

Configuration is read from environment variables at module load time.
If ALERT_EMAIL or ALERT_EMAIL_PASSWORD are not set, alerts degrade gracefully
to stdout-only output — the system continues operating without email.

Gmail setup required (one-time):
    1. Enable 2-factor authentication on your Google account
    2. Generate an App Password at myaccount.google.com/apppasswords
    3. Add to .env:
           ALERT_EMAIL=your_email@gmail.com
           ALERT_EMAIL_PASSWORD=your_gmail_app_password
    Note: Use the App Password, NOT your regular Gmail password.

Usage:
    from notifier import alert_circuit_breaker, alert_trade_placed
    alert_circuit_breaker(portfolio_value=95000.0, drawdown=0.11)
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from config import config
from logger import log_error


# ── Credential Loading ────────────────────────────────────────────────────────
# Read at module level so the values are available for all alert functions
# without repeated os.getenv calls. SENDER and RECIPIENT are the same address
# (the agent emails itself) — split into separate variables for future flexibility
# if a different recipient is needed.
SENDER_EMAIL    = os.getenv('ALERT_EMAIL', '')
SENDER_PASSWORD = os.getenv('ALERT_EMAIL_PASSWORD', '')
RECIPIENT_EMAIL = os.getenv('ALERT_EMAIL', '')  # Currently mirrors sender; update to override


# ── Core Mailer ───────────────────────────────────────────────────────────────

def _send_email(subject: str, body: str):
    """
    Send a plain-text email via Gmail SMTP over SSL (port 465).

    Fails silently if credentials are not configured — prints the alert
    to stdout instead so operational visibility is preserved even without email.
    Errors during send are logged to errors.log but do not propagate, ensuring
    an email failure never interrupts the trading cycle.

    Args:
        subject: Email subject line (prefixed with '[AI Trading Agent]' automatically).
        body:    Plain-text email body.
    """
    # Graceful degradation — stdout fallback when email is not configured
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(f'📧 Alert (email not configured): {subject}')
        return

    try:
        msg = MIMEMultipart()
        msg['From']    = SENDER_EMAIL
        msg['To']      = RECIPIENT_EMAIL
        msg['Subject'] = f'[AI Trading Agent] {subject}'
        msg.attach(MIMEText(body, 'plain'))

        # SMTP_SSL on port 465 establishes TLS from the start (vs. STARTTLS on 587)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print(f'📧 Alert sent: {subject}')

    except Exception as e:
        # Log but do not raise — email delivery failure is non-fatal
        log_error('notifier', 'email', str(e))


# ── Alert Functions ───────────────────────────────────────────────────────────

def alert_circuit_breaker(portfolio_value: float, drawdown: float):
    """
    Fire an urgent alert when the circuit breaker halts all trading.

    This is the highest-priority alert — it signals that the portfolio has
    lost ≥10% from its peak and requires immediate manual review before
    trading can resume.

    Args:
        portfolio_value: Current total portfolio value in USD.
        drawdown:        Drawdown fraction (e.g. 0.11 = 11% below peak).
    """
    subject = '🚨 CIRCUIT BREAKER TRIGGERED — Trading Halted'
    body = (
        f'The circuit breaker has been triggered.\n\n'
        f'Portfolio Value: ${portfolio_value:,.2f}\n'
        f'Drawdown from Peak: {drawdown:.1%}\n'
        f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n'
        f'All trading has been halted. Manual review required.\n'
        f'Log in to your Alpaca dashboard to review positions.'
    )
    _send_email(subject, body)


def alert_trade_placed(ticker: str, trade_type: str, amount: float, confidence: float):
    """
    Send a confirmation alert when a trade order is successfully placed.

    Gives the operator real-time awareness of new positions without having
    to actively monitor the dashboard or logs.

    Args:
        ticker:     Symbol of the traded asset (e.g. 'AAPL').
        trade_type: Direction of the trade ('buy', 'sell', 'short', 'cover').
        amount:     Dollar value of the position opened.
        confidence: Risk manager confidence score at the time of execution.
    """
    subject = f'✅ Trade Placed: {trade_type.upper()} {ticker}'
    body = (
        f'A trade has been executed.\n\n'
        f'Ticker: {ticker}\n'
        f'Type: {trade_type.upper()}\n'
        f'Amount: ${amount:,.2f}\n'
        f'Confidence: {confidence:.0%}\n'
        f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )
    _send_email(subject, body)


def alert_daily_summary(total_pnl: float, trades_today: int, portfolio_value: float):
    """
    Send end-of-day P&L summary after market close.

    Called by the scheduler's 4:00 PM end_of_day() job alongside report
    generation. The emoji in the subject line (📈/📉) gives immediate visual
    context when scanning an inbox without opening the message.

    Args:
        total_pnl:       Net realised P&L for the current trading day in USD.
        trades_today:    Number of orders placed during the day's cycles.
        portfolio_value: Closing total portfolio value in USD.
    """
    pnl_emoji = '📈' if total_pnl >= 0 else '📉'
    subject = f'{pnl_emoji} Daily Summary — P&L: ${total_pnl:,.2f}'
    body = (
        f'End of Day Summary\n\n'
        f'Date: {datetime.now().strftime("%Y-%m-%d")}\n'
        f'Portfolio Value: ${portfolio_value:,.2f}\n'
        f'Daily P&L: ${total_pnl:,.2f}\n'
        f'Trades Executed Today: {trades_today}\n'
        f'\nLog in to the dashboard for full details.'
    )
    _send_email(subject, body)


def alert_api_failure(source: str, error: str):
    """
    Notify when a data source becomes unavailable during a collection cycle.

    Sent when a source degrades to False in DataSourceStatus. The body
    explicitly states that trading continues with remaining sources to
    prevent false-alarm concern from the operator.

    Args:
        source: Name of the failed data source (e.g. 'finnhub', 'yfinance').
        error:  Exception message or description of the failure condition.
    """
    subject = f'⚠️ API Failure: {source}'
    body = (
        f'A data source has failed.\n\n'
        f'Source: {source}\n'
        f'Error: {error}\n'
        f'Time: {datetime.now().isoformat()}\n\n'
        f'Trading continues with remaining data sources.'
    )
    _send_email(subject, body)
