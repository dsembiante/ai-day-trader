"""
report_generator.py — Generates daily, weekly, and monthly PDF trading reports.

Reports are produced using ReportLab and written to the reports/ directory tree:
    reports/daily/    — daily_{YYYY-MM-DD}.pdf
    reports/weekly/   — weekly_{YYYY-W##}.pdf
    reports/monthly/  — monthly_{YYYY-MM}.pdf

Each report contains:
    - Performance summary (total P&L, win rate, open positions)
    - Breakdown by hold period tier (intraday / swing / position)
    - Trade table showing up to 20 trades for the period

All three report types share the same _generate_report() core builder so the
structure and styling are consistent. Period filtering is the only variation.

Usage:
    from report_generator import generate_daily_report, generate_weekly_report
    path = generate_daily_report()
    # path is the absolute file path to the generated PDF
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle,
)
from reportlab.lib.units import inch
from reportlab.lib import colors
from database import Database
from config import config
from datetime import datetime, timedelta
import os


# ── Brand Colours ─────────────────────────────────────────────────────────────
# Defined once at module level and referenced throughout _get_styles()
# so rebranding requires only changes in this block.
BLUE   = HexColor('#1F4E79')   # Primary — headers, rule lines
ORANGE = HexColor('#C55A11')   # Section headings
GRAY   = HexColor('#666666')   # Secondary metadata text
GREEN  = HexColor('#1E6B3C')   # Positive P&L values
RED    = HexColor('#C00000')   # Negative P&L values


# ── Module-Level Database Instance ────────────────────────────────────────────
# Shared across all report functions within the process. Reports are generated
# infrequently (at most once per day per type), so a single connection is safe.
db = Database()


# ── Style Definitions ─────────────────────────────────────────────────────────

def _get_styles() -> dict:
    """
    Build and return a dict of named ParagraphStyle objects.

    Styles are derived from ReportLab's built-in stylesheet so that base
    font metrics (line height, paragraph spacing) are inherited consistently.
    Only the overrides specific to this report's design are applied.

    Returns:
        Dict mapping style names to ParagraphStyle instances.
    """
    styles = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('Title',
            parent=styles['Title'],
            textColor=BLUE, fontSize=20, spaceAfter=12),

        'h2': ParagraphStyle('H2',
            parent=styles['Heading2'],
            textColor=ORANGE, fontSize=14, spaceBefore=16, spaceAfter=8),

        'body': ParagraphStyle('Body',
            parent=styles['Normal'],
            fontSize=11, spaceAfter=6, leading=16),

        'meta': ParagraphStyle('Meta',
            parent=styles['Normal'],
            fontSize=9, textColor=GRAY, spaceAfter=4),

        # Conditional styles — applied based on P&L sign
        'green': ParagraphStyle('Green',
            parent=styles['Normal'],
            fontSize=11, textColor=GREEN),
        'red': ParagraphStyle('Red',
            parent=styles['Normal'],
            fontSize=11, textColor=RED),
    }


# ── Trade Table Builder ───────────────────────────────────────────────────────

def _build_trades_table(trades: list, styles: dict) -> list:
    """
    Build a ReportLab Table flowable from a list of trade dicts.

    Capped at 20 rows to keep PDF pages manageable — the full trade log
    is always available in the Streamlit dashboard. Open trades show 'Open'
    in the P&L and Exit columns rather than null values.

    Args:
        trades: List of trade record dicts as returned by Database.get_all_trades().
        styles: Style dict from _get_styles().

    Returns:
        List of ReportLab flowables to extend into the story list.
    """
    if not trades:
        return [Paragraph('No trades in this period.', styles['body'])]

    story = []

    # Build header row + one data row per trade (max 20)
    headers = ['Ticker', 'Type', 'Hold', 'Entry', 'Exit', 'P&L', 'Confidence']
    data = [headers]

    for t in trades[:20]:  # Cap at 20 rows to avoid multi-page tables
        pnl = t.get('pnl')
        pnl_str = f"${pnl:,.2f}" if pnl is not None else 'Open'
        data.append([
            t.get('ticker', ''),
            t.get('trade_type', ''),
            t.get('hold_period', ''),
            f"${t.get('entry_price', 0):,.2f}",
            f"${t.get('exit_price', 0):,.2f}" if t.get('exit_price') else 'Open',
            pnl_str,
            f"{t.get('confidence_at_entry', 0):.0%}",
        ])

    # Fixed column widths sum to ~6.9 inches — fits within 7.5" content width
    table = Table(data, colWidths=[
        0.8 * inch,   # Ticker
        0.7 * inch,   # Type
        0.8 * inch,   # Hold
        0.9 * inch,   # Entry
        0.9 * inch,   # Exit
        0.9 * inch,   # P&L
        0.9 * inch,   # Confidence
    ])

    table.setStyle(TableStyle([
        # Header row — navy background with white text
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        # Alternating row shading to improve readability
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    story.append(table)
    return story


# ── Core Report Builder ───────────────────────────────────────────────────────

def _generate_report(title: str, period_label: str, trades: list, filename: str) -> str:
    """
    Compile and write a PDF report for a given set of trades.

    All three public report functions (daily, weekly, monthly) delegate here.
    The period_label and trades list are the only meaningful differences between them.

    Structure:
        1. Title + metadata header with horizontal rule
        2. Performance summary (total P&L, win rate, open count)
        3. P&L breakdown by hold period tier
        4. Trade table (up to 20 trades)

    Args:
        title:        Report title string displayed at the top of the PDF.
        period_label: Human-readable date range (e.g. "2026-03-23" or "Mar 16 - Mar 23, 2026").
        trades:       Trade records to include in this report (pre-filtered by caller).
        filename:     Absolute output path for the PDF file.

    Returns:
        The filename path, so the caller (app.py) can open it for download.
    """
    # Ensure the output directory exists — report type subdirs may not exist yet
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    doc = SimpleDocTemplate(
        filename, pagesize=letter,
        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72,
    )
    styles = _get_styles()
    story  = []

    # Pull performance aggregates — used across multiple sections
    metrics  = db.get_performance_metrics()
    hp_data  = db.get_performance_by_hold_period()

    # Compute period-specific metrics from the filtered trades list
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    total_pnl     = sum(t.get('pnl', 0) or 0 for t in closed_trades)
    wins          = [t for t in closed_trades if (t.get('pnl') or 0) > 0]
    win_rate      = len(wins) / len(closed_trades) if closed_trades else 0

    # ── Section 1: Header ─────────────────────────────────────────────────────
    story.append(Paragraph(title, styles['title']))
    story.append(Paragraph(
        f'Period: {period_label} | Generated: {datetime.now().strftime("%B %d, %Y %H:%M")}',
        styles['meta'],
    ))
    story.append(HRFlowable(width='100%', thickness=2, color=BLUE, spaceAfter=16))

    # ── Section 2: Performance Summary ───────────────────────────────────────
    story.append(Paragraph('Performance Summary', styles['h2']))

    # Apply green or red style based on P&L sign for immediate visual clarity
    pnl_style = styles['green'] if total_pnl >= 0 else styles['red']
    story.append(Paragraph(f'Total P&L: ${total_pnl:,.2f}', pnl_style))
    story.append(Paragraph(
        f'Win Rate: {win_rate:.1%} ({len(wins)} wins / {len(closed_trades)} closed trades)',
        styles['body'],
    ))
    story.append(Paragraph(
        f'Open Positions: {len([t for t in trades if t.get("status") == "open"])}',
        styles['body'],
    ))
    story.append(Spacer(1, 12))

    # ── Section 3: Hold Period Breakdown ──────────────────────────────────────
    # Helps identify which tier (intraday / swing / position) is driving returns
    story.append(Paragraph('Performance by Hold Period', styles['h2']))
    for hp, data in hp_data.items():
        story.append(Paragraph(
            f'{hp.title()}: {data["count"]} trades | '
            f'Total P&L: ${data["total_pnl"]:,.2f} | '
            f'Avg P&L: ${data["avg_pnl"]:,.2f}',
            styles['body'],
        ))
    story.append(Spacer(1, 12))

    # ── Section 4: Trade Table ────────────────────────────────────────────────
    story.append(Paragraph(f'Trades This Period ({len(trades)} total)', styles['h2']))
    story.extend(_build_trades_table(trades, styles))

    doc.build(story)
    return filename


# ── Public Report Functions ───────────────────────────────────────────────────

def generate_daily_report() -> str:
    """
    Generate a PDF report for all trades entered today.

    Filters by entry_time prefix (YYYY-MM-DD) rather than a range query
    so trades entered at any time during the current calendar day are included.

    Returns:
        Absolute path to the generated PDF file.
    """
    today       = datetime.now().strftime('%Y-%m-%d')
    all_trades  = db.get_all_trades()
    today_trades = [t for t in all_trades if t.get('entry_time', '').startswith(today)]
    filename    = os.path.join(config.reports_dir, 'daily', f'daily_{today}.pdf')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return _generate_report('Daily Trading Report', today, today_trades, filename)


def generate_weekly_report() -> str:
    """
    Generate a PDF report for all trades entered in the last 7 calendar days.

    Uses ISO string comparison on entry_time (which is stored as ISO-8601)
    so no date parsing is required for the filter.

    Returns:
        Absolute path to the generated PDF file.
    """
    week_ago     = (datetime.now() - timedelta(days=7)).isoformat()
    all_trades   = db.get_all_trades()
    week_trades  = [t for t in all_trades if t.get('entry_time', '') >= week_ago]
    week_label   = (
        f"{(datetime.now() - timedelta(days=7)).strftime('%b %d')} - "
        f"{datetime.now().strftime('%b %d, %Y')}"
    )
    # Week number in filename ensures one file per ISO week — overwrites if re-generated
    filename = os.path.join(
        config.reports_dir, 'weekly',
        f'weekly_{datetime.now().strftime("%Y-W%U")}.pdf',
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return _generate_report('Weekly Trading Report', week_label, week_trades, filename)


def generate_monthly_report() -> str:
    """
    Generate a PDF report for all trades entered in the current calendar month.

    month_start is set to the first day of the current month at midnight,
    so all trades from day 1 onward are captured.

    Returns:
        Absolute path to the generated PDF file.
    """
    month_start  = datetime.now().replace(day=1).isoformat()
    all_trades   = db.get_all_trades()
    month_trades = [t for t in all_trades if t.get('entry_time', '') >= month_start]
    month_label  = datetime.now().strftime('%B %Y')
    filename     = os.path.join(
        config.reports_dir, 'monthly',
        f'monthly_{datetime.now().strftime("%Y-%m")}.pdf',
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return _generate_report('Monthly Trading Report', month_label, month_trades, filename)
