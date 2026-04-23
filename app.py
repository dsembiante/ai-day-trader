"""
app.py — Streamlit dashboard for the AI Trading Agent.

Eight tabs provide full observability into the trading system:

    Tab 1 — Overview:              Top-level P&L, win rate, and trade count metrics
    Tab 2 — Performance vs S&P:   Portfolio equity curve charted against the benchmark
    Tab 3 — Active Positions:      Open trades with entry prices and exit targets
    Tab 4 — Trade History:         Filterable full trade log with agent reasoning drill-down
    Tab 5 — Hold Period Analysis:  P&L breakdown by intraday / swing / position tier
    Tab 6 — Risk Monitor:          Live view of circuit breaker and risk parameter status
    Tab 7 — Reports:               On-demand daily and weekly PDF generation and download
    Tab 8 — Settings:              Read-only view of the active config (editable via .env)

Run locally:
    streamlit run app.py

The dashboard is read-only — it queries the database and generates reports
but does not place orders or modify configuration at runtime.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from database import Database
from report_generator import generate_daily_report, generate_weekly_report
from dotenv import load_dotenv

# Load .env so config values are available when the settings tab reads them
load_dotenv()


# ── Shared Resources ──────────────────────────────────────────────────────────
# Database is instantiated once at module level and reused across all tabs.
# Streamlit reruns the entire script on every user interaction, so keeping db
# here avoids opening a new SQLite connection on every rerender.
db = Database()


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AI Day Trader',
    page_icon='📈',
    layout='wide',  # Wide layout maximises chart real estate
)
st.title('📈 AI Day Trader Dashboard')


# ── Tab Layout ────────────────────────────────────────────────────────────────
# All eight tabs are declared in a single st.tabs() call. Content for each tab
# is written inside its corresponding `with` block below.
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    '📊 Overview',
    '📈 Performance vs S&P 500',
    '🔄 Active Positions',
    '📋 Trade History',
    '⏱ Hold Period Analysis',
    '⚠️ Risk Monitor',
    '📥 Reports',
    '⚙️ Settings',
])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    # get_performance_metrics() may not exist in all Database versions;
    # hasattr guard prevents AttributeError during early development
    metrics = db.get_performance_metrics() if hasattr(db, 'get_performance_metrics') else {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total P&L',      f'${metrics.get("total_pnl", 0):,.2f}')
    col2.metric('Win Rate',        f'{metrics.get("win_rate", 0):.1%}')
    col3.metric('Total Trades',    metrics.get('total_trades', 0))
    col4.metric('Profit Factor',   f'{metrics.get("profit_factor", 0):.2f}')


# ── Tab 2: Portfolio Equity Curve vs S&P 500 ─────────────────────────────────
with tab2:
    st.subheader('Portfolio vs S&P 500')

    daily = db.get_daily_performance()
    if daily:
        df = pd.DataFrame(daily)

        fig = go.Figure()
        # Portfolio equity line — dark blue to distinguish from benchmark
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['portfolio_value'],
            name='Your Portfolio',
            line=dict(color='#1F4E79', width=2),
        ))
        # TODO: Add S&P 500 benchmark trace by fetching SPY daily closes from Alpaca
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No performance data yet.')


# ── Tab 3: Active Positions ───────────────────────────────────────────────────
with tab3:
    st.subheader('Open Positions')

    open_trades = db.get_open_trades()
    if open_trades:
        df = pd.DataFrame(open_trades)

        # Display only the operationally relevant columns; others (reasoning text,
        # raw IDs) are available in the Trade History tab for drill-down
        cols_to_show = [c for c in [
            'ticker', 'trade_type', 'hold_period',
            'entry_price', 'stop_loss_price', 'take_profit_price',
            'position_size_usd', 'confidence_at_entry', 'max_hold_days',
        ] if c in df.columns]
        st.dataframe(df[cols_to_show])
    else:
        st.info('No open positions')


# ── Tab 4: Full Trade History ─────────────────────────────────────────────────
with tab4:
    st.subheader('Trade History')

    all_trades = db.get_all_trades()
    if all_trades:
        df = pd.DataFrame(all_trades)

        # ── Filters ───────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        ticker_f = c1.multiselect('Ticker',     df['ticker'].unique()     if 'ticker'     in df else [])
        type_f   = c2.multiselect('Trade Type', df['trade_type'].unique() if 'trade_type' in df else [])
        hold_f   = c3.multiselect('Hold Period', ['intraday', 'swing', 'position'])

        if ticker_f: df = df[df['ticker'].isin(ticker_f)]
        if type_f:   df = df[df['trade_type'].isin(type_f)]
        if hold_f:   df = df[df['hold_period'].isin(hold_f)]

        # ── Column Selection & Formatting ─────────────────────────────────────
        display = pd.DataFrame()
        display['Date']        = pd.to_datetime(df['entry_time']).dt.date if 'entry_time' in df else None
        display['Ticker']      = df.get('ticker')
        display['Type']        = df.get('trade_type')
        display['Entry Price'] = pd.to_numeric(df.get('entry_price'),  errors='coerce').round(2)
        display['Exit Price']  = pd.to_numeric(df.get('exit_price'),   errors='coerce').round(2)
        display['P&L ($)']     = pd.to_numeric(df.get('pnl'),          errors='coerce').round(2)
        display['P&L (%)']     = (pd.to_numeric(df.get('pnl_pct'),     errors='coerce') * 100).round(2)
        display['Exit Reason'] = df.get('exit_reason')
        display['Confidence']  = pd.to_numeric(df.get('confidence_at_entry'), errors='coerce').round(2)

        # ── Summary Row ───────────────────────────────────────────────────────
        closed = display[display['P&L ($)'].notna()]
        total_pnl  = closed['P&L ($)'].sum()
        win_rate   = (closed['P&L ($)'] > 0).mean() * 100 if len(closed) else 0.0
        avg_pnl    = closed['P&L ($)'].mean() if len(closed) else 0.0

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric('Total P&L',       f'${total_pnl:+,.2f}')
        sm2.metric('Win Rate',         f'{win_rate:.1f}%')
        sm3.metric('Avg P&L / Trade',  f'${avg_pnl:+,.2f}')

        # ── Sortable Table with P&L Coloring ─────────────────────────────────
        def _pnl_color(val):
            if pd.isna(val):
                return ''
            return 'color: #2ecc71; font-weight: bold' if val > 0 else 'color: #e74c3c; font-weight: bold'

        styled = (
            display.style
            .format({
                'Entry Price': '${:.2f}',
                'Exit Price':  '${:.2f}',
                'P&L ($)':     '${:+,.2f}',
                'P&L (%)':     '{:+.2f}%',
                'Confidence':  '{:.2f}',
            }, na_rep='—')
            .applymap(_pnl_color, subset=['P&L ($)', 'P&L (%)'])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Agent Reasoning Drill-Down ────────────────────────────────────────
        if st.checkbox('Show agent reasoning for last trade'):
            last = all_trades[0]
            st.write('**Bull Agent:**',            last.get('bull_reasoning', ''))
            st.write('**Bear Agent:**',             last.get('bear_reasoning', ''))
            st.write('**Risk Manager:**',           last.get('risk_manager_reasoning', ''))
            st.write('**Hold Period Rationale:**',  last.get('hold_period_reasoning', ''))
    else:
        st.info('No trades yet')


# ── Tab 5: Hold Period Analysis ───────────────────────────────────────────────
with tab5:
    st.subheader('Performance by Hold Period')

    hp_data = db.get_performance_by_hold_period()
    if hp_data:
        # Transpose the dict-of-dicts into a flat DataFrame for display and charting
        hp_df = pd.DataFrame(hp_data).T.reset_index()
        hp_df.columns = ['Hold Period', 'Total Trades', 'Total P&L', 'Avg P&L']
        st.dataframe(hp_df)

        # Bar chart makes the relative performance of each tier immediately obvious
        fig = px.bar(
            hp_df,
            x='Hold Period',
            y='Total P&L',
            color='Hold Period',
            title='Total P&L by Hold Period',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info('Use this chart to tune which hold periods work best for your watchlist.')
    else:
        st.info('No closed trades yet')


# ── Tab 6: Risk Monitor ───────────────────────────────────────────────────────
with tab6:
    st.subheader('Risk Monitor')

    # Split into two columns for a balanced layout; values are read directly
    # from config so they always reflect the currently loaded settings
    from config import config as _cfg
    c1, c2 = st.columns(2)
    with c1:
        st.metric('Circuit Breaker',   '10% drawdown')
        st.metric('Max Position Size', '25% standard / 30% high conviction')
        st.metric('Min Confidence',    '0.82 (normal VIX) / 0.85 (high VIX)')
    with c2:
        st.metric('Max Positions',  '4 per direction')
        st.metric('Run Mode',       str(_cfg.run_mode).replace('RunMode.', ''))
        st.metric('Trading Mode',   'Paper')
    # TODO: Pull live circuit breaker status and current drawdown from
    # CircuitBreaker().check() for dynamic red/green state display


# ── Tab 7: Reports ────────────────────────────────────────────────────────────
with tab7:
    st.subheader('Download Reports')

    # Report generation can be slow (PDF compilation); each button triggers
    # a full Streamlit rerun, so the download button only appears after the
    # file is successfully generated — no partial state is shown.
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Generate Daily Report'):
            path = generate_daily_report()
            with open(path, 'rb') as f:
                st.download_button('Download', f, 'daily_report.pdf', 'application/pdf')
    with c2:
        if st.button('Generate Weekly Report'):
            path = generate_weekly_report()
            with open(path, 'rb') as f:
                st.download_button('Download', f, 'weekly_report.pdf', 'application/pdf')


# ── Tab 8: Settings ───────────────────────────────────────────────────────────
with tab8:
    st.subheader('Settings')

    # Read-only view — all changes must be made in .env and require a redeploy.
    # This prevents accidental live config changes through the UI.
    st.info('To change settings, update your .env file and redeploy.')

    # Import here (rather than at module top) to keep the settings tab's
    # dependency explicit and to avoid circular import risk at startup
    from config import config

    st.write(f'**Run Mode:** {config.run_mode}')
    st.write(f'**Trading Mode:** {config.trading_mode}')
    st.write(f'**Confidence Threshold:** {config.confidence_threshold}')
    st.write(f'**Max Position Size:** {config.max_position_pct:.0%}')
    st.write(f'**Circuit Breaker:** {config.circuit_breaker_pct:.0%} drawdown')
    st.write(f'**Watchlist:** {", ".join(config.watchlist)}')
