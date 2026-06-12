# CLAUDE.md — Project conventions for Claude Code

## Timezone conventions

Naive timestamps in this codebase and in the `trades` table are **US/Eastern
wall-clock time** — not UTC. The Railway container runs with
`TZ=America/New_York`, so `datetime.now()` returns ET without tzinfo.

**Rules:**
- Never attach UTC to a naive timestamp read from the `trades` table (e.g.
  `dt.replace(tzinfo=timezone.utc)` is wrong for these values).
- To convert a naive DB timestamp to UTC for an Alpaca bar request:
  ```python
  from zoneinfo import ZoneInfo
  ET = ZoneInfo('America/New_York')
  dt_utc = naive_dt.replace(tzinfo=ET).astimezone(timezone.utc)
  ```
- Aware timestamps (strings containing `+00:00`, from `order.filled_at` stored
  via `exit_time_override`) are already UTC and need no conversion.
- Alpaca API **response** bar timestamps are UTC. The existing
  `ts.dt.tz_localize('UTC')` calls on Alpaca bar DataFrames are correct and
  should not be changed.

**Known locations with the wrong UTC assumption (not yet fixed):**
- `mfe_reconstruction.py:107-109` — bar-fetch window is ~4–5 h off for older trades
- `backfill_trades.py:44-45` — `_to_utc()` helper; low impact (relative ordering only)

## Strategy configuration

As of 2026-06-12 the system runs in **gap-fade specialist mode**: momentum is
disabled by config, vwap reversion remains off.

- To re-enable momentum: set `MOMENTUM_ENABLED=true` in `.env` or Railway env vars.
- To re-enable VWAP reversion: set `config.vwap_reversion_enabled = True` in `config.py`
  (no env var backing currently).

Historical and any still-open momentum trades in the DB are fully
readable/processable — exit logic is strategy-agnostic and is not affected by
this flag.
