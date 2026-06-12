# CHANGES

## 2026-06-12 — Era boundary: V1 becomes gap-fade specialist

Momentum strategy disabled by config (`MOMENTUM_ENABLED` env var, default `false`).
System now operates as a gap-fade specialist. VWAP reversion remains off.
All momentum code intact and re-enableable by setting `MOMENTUM_ENABLED=true`.
