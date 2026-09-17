# Live promotion criteria (locked before inspecting final eval results)

A strategy may be marked **live_eligible** only if **all** of the following hold on an untouched final evaluation period (chronological walk-forward; train/select on earlier folds only):

1. **Sample size:** ≥ 100 settled binary decisions in the final holdout (or ≥ 50 for monthly economics with explicit caveat).
2. **Calibration:** reliability slope in [0.7, 1.3] across occupied deciles with ≥ 10 obs each, or Brier skill vs climatology ≥ 0.0 with 95% bootstrap CI excluding large negative skill (CI lower bound ≥ −0.02).
3. **Discrimination:** log loss strictly better than market-implied probabilities on the same decision timestamps (same information set), or if market mid missing, better than climatology by CI.
4. **Economics after costs:** net PnL after modeled fees/depth ≥ 0 in holdout is **not** sufficient alone; require lower CI of mean EV per contract ≥ configured `min_net_edge` / 2.
5. **Risk:** max drawdown in holdout ≤ configured `max_drawdown_dollars` when sized with production risk caps.
6. **Execution realism:** backtest used executable asks + depth; if historical depth unavailable, strategy remains **paper-only** until forward paper fills ≥ 30 with reconciliation.
7. **Settlement alignment:** model target matches Kalshi settlement source/station/definition (e.g. CLINYC / Weather Company for KXHIGHNY — not an unverified proxy).
8. **Combos separately:** joint model must pass the same gates on combo settlements; independence assumptions require pre-registered justification and holdout test of dependence residuals.
9. **Multiple testing:** log all model versions/configs tried; promotion uses only pre-registered candidate, not the max over a search without correction.
10. **Human gate:** dashboard one-time live enablement still required; meeting these criteria does not auto-enable live.

**Current status:** weather and CPI models are **research/paper only**. No strategy is live_eligible.
