# Corrected trade reconciliation (addendum to audit-20260916T204915Z)

Source CSV (unchanged): `data/audit/audit-20260916T204915Z/trade_audit.csv`  
Normalization: `fill_price_norm.v1`  
These rows are **fixture expectations** for regression tests, not live account statements.

| contract | side | audit_csv_pnl | corrected_net_pnl | acquisition_cost | fees_note | status |
|----------|------|---------------|-------------------|-------------------|-----------|--------|
| KXHIGHNY-26SEP15-T70 | yes | -4.900000 | -4.9000 | 0.01 | fee ~0.3176 included in loss | confirmed_fixture |
| KXHIGHLAX-26SEP15-T82 | yes | -4.900000 | -0.6352 | 0.01 in / 0.01 out | 0.3176 × 2 sides | **csv_was_wrong** |
| KXHIGHNY-26SEP16-B77.5 | no | 3.284000 | +2.9822 if fees separate | **0.29** (not 0.71) | fees_paid 0.3018; whether 3.284 already nets fees = **unresolved** | corrected_cost |
| KXHIGHCHI-26SEP16-B75.5 | no | 0.000000 | unresolved (open) | **0.03** (not 0.97) | — | open |
| KXHIGHLAX-26SEP16-T80 | yes | 0.000000 | unresolved (open) | 0.10 | — | open |

Evaluator parity claims from the original narrative: **unsupported / unresolved** — no frozen decision-time feature vector re-score was archived for those fills.
