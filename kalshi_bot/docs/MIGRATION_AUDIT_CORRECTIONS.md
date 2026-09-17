# Database migration notes (audit corrections)

## Forward

Opening `Store` runs `CREATE TABLE IF NOT EXISTS` for:

- `reconcile_checkpoints` — last settlement reconcile payload
- `order_intents` — durable intent keyed by `opportunity_id` before exchange submit

No ALTER of existing `orders` / `positions` columns. New semantics live in `details_json`:

- `fill_normalization_version`: `fill_price_norm.v1`
- `settlement_applied`: true after first successful reconcile
- `booked_fill_quantity`: incremental fill booking

## Rollback

1. Check out prior revision.
2. Leave new tables in place (unused) or `DROP TABLE order_intents; DROP TABLE reconcile_checkpoints;` if desired.
3. Do not rewrite historical `avg_price` rows; only new fills use outcome-side acquisition cost.
