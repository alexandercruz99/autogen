# Data sources and model notes

## Kalshi exchange

- Public market data does not require auth.
- Trading, balances, RFQs, and MVE market creation require API keys (RSA-PSS).
- Production and demo credentials are **not** interchangeable.
- Rate limits are tiered; the client uses a token bucket and retries on HTTP 429.

## Weather model (`weather.high_temp.v0.1-unvalidated`)

1. Discover open markets for configured series (e.g. `KXHIGHNY`).
2. Parse contract definition from ticker/title (bucket `B79.5` → 79–80°F band; `T82` with `>`/`<` in title).
3. Fetch NWS daytime forecast high for the settlement date at the configured lat/lon.
4. Map forecast ± Gaussian error (σ from config) to P(YES).
5. Shrink toward 0.5 for a conservative probability; require EV after fees and uncertainty buffer.

**Not validated:** σ is not fit on held-out Kalshi settlements yet. Market disagreement is recorded as a benchmark only.

Settlement station / index rules must be verified against each Kalshi market’s official resolution source before live trading. City lat/lon in config is a starting point, not a guarantee of match to Kalshi’s weather index.

## Combos

- Eligible sets come from `GET /multivariate_event_collections`.
- Pricing uses RFQ quotes; missing quotes ⇒ skip (never invent a combo price).
- Settlement payout = product of underlying leg settlement values; DNP may yield a scalar ≠ 0/1 and is **not** an automatic refund.
- Dependent legs without a joint model are skipped.

## Categories not yet modeled

Sports, player props, economics, politics, and other series are discoverable but evaluated as **skip: no validated model**.
