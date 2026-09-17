# Where to put / change Kalshi API credentials

Secrets are **not** committed. Use these local paths:

| What | Path | Notes |
| --- | --- | --- |
| API Key ID | `kalshi_bot/config.yaml` → `api.api_key_id` | Also mirrored in `data/secrets/api_key_id.txt` |
| RSA private key PEM | `kalshi_bot/data/secrets/kalshi_api.key` | `chmod 600`; referenced by `api.private_key_path` |
| Local config | `kalshi_bot/config.yaml` | Copy from `config.example.yaml` |

## Rotate / replace

1. Kalshi → Account & security → API Keys → revoke old key if it was exposed.
2. Create a new key; download the `.key` file.
3. Overwrite `data/secrets/kalshi_api.key` with the new PEM.
4. Set `api.api_key_id` in `config.yaml` to the new Key ID.
5. Keep mode `paper` until you intentionally enable live in the dashboard.

## Git ignore

Already ignored: `config.yaml`, `data/secrets/`, `*.key`, `.env`.

## Dashboard password (phone / remote access)

| What | Path |
| --- | --- |
| Username | `kalshi` (set in `config.yaml` → `dashboard.username`) |
| Password | `kalshi_bot/data/secrets/dashboard_password.txt` and `config.yaml` → `dashboard.password` |

When exposing via a tunnel, keep `dashboard.host: 0.0.0.0` and always set a password.
