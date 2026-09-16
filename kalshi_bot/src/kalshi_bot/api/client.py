from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_bot.config import ApiConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, tokens_per_second: float, burst: float | None = None) -> None:
        self.rate = max(tokens_per_second, 0.1)
        self.capacity = burst or max(self.rate * 2, 1.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, cost: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
                need = (cost - self.tokens) / self.rate
            time.sleep(min(need, 1.0))


class KalshiClient:
    """Kalshi Trade API client (public + optional RSA-PSS authenticated calls)."""

    def __init__(self, config: ApiConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._private_key: rsa.RSAPrivateKey | None = None
        self._limiter = RateLimiter(config.rate_limit_tokens_per_second)
        self._client = httpx.Client(timeout=config.request_timeout_seconds)
        if config.private_key_path and config.api_key_id:
            self._load_private_key(config.private_key_path)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def authenticated(self) -> bool:
        return self._private_key is not None and bool(self.config.api_key_id)

    def _load_private_key(self, path: str) -> None:
        key_path = Path(path).expanduser()
        if not key_path.exists():
            raise FileNotFoundError(
                f"Private key not found at {key_path}. "
                "Create an API key in Kalshi and set api.private_key_path locally."
            )
        data = key_path.read_bytes()
        key = serialization.load_pem_private_key(data, password=None, backend=default_backend())
        if not isinstance(key, rsa.RSAPrivateKey):
            raise TypeError("Kalshi API key must be an RSA private key")
        self._private_key = key
        logger.info("Loaded Kalshi API private key from %s (path only; key material not logged)", key_path)

    def _sign(self, timestamp_ms: str, method: str, full_path: str) -> str:
        if self._private_key is None:
            raise RuntimeError("Authenticated request requires a loaded private key")
        path_without_query = full_path.split("?", 1)[0]
        message = f"{timestamp_ms}{method.upper()}{path_without_query}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _sign_path(self, url: str) -> str:
        return urlparse(url).path

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        auth: bool = False,
        token_cost: float = 1.0,
    ) -> Any:
        url = self._url(path)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth:
            if not self.authenticated:
                raise RuntimeError(
                    "Authenticated endpoint requested but API credentials are not configured. "
                    "Set api.api_key_id and api.private_key_path in local config (never paste keys in chat)."
                )
            ts = str(int(time.time() * 1000))
            headers.update(
                {
                    "KALSHI-ACCESS-KEY": self.config.api_key_id,
                    "KALSHI-ACCESS-TIMESTAMP": ts,
                    "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, self._sign_path(url)),
                }
            )

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._limiter.acquire(token_cost)
            try:
                resp = self._client.request(method.upper(), url, params=params, json=json_body, headers=headers)
                if resp.status_code == 429:
                    sleep_for = min(2**attempt, 16)
                    logger.warning("Rate limited (429); backing off %ss", sleep_for)
                    time.sleep(sleep_for)
                    continue
                if resp.status_code >= 500:
                    sleep_for = min(2**attempt, 16)
                    logger.warning("Server error %s; retrying in %ss", resp.status_code, sleep_for)
                    time.sleep(sleep_for)
                    continue
                if resp.status_code >= 400:
                    # Redact potential secrets from error bodies in logs.
                    body = resp.text[:500]
                    raise httpx.HTTPStatusError(
                        f"Kalshi API {resp.status_code}: {body}",
                        request=resp.request,
                        response=resp,
                    )
                if resp.status_code == 204 or not resp.content:
                    return {}
                return resp.json()
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Kalshi request failed after retries: {last_exc}")

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, json_body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.request("POST", path, json_body=json_body, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    def put(self, path: str, json_body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.request("PUT", path, json_body=json_body, **kwargs)

    # ---- Public market data ----

    def get_exchange_status(self) -> dict[str, Any]:
        return self.get("/exchange/status")

    def get_markets(
        self,
        *,
        status: str | None = "open",
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        return self.get("/markets", params=params)

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self.get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int | None = None) -> dict[str, Any]:
        params = {"depth": depth} if depth else None
        return self.get(f"/markets/{ticker}/orderbook", params=params)

    def get_events(
        self,
        *,
        status: str | None = "open",
        series_ticker: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
        with_nested_markets: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "with_nested_markets": with_nested_markets}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self.get("/events", params=params)

    def get_series_list(self, *, category: str | None = None, limit: int = 200) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": min(limit, 200)}
        if category:
            params["category"] = category
        return self.get("/series", params=params)

    def get_multivariate_collections(
        self,
        *,
        status: str | None = "open",
        limit: int = 100,
        cursor: str | None = None,
        series_ticker: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self.get("/multivariate_event_collections", params=params)

    def get_multivariate_collection(self, collection_ticker: str) -> dict[str, Any]:
        return self.get(f"/multivariate_event_collections/{collection_ticker}")

    def create_mve_market(self, collection_ticker: str, selected_markets: list[dict[str, str]]) -> dict[str, Any]:
        """Create/lookup a combo market in a multivariate collection (auth required)."""
        return self.post(
            f"/multivariate_event_collections/{collection_ticker}",
            json_body={"selected_markets": selected_markets},
            auth=True,
        )

    def get_weather_index(self, city: str) -> dict[str, Any]:
        return self.get(f"/live_data/weather_index/{city}")

    # ---- Authenticated portfolio / orders / RFQ ----

    def get_balance(self) -> dict[str, Any]:
        return self.get("/portfolio/balance", auth=True)

    def get_positions(self, **params: Any) -> dict[str, Any]:
        return self.get("/portfolio/positions", params=params or None, auth=True)

    def get_orders(self, **params: Any) -> dict[str, Any]:
        return self.get("/portfolio/orders", params=params or None, auth=True)

    def get_fills(self, **params: Any) -> dict[str, Any]:
        return self.get("/portfolio/fills", params=params or None, auth=True)

    def create_order_v2(self, body: dict[str, Any]) -> dict[str, Any]:
        if "client_order_id" not in body:
            body = {**body, "client_order_id": str(uuid.uuid4())}
        return self.post("/portfolio/events/orders", json_body=body, auth=True)

    def cancel_order_v2(self, order_id: str, market_ticker: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if market_ticker:
            body["market_ticker"] = market_ticker
            body["exchange_index"] = -1
        return self.delete(f"/portfolio/events/orders/{order_id}", auth=True)  # path-style fallback

    def cancel_all_orders(self) -> dict[str, Any]:
        return self.delete("/portfolio/orders", auth=True)

    def create_rfq(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.post("/communications/rfqs", json_body=body, auth=True)

    def get_rfqs(self, **params: Any) -> dict[str, Any]:
        return self.get("/communications/rfqs", params=params or None, auth=True)

    def get_quotes(self, rfq_id: str | None = None, **params: Any) -> dict[str, Any]:
        p = dict(params)
        if rfq_id:
            p["rfq_id"] = rfq_id
        return self.get("/communications/quotes", params=p or None, auth=True)

    def accept_quote(self, rfq_id: str, quote_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self.put(
            f"/communications/rfqs/{rfq_id}/quotes/{quote_id}/accept",
            json_body=body,
            auth=True,
        )

    def delete_rfq(self, rfq_id: str) -> dict[str, Any]:
        return self.delete(f"/communications/rfqs/{rfq_id}", auth=True)
