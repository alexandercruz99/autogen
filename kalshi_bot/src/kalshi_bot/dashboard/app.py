from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from kalshi_bot.bot.loop import BotLoop
from kalshi_bot.bot.pipeline import TradingPipeline
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import Store
from kalshi_bot.money import D
from kalshi_bot.risk.limits import RiskManager


TEMPLATES_DIR = Path(__file__).parent / "templates"
security = HTTPBasic(auto_error=False)


def create_app(
    config: AppConfig,
    store: Store,
    pipeline: TradingPipeline,
    loop: BotLoop | None = None,
) -> FastAPI:
    app = FastAPI(title="Kalshi Trading Bot", version="0.1.0")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    risk = RiskManager(store, config.trading)

    def require_dashboard_auth(
        credentials: HTTPBasicCredentials | None = Depends(security),
    ) -> None:
        expected_user = config.dashboard.username or "kalshi"
        expected_pass = config.dashboard.password or ""
        if not expected_pass:
            # No password configured — allow (intended for localhost-only binds).
            return
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": 'Basic realm="Kalshi Bot"'},
            )
        user_ok = secrets.compare_digest(credentials.username, expected_user)
        pass_ok = secrets.compare_digest(credentials.password, expected_pass)
        if not (user_ok and pass_ok):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": 'Basic realm="Kalshi Bot"'},
            )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        expected_pass = config.dashboard.password or ""
        if expected_pass:
            auth = request.headers.get("Authorization")
            ok = False
            if auth and auth.startswith("Basic "):
                import base64

                try:
                    raw = base64.b64decode(auth.split(" ", 1)[1]).decode()
                    user, pwd = raw.split(":", 1)
                    ok = secrets.compare_digest(user, config.dashboard.username or "kalshi") and secrets.compare_digest(
                        pwd, expected_pass
                    )
                except Exception:
                    ok = False
            if not ok:
                from fastapi.responses import Response

                return Response(
                    content="Authentication required",
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Kalshi Bot"'},
                )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        state = store.get_state()
        snap = risk.portfolio_snapshot()
        opps = store.list_opportunities(limit=100)
        orders = store.list_orders(limit=50)
        positions = store.list_positions()
        audit = store.list_audit(limit=30)
        validation = pipeline.validation_status()
        feeds = {}
        research_forecast = None
        try:
            from kalshi_bot.models.weather.ops import weather_feeds_status

            feeds = weather_feeds_status(config)
        except Exception as exc:
            feeds = {"error": str(exc)}
        try:
            import json
            from pathlib import Path

            p = Path("data/obs_engine/feeds/latest_prediction.json")
            if p.exists():
                research_forecast = json.loads(p.read_text())
        except Exception:
            research_forecast = None
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "state": state,
                "snap": {k: str(v) for k, v in snap.items() if not k.startswith("event::")},
                "opps": opps,
                "orders": orders,
                "positions": positions,
                "audit": audit,
                "config_mode": config.mode,
                "live_config_enabled": config.live.enabled,
                "validation": validation,
                "feeds": feeds,
                "research_forecast": research_forecast,
            },
        )

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        state = store.get_state()
        snap = risk.portfolio_snapshot()
        return {
            "mode": state.mode,
            "live_enabled": state.live_enabled,
            "pause_buying": state.pause_buying,
            "kill_switch": state.kill_switch,
            "connection_ok": state.connection_ok,
            "last_scan_at": state.last_scan_at,
            "last_scan_ok": state.last_scan_ok,
            "last_error": state.last_error,
            "portfolio": {k: str(v) for k, v in snap.items()},
        }

    @app.get("/api/opportunities")
    def api_opportunities() -> list[dict[str, Any]]:
        return store.list_opportunities(limit=200)

    @app.post("/actions/pause")
    def pause() -> RedirectResponse:
        store.update_state(pause_buying=True)
        store.audit("pause", "buying paused")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/resume")
    def resume() -> RedirectResponse:
        store.update_state(pause_buying=False)
        store.audit("resume", "buying resumed")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/kill")
    def kill() -> RedirectResponse:
        store.update_state(kill_switch=True, pause_buying=True)
        n = pipeline.execution.cancel_outstanding()
        store.audit("kill_switch", f"activated; canceled {n} orders", level="warning")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/unkill")
    def unkill() -> RedirectResponse:
        store.update_state(kill_switch=False)
        store.audit("kill_switch", "cleared")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/cancel-orders")
    def cancel_orders() -> RedirectResponse:
        n = pipeline.execution.cancel_outstanding()
        store.audit("cancel_orders", f"canceled {n}")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/scan")
    def scan_now() -> RedirectResponse:
        pipeline.run_scan_once()
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/set-mode")
    def set_mode(mode: str = Form(...)) -> RedirectResponse:
        if mode not in ("research", "paper", "live"):
            return RedirectResponse("/", status_code=303)
        state = store.get_state()
        if mode == "live" and not state.live_enabled:
            store.audit("mode_block", "cannot switch to live until Enable live trading")
            return RedirectResponse("/?err=live_not_enabled", status_code=303)
        store.update_state(mode=mode)
        store.audit("mode", f"set to {mode}")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/enable-live")
    def enable_live(
        budget: str = Form(...),
        max_loss: str = Form(...),
        confirm: str = Form(...),
    ) -> RedirectResponse:
        if confirm.strip().upper() != "ENABLE LIVE":
            store.audit("live_enable_rejected", "confirmation phrase mismatch")
            return RedirectResponse("/?err=confirm", status_code=303)
        try:
            budget_d = D(budget)
            max_loss_d = D(max_loss)
        except Exception:
            return RedirectResponse("/?err=numbers", status_code=303)
        if budget_d <= 0 or max_loss_d <= 0:
            return RedirectResponse("/?err=limits", status_code=303)

        # Show validation snapshot in audit — live still requires credentials.
        validation = {
            "last_scan_ok": store.get_state().last_scan_ok,
            "connection_ok": store.get_state().connection_ok,
            "model": "weather.high_temp.v0.1-unvalidated",
            "warning": "Model is unvalidated; live trading can lose money. No profit promised.",
            "authenticated": pipeline.client.authenticated,
        }
        if not validation["authenticated"]:
            store.audit("live_enable_rejected", "API credentials not configured", details=validation)
            return RedirectResponse("/?err=auth", status_code=303)
        if not validation["last_scan_ok"] or not validation["connection_ok"]:
            store.audit("live_enable_rejected", "validation scans incomplete", details=validation)
            return RedirectResponse("/?err=validation", status_code=303)

        from kalshi_bot.data.store import utcnow

        store.update_state(
            live_enabled=True,
            trading_budget=str(budget_d),
            mode="live",
            live_ack_at=utcnow(),
        )
        # Persist risk limit update into runtime config object
        config.trading.budget_dollars = budget_d
        config.trading.max_loss_per_trade_dollars = max_loss_d
        store.audit(
            "live_enabled",
            "Live trading enabled by operator. Individual trades will not ask for approval.",
            level="warning",
            details=validation,
        )
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/set-budget")
    def set_budget(budget: str = Form(...)) -> RedirectResponse:
        """Explicit budget change only — deposits never auto-raise budget."""
        try:
            b = D(budget)
        except Exception:
            return RedirectResponse("/", status_code=303)
        store.update_state(trading_budget=str(b))
        config.trading.budget_dollars = b
        store.audit("budget", f"trading budget set to {b} (deposits do not auto-change this)")
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/close-position")
    def close_position(position_id: str = Form(...)) -> RedirectResponse:
        """Distinct from pause: mark intent to close; paper closes at mid if available else skip."""
        positions = store.list_positions(status="open")
        target = next((p for p in positions if p["id"] == position_id), None)
        if not target:
            store.audit("close_miss", f"position {position_id} not found")
            return RedirectResponse("/", status_code=303)
        # Closing is separate: we do not auto-flatten. Paper records a close request;
        # without a reliable exit model we refuse to invent exit fills.
        store.audit(
            "close_requested",
            f"Close requested for {target['market_ticker']} — "
            "early exit requires separate validated exit model; hold-to-settlement is default. "
            "Position left open.",
            details={"position_id": position_id},
        )
        return RedirectResponse("/", status_code=303)

    @app.get("/api/validation")
    def api_validation() -> dict[str, Any]:
        return pipeline.validation_status()

    @app.post("/actions/start-loop")
    def start_loop() -> RedirectResponse:
        if loop:
            loop.start()
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/stop-loop")
    def stop_loop() -> RedirectResponse:
        if loop:
            loop.stop()
        return RedirectResponse("/", status_code=303)

    return app
