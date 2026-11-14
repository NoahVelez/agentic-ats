"""FastAPI app with the MCP server mounted at ``/mcp`` (SPEC §3.1).

Run with: ``uv run python -m uvicorn ats.main:app``
"""

from __future__ import annotations

import hmac

from fastapi import FastAPI
from fastmcp import FastMCP
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ats import mcp_server
from ats.api import create_router
from ats.mcp_server import Runtime


class BearerAuth:
    """Require ``Authorization: Bearer <token>`` on every HTTP request to the wrapped app."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":
            # MCP over HTTP needs nothing else (no websockets), so anything else is refused.
            await send({"type": "websocket.close", "code": 1008})
            return
        scheme, _, given = Headers(scope=scope).get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(given.strip().encode(), self._token):
            await self._app(scope, receive, send)
            return
        response = JSONResponse(
            {"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
        )
        await response(scope, receive, send)


def create_app(runtime: Runtime | None = None, server: FastMCP | None = None) -> FastAPI:
    runtime = runtime or mcp_server.runtime
    server = server or mcp_server.mcp
    mcp_app = server.http_app(path="/")
    app = FastAPI(title="Agentic ATS", lifespan=mcp_app.lifespan)
    app.include_router(create_router(runtime), prefix="/api")
    key = runtime.settings.mcp_api_key
    token = key.get_secret_value() if key else ""
    app.mount("/mcp", BearerAuth(mcp_app, token) if token else mcp_app)
    return app


app = create_app()
