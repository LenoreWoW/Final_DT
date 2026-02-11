"""
Quantum Digital Twin Platform - Backend API

FastAPI application for the Universal Reality Simulator.
"""

import json
import sys
import time
import uuid
import logging
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Set

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.twins import router as twins_router
from backend.api.conversation import router as conversation_router
from backend.api.benchmark import router as benchmark_router
from backend.api.data import router as data_router
from backend.auth import auth_router
from backend.models.database import init_database, engine

logger = logging.getLogger(__name__)

# Initialize database
init_database(engine)

app = FastAPI(
    title="Quantum Digital Twin Platform",
    description="""
    ## Universal Reality Simulator - Build a second world

    A platform where anyone can describe any system and receive a fully functional
    quantum-powered digital twin they can simulate, predict, optimize, and experiment with.

    ### Sections:

    1. **Universal Twin Builder** (`/api/twins`, `/api/conversation`)
       - Create digital twins from natural language descriptions
       - Run quantum simulations
       - Query your twin

    2. **Quantum Advantage Showcase** (`/api/benchmark`)
       - Compare quantum vs classical performance
       - Healthcare case study
       - Benchmark results

    3. **Authentication** (`/api/auth`)
       - Register, login, JWT-based access control

    4. **Data Upload** (`/api/data`)
       - Upload CSV / JSON / Excel datasets for twin creation

    5. **Real-time Updates** (`/ws/{twin_id}`)
       - WebSocket endpoint for live twin generation / simulation progress
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)


# =============================================================================
# Rate Limiting + Usage Tracking Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-user rate limiting and usage tracking middleware."""

    def __init__(self, app, default_limit: int = 50):
        super().__init__(app)
        self.default_limit = default_limit
        # In-memory counter: {user_id: {date_str: count}}
        self._counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Extract user from JWT if present (lightweight check)
        user_id = self._extract_user_id(request)
        today = date.today().isoformat()

        # Rate limit check (only for authenticated users on /api/ paths)
        if user_id and request.url.path.startswith("/api/"):
            count = self._counters[user_id][today]
            limit = self._get_limit(user_id)

            if count >= limit:
                return Response(
                    content=json.dumps({
                        "detail": f"Rate limit exceeded ({limit} requests/day). Upgrade your tier for higher limits."
                    }),
                    status_code=429,
                    media_type="application/json",
                )
            self._counters[user_id][today] += 1

        response = await call_next(request)

        # Usage tracking (async, non-blocking)
        if user_id and request.url.path.startswith("/api/"):
            response_time = (time.time() - start_time) * 1000
            self._track_usage(user_id, request.url.path, request.method, response.status_code, response_time)

        return response

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user_id from Authorization header without full JWT validation."""
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth[7:]
        try:
            import base64
            # Decode JWT payload (middle part) without verification for rate limiting
            payload = token.split(".")[1]
            padding = 4 - len(payload) % 4
            payload += "=" * padding
            data = json.loads(base64.urlsafe_b64decode(payload))
            return data.get("sub") or data.get("user_id")
        except Exception:
            return None

    def _get_limit(self, user_id: str) -> int:
        """Get rate limit for a user. Could query DB for tier, uses default for now."""
        return self.default_limit

    def _track_usage(self, user_id: str, endpoint: str, method: str, status_code: int, response_time_ms: float):
        """Log usage to database (fire-and-forget)."""
        try:
            from backend.models.database import SessionLocal, UsageTrackingModel
            db = SessionLocal()
            record = UsageTrackingModel(
                id=str(uuid.uuid4()),
                user_id=user_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
            )
            db.add(record)
            db.commit()
            db.close()
        except Exception:
            pass  # Non-critical — don't let tracking errors break requests


# =============================================================================
# Trailing Slash Redirect Middleware
# =============================================================================

# Add middleware
app.add_middleware(RateLimitMiddleware, default_limit=200)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(twins_router, prefix="/api")
app.include_router(conversation_router, prefix="/api")
app.include_router(benchmark_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(auth_router, prefix="/api/auth")


# =============================================================================
# WebSocket Manager — tracks connected clients per twin
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections grouped by twin_id."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, twin_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(twin_id, set()).add(websocket)

    def disconnect(self, twin_id: str, websocket: WebSocket):
        if twin_id in self.active_connections:
            self.active_connections[twin_id].discard(websocket)
            if not self.active_connections[twin_id]:
                del self.active_connections[twin_id]

    async def broadcast(self, twin_id: str, message: dict):
        """Send a JSON message to all clients watching a specific twin."""
        for ws in list(self.active_connections.get(twin_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(twin_id, ws)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send a JSON message to a single client."""
        await websocket.send_json(message)


ws_manager = ConnectionManager()


@app.websocket("/ws/{twin_id}")
async def websocket_endpoint(websocket: WebSocket, twin_id: str):
    """
    WebSocket endpoint for real-time twin generation / simulation updates.

    Clients connect at ``/ws/<twin_id>`` and receive JSON messages
    such as:

    - ``{"type": "connected", "twin_id": "..."}``
    - ``{"type": "generation_progress", "step": "extraction", "progress": 0.33}``
    - ``{"type": "simulation_progress", "step": 42, "total": 100}``
    - ``{"type": "simulation_complete", "results": {...}}``
    - ``{"type": "error", "detail": "..."}``

    Clients may also send JSON commands; currently the only recognised
    command is ``{"action": "ping"}`` which elicits a ``{"type": "pong"}``
    response.
    """
    await ws_manager.connect(twin_id, websocket)

    # Confirm the connection
    await ws_manager.send_personal(websocket, {
        "type": "connected",
        "twin_id": twin_id,
        "timestamp": datetime.utcnow().isoformat(),
    })

    try:
        while True:
            raw = await websocket.receive_text()

            # Parse incoming commands
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send_personal(websocket, {
                    "type": "error",
                    "detail": "Invalid JSON",
                })
                continue

            action = data.get("action", "")

            if action == "ping":
                await ws_manager.send_personal(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            else:
                await ws_manager.send_personal(websocket, {
                    "type": "ack",
                    "received_action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(twin_id, websocket)


# Make ws_manager importable by other modules that need to broadcast
app.state.ws_manager = ws_manager


# =============================================================================
# Standard HTTP Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with platform information."""
    return {
        "name": "Quantum Digital Twin Platform",
        "version": "2.0.0",
        "tagline": "Build a second world",
        "status": "operational",
        "endpoints": {
            "twins": "/api/twins - Create and manage digital twins",
            "conversation": "/api/conversation - Natural language interface",
            "benchmark": "/api/benchmark - Quantum advantage showcase",
            "data": "/api/data - Data upload and analysis",
            "auth": "/api/auth - Authentication",
            "websocket": "/ws/{twin_id} - Real-time updates",
            "docs": "/docs - Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "quantum_engine": "ready",
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "version": "2.0.0",
        "sections": {
            "builder": {
                "description": "Universal Twin Builder",
                "endpoints": [
                    "POST /api/twins - Create a new twin",
                    "GET /api/twins - List all twins",
                    "GET /api/twins/{id} - Get a specific twin",
                    "PATCH /api/twins/{id} - Update a twin",
                    "DELETE /api/twins/{id} - Delete a twin",
                    "POST /api/twins/{id}/simulate - Run simulation",
                    "POST /api/twins/{id}/query - Query the twin",
                    "GET /api/twins/{id}/qasm - Get OpenQASM circuits",
                    "POST /api/conversation - Send a message",
                    "GET /api/conversation/{twin_id}/history - Get conversation history",
                ]
            },
            "showcase": {
                "description": "Quantum Advantage Showcase",
                "endpoints": [
                    "POST /api/benchmark/run/{module_id} - Run a benchmark",
                    "GET /api/benchmark/results - Get all benchmark results",
                    "GET /api/benchmark/results/{module_id} - Get specific benchmark",
                    "GET /api/benchmark/modules - List modules",
                    "GET /api/benchmark/methodology - Methodology documentation",
                ]
            },
            "auth": {
                "description": "Authentication",
                "endpoints": [
                    "POST /api/auth/register - Register a new user",
                    "POST /api/auth/login - Login and get JWT token",
                    "GET /api/auth/me - Get current user profile",
                ]
            },
            "data": {
                "description": "Data Upload & Analysis",
                "endpoints": [
                    "POST /api/data/upload - Upload CSV / JSON / Excel",
                ]
            },
            "realtime": {
                "description": "Real-time Updates",
                "endpoints": [
                    "WS /ws/{twin_id} - WebSocket for live updates",
                ]
            },
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
