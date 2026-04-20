"""FastAPI application entry point.

Serves the API, WebSocket, and static frontend files.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from omnilens.viz.live.config import settings
from omnilens.viz.live.api import routes_analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="OmniLens", version="0.2.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(routes_analysis.router)


# Health check
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve frontend static files
_frontend_dist = Path(__file__).parent / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = _frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dist / "index.html"))
