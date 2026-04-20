"""WebSocket endpoint for real-time metric and flow streaming."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from omnilens.viz.live.config import settings

router = APIRouter()

# Set by the app when engine is initialized
_get_snapshot = None
_get_flow = None


def set_providers(snapshot_fn, flow_fn):
    global _get_snapshot, _get_flow
    _get_snapshot = snapshot_fn
    _get_flow = flow_fn


@router.websocket("/ws/metrics")
async def metrics_stream(websocket: WebSocket):
    await websocket.accept()
    interval_ms = settings.ws_interval_ms
    send_flow = True

    try:
        while True:
            # Check for config messages (non-blocking)
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.01
                )
                try:
                    config = json.loads(msg)
                    if "interval_ms" in config:
                        interval_ms = max(50, int(config["interval_ms"]))
                    if "send_flow" in config:
                        send_flow = bool(config["send_flow"])
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                pass

            # Send metric snapshot
            if _get_snapshot:
                snapshot = _get_snapshot()
                await websocket.send_json({
                    "type": "metrics",
                    "data": snapshot.model_dump(),
                })

            # Send flow data if enabled
            if send_flow and _get_flow:
                flow = _get_flow()
                if flow.frames:
                    await websocket.send_json({
                        "type": "flow",
                        "data": flow.model_dump(),
                    })

            await asyncio.sleep(interval_ms / 1000.0)

    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()
