"""REST endpoints for metric snapshots (polling fallback)."""

from fastapi import APIRouter, HTTPException

from omnilens.viz.live.models.metric_schema import MetricSnapshot

router = APIRouter(prefix="/api")

# Set by the app when engine is initialized
_get_snapshot = None


def set_snapshot_fn(fn):
    global _get_snapshot
    _get_snapshot = fn


@router.get("/metrics")
async def get_metrics() -> MetricSnapshot:
    if _get_snapshot is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    return _get_snapshot()
