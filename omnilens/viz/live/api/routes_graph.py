"""REST endpoints for graph structure and model info."""

from fastapi import APIRouter, HTTPException

from omnilens.viz.live.models.graph_schema import GraphResponse
from omnilens.viz.live.models.session_schema import ModelInfo

router = APIRouter(prefix="/api")

# These get set by the app lifespan when a model is loaded
_graph_response: GraphResponse | None = None
_model_info: ModelInfo | None = None


def set_graph(graph: GraphResponse) -> None:
    global _graph_response
    _graph_response = graph


def set_model_info(info: ModelInfo) -> None:
    global _model_info
    _model_info = info


@router.get("/graph")
async def get_graph() -> GraphResponse:
    if _graph_response is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    return _graph_response


@router.get("/model")
async def get_model_info() -> ModelInfo:
    if _model_info is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    return _model_info
