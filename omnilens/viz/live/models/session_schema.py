from pydantic import BaseModel


class SessionConfig(BaseModel):
    metrics: list[str] = ["activation", "kl_divergence", "loss_attribution", "routing_importance"]
    ws_interval_ms: int = 200
    record_flow: bool = True


class ModelInfo(BaseModel):
    name: str
    total_params: int
    num_layers: int
    device: str
